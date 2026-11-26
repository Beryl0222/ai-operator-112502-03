"""确定性 CSR × 稠密右矩阵批算子（核心计算，无 IO 副作用）。

累加契约
--------
对每个样本边界内的每一行，累加顺序被完全定义：

1. 块按 :class:`OperatorConfig.block_order` 排序（默认 ``block_id`` 升序，
   与线程到达顺序无关）；
2. 块内列按升序处理；重复列在 :meth:`CsrBlock.canonicalize` 中按原始相对
   顺序左折叠归并；
3. 对有序项序列中的每个 ``(column, merged_weight)``，依次执行
   ``acc[k] = acc[k] + merged_weight * rhs[column][k]``。

因此相同的有效块集合 + 相同 RHS 必然产生比特一致的浮点结果。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .csr import CanonicalBlock, CsrBlock
from .declaration import Declaration
from .hashing import digest
from .provenance import (
    OPERATOR_VERSION,
    BlockBasis,
    BlockContribution,
    ContributionSummary,
    MissingBlock,
    build_proof,
    compute_merge_fingerprint,
)
from .results import BatchResult, Candidate, DroppedRow, RetentionReport, RowResult, SampleResult
from .retention import (
    DEFAULT_POLICY,
    RetentionPolicy,
    policy_for_sample,
    project_scores,
    resolve_policy_map,
)
from .status import BlockStatus, Completeness, ReasonCode, RowStatus, ScoreDecision


@dataclass(frozen=True)
class OperatorConfig:
    """算子行为配置（全部参与证明/缓存语义）。"""

    block_order: str = "block_id"
    """``block_id``：按 ID 升序；``declaration``：按声明顺序。两者都确定。"""

    sample_order: str = "sample_id"
    """``sample_id``：样本按 ID 升序输出（默认，对乱序到达最稳健）；
    ``registration``：按注册顺序。"""

    operator_version: str = OPERATOR_VERSION

    def __post_init__(self) -> None:
        if self.block_order not in ("block_id", "declaration"):
            raise ValueError(f"unknown block_order: {self.block_order!r}")
        if self.sample_order not in ("sample_id", "registration"):
            raise ValueError(f"unknown sample_order: {self.sample_order!r}")


@dataclass(frozen=True)
class BlockView:
    """计算时刻某个块的只读有效状态。"""

    block_id: str
    status: str
    digest: str | None = None
    revision: int = 1
    block: CsrBlock | None = None
    canonical: CanonicalBlock | None = None
    issues: tuple[str, ...] = ()

    @property
    def basis_digest(self) -> str | None:
        """进入来源证明的摘要：优先使用规范化归并摘要。

        这样仅造成稳定重排的未排序输入不会产生不同的“归并证明”；
        原始内容摘要 :attr:`digest` 仍保留在贡献摘要中供审计。
        """
        if self.canonical is not None:
            return self.canonical.canonical_digest
        return self.digest


@dataclass
class SampleComputeInput:
    sample_id: str
    n_rows: int
    declarations: list[Declaration] = field(default_factory=list)
    views: dict[str, BlockView] = field(default_factory=dict)
    policy: RetentionPolicy = DEFAULT_POLICY


def block_sort_key(decl: Declaration, config: OperatorConfig) -> tuple[Any, ...]:
    if config.block_order == "declaration":
        return (decl.order_index, decl.block_id)
    return (decl.block_id,)


def _missing_descriptor(status: str, issues: Sequence[str]) -> tuple[str, ...]:
    if status == BlockStatus.REJECTED:
        primary = ReasonCode.BLOCK_REJECTED.value
    elif status == BlockStatus.WITHDRAWN:
        primary = ReasonCode.BLOCK_WITHDRAWN.value
    elif status == BlockStatus.SUPERSEDED:
        primary = ReasonCode.BLOCK_SUPERSEDED.value
    elif status == BlockStatus.DECLARED:
        primary = ReasonCode.DECLARED_AWAITING_DATA.value
    else:
        primary = f"block_{status}"
    extra = tuple(i for i in issues if i)
    return (primary, *extra)


def _structural_status(*, raw_nnz: int, duplicate: bool, unsorted: bool, explicit_zero: bool) -> RowStatus:
    if raw_nnz == 0:
        return RowStatus.ZERO_ROW
    if duplicate:
        return RowStatus.DUPLICATE_COLUMNS_MERGED
    if unsorted:
        return RowStatus.INDICES_UNSORTED
    if explicit_zero:
        return RowStatus.EXPLICIT_ZERO
    return RowStatus.OK


def _dominant_status(completeness: str, missing: Sequence[MissingBlock], structural: RowStatus) -> RowStatus:
    if completeness == Completeness.UNSOURCED:
        return RowStatus.UNSOURCED
    if completeness == Completeness.INCOMPLETE:
        statuses = {m.status for m in missing}
        if any(s == BlockStatus.REJECTED for s in statuses):
            return RowStatus.BLOCK_REJECTED
        if any(s == BlockStatus.WITHDRAWN for s in statuses):
            return RowStatus.BLOCK_WITHDRAWN
        if any(s == BlockStatus.SUPERSEDED for s in statuses):
            return RowStatus.BLOCK_SUPERSEDED
        return RowStatus.BLOCK_PENDING
    return structural


def build_sample_result(
    inp: SampleComputeInput,
    *,
    rhs: tuple[tuple[float, ...], ...],
    rhs_digest_value: str,
    epoch: int,
    effective_set_fingerprint: str,
    config: OperatorConfig,
    order_index: int,
    arith_cache: dict[str, tuple[float, ...]] | None = None,
) -> SampleResult:
    """按样本边界计算一个样本的全部行结果、证明与候选。"""

    sample_id = inp.sample_id
    n_rows = inp.n_rows
    k_cols = len(rhs[0]) if rhs else 0
    cache = arith_cache if arith_cache is not None else {}

    ordered_decls = sorted(inp.declarations, key=lambda d: block_sort_key(d, config))

    row_results: list[RowResult] = []

    for row_index in range(n_rows):
        # ---- 该行期望的声明块（按覆盖关系过滤） -------------------------
        expected_decls: list[Declaration] = []
        coverage_cache: dict[str, tuple[int, ...]] = {}
        for decl in ordered_decls:
            view = inp.views.get(decl.block_id)
            block_n_rows = view.block.n_rows if view and view.block else decl.block_n_rows
            rows = decl.resolve_global_rows(
                block_n_rows=block_n_rows,
                sample_n_rows=n_rows,
            )
            coverage_cache[decl.block_id] = rows
            if row_index in rows:
                expected_decls.append(decl)
        expected_decls.sort(key=lambda d: block_sort_key(d, config))

        bases: list[BlockBasis] = []
        missing: list[MissingBlock] = []
        contributions: list[BlockContribution] = []
        merged_blocks: list[str] = []
        block_digests: list[str] = []

        ordered_terms: list[list[Any]] = []
        raw_ops: list[tuple[int, float]] = []  # (column, merged_weight) 与 terms 对齐

        total_raw_nnz = 0
        duplicate_seen = False
        unsorted_seen = False
        explicit_zero_count = 0
        duplicate_columns: list[int] = []
        any_accepted_cover = False

        for merge_order, decl in enumerate(expected_decls):
            view = inp.views.get(decl.block_id)
            status = view.status if view is not None else BlockStatus.DECLARED.value
            block_digest = view.basis_digest if view is not None else None
            raw_digest = view.digest if view is not None else None
            revision = view.revision if view is not None else 1
            issues = view.issues if view is not None else ()
            issues_fp = digest(list(issues)) if issues else None

            bases.append(
                BlockBasis(
                    block_id=decl.block_id,
                    status=str(status),
                    digest=block_digest,
                    revision=revision,
                    declaration_fingerprint=decl.declaration_fingerprint,
                    issues_fingerprint=issues_fp,
                )
            )

            if status != BlockStatus.ACCEPTED or view is None or view.canonical is None or view.block is None:
                missing.append(
                    MissingBlock(
                        block_id=decl.block_id,
                        status=str(status),
                        reasons=_missing_descriptor(str(status), issues),
                        digest=raw_digest,
                    )
                )
                continue

            global_rows = coverage_cache[decl.block_id]
            local_row = global_rows.index(row_index)
            crow = view.canonical.rows[local_row]
            any_accepted_cover = True

            total_raw_nnz += crow.raw_nnz
            duplicate_seen = duplicate_seen or crow.had_duplicates
            unsorted_seen = unsorted_seen or crow.was_unsorted
            if crow.had_duplicates:
                duplicate_columns.extend(e.column for e in crow.entries if e.multiplicity > 1)

            # 显式 0 值计数（按原始存储条目，区别于结构空行）
            start, end = view.block.indptr[local_row], view.block.indptr[local_row + 1]
            row_explicit_zeros = sum(1 for v in view.block.values[start:end] if v == 0.0)
            explicit_zero_count += row_explicit_zeros

            contributions.append(
                BlockContribution(
                    block_id=decl.block_id,
                    status=BlockStatus.ACCEPTED.value,
                    digest=view.digest,
                    revision=view.revision,
                    merge_order=merge_order,
                    local_row=local_row,
                    global_row=row_index,
                    raw_nnz=crow.raw_nnz,
                    merged_nnz=crow.merged_nnz,
                    duplicate_columns=tuple(e.column for e in crow.entries if e.multiplicity > 1),
                    unsorted=crow.was_unsorted,
                    explicit_zero_entries=row_explicit_zeros,
                )
            )

            merged_blocks.append(decl.block_id)
            block_digests.append(f"{decl.block_id}@{(view.digest or 'none')[:16]}")
            for entry in crow.entries:
                ordered_terms.append(entry.to_term())
                raw_ops.append((entry.column, entry.value))

        # ---- 完整性 -----------------------------------------------------
        if not expected_decls:
            completeness = Completeness.UNSOURCED.value
        elif missing:
            completeness = Completeness.INCOMPLETE.value
        else:
            completeness = Completeness.COMPLETE.value

        # ---- 指纹 + 算术（带缓存） --------------------------------------
        merge_fp = compute_merge_fingerprint(
            operator_version=config.operator_version,
            rhs_digest=rhs_digest_value,
            sample_id=sample_id,
            row_index=row_index,
            ordered_terms=ordered_terms,
        )

        cache_status = ScoreDecision.FRESH.value
        scores_t = cache.get(merge_fp)
        if scores_t is not None:
            cache_status = ScoreDecision.REUSED_FROM_CACHE.value
        else:
            acc = [0.0 for _ in range(k_cols)]
            for column, weight in raw_ops:
                rhs_row = rhs[column]
                for k in range(k_cols):
                    acc[k] = acc[k] + weight * rhs_row[k]
            scores_t = tuple(acc)
            cache[merge_fp] = scores_t

        # ---- 状态标记（四个概念严格区分） -------------------------------
        flags: list[str] = []
        reasons: list[str] = []

        structural = _structural_status(
            raw_nnz=total_raw_nnz,
            duplicate=duplicate_seen,
            unsorted=unsorted_seen,
            explicit_zero=explicit_zero_count > 0,
        )

        if completeness == Completeness.COMPLETE:
            if structural == RowStatus.ZERO_ROW:
                flags.append(RowStatus.ZERO_ROW)
                reasons.append(ReasonCode.ZERO_ROW.value)
            if duplicate_seen:
                flags.append(RowStatus.DUPLICATE_COLUMNS_MERGED)
                reasons.append(ReasonCode.DUPLICATE_COLUMNS.value)
            if unsorted_seen:
                flags.append(RowStatus.INDICES_UNSORTED)
                reasons.append(ReasonCode.UNSORTED_INDICES.value)
            if explicit_zero_count > 0:
                flags.append(RowStatus.EXPLICIT_ZERO)
                reasons.append(ReasonCode.EXPLICIT_ZERO_VALUE.value)
        else:
            # 来源不完整时绝不把该行称为 zero_row。
            if any_accepted_cover and total_raw_nnz == 0:
                reasons.append(ReasonCode.ACCEPTED_SOURCES_EMPTY.value)
            if duplicate_seen:
                flags.append(RowStatus.DUPLICATE_COLUMNS_MERGED)
                reasons.append(ReasonCode.DUPLICATE_COLUMNS.value)
            if unsorted_seen:
                flags.append(RowStatus.INDICES_UNSORTED)
                reasons.append(ReasonCode.UNSORTED_INDICES.value)

        for mb in missing:
            reasons.extend(r for r in mb.reasons if r not in reasons)

        if all(s == 0.0 for s in scores_t):
            reasons.append(ReasonCode.RESULT_ZERO.value)

        dominant = _dominant_status(completeness, missing, structural)

        proof = build_proof(
            sample_id=sample_id,
            row_index=row_index,
            epoch=epoch,
            effective_set_fingerprint=effective_set_fingerprint,
            rhs_digest_value=rhs_digest_value,
            expected_bases=bases,
            merged_blocks=merged_blocks,
            block_digests=block_digests,
            merge_fingerprint=merge_fp,
            completeness=completeness,
            issued_cache_status=cache_status,
            operator_version=config.operator_version,
        )

        completeness_enum = Completeness(completeness)

        summary = ContributionSummary(
            sample_id=sample_id,
            row_index=row_index,
            completeness=completeness_enum,
            epoch=epoch,
            effective_set_fingerprint=effective_set_fingerprint,
            proof=proof,
            declared_blocks=tuple(d.block_id for d in inp.declarations),
            expected_blocks=tuple(d.block_id for d in expected_decls),
            contributed_blocks=tuple(contributions),
            missing_blocks=tuple(missing),
            rhs_digest=rhs_digest_value,
            operator_version=config.operator_version,
        )

        row_result = RowResult(
            sample_id=sample_id,
            row_index=row_index,
            scores=scores_t,
            completeness=completeness_enum,
            row_status=dominant,
            structural_status=structural,
            flags=tuple(sorted(set(flags), key=lambda member: member.value)),
            reasons=tuple(dict.fromkeys(reasons)),
            nnz_raw=total_raw_nnz,
            nnz_merged=len(raw_ops),
            contribution=summary,
            cache_status=cache_status,
        )
        row_results.append(row_result)

    # ---- 样本级贡献摘要 --------------------------------------------------
    sample_missing: list[MissingBlock] = []
    sample_contribs: list[BlockContribution] = []
    all_decl_ids: list[str] = []
    for order, decl in enumerate(ordered_decls):
        all_decl_ids.append(decl.block_id)
        view = inp.views.get(decl.block_id)
        status = view.status if view else BlockStatus.DECLARED.value
        if status == BlockStatus.ACCEPTED and view and view.canonical:
            total_nnz = sum(r.raw_nnz for r in view.canonical.rows)
            sample_contribs.append(
                BlockContribution(
                    block_id=decl.block_id,
                    status=BlockStatus.ACCEPTED.value,
                    digest=view.digest,
                    revision=view.revision if view else 1,
                    merge_order=order,
                    global_row=None,
                    raw_nnz=total_nnz,
                    merged_nnz=sum(r.merged_nnz for r in view.canonical.rows),
                )
            )
        else:
            issues = view.issues if view else ()
            sample_missing.append(
                MissingBlock(
                    block_id=decl.block_id,
                    status=str(status),
                    reasons=_missing_descriptor(str(status), issues),
                    digest=view.digest if view else None,
                )
            )

    sample_completeness = (
        Completeness.UNSOURCED
        if not ordered_decls
        else (Completeness.COMPLETE if not sample_missing else Completeness.INCOMPLETE)
    )
    sample_summary = ContributionSummary(
        sample_id=sample_id,
        row_index=None,
        completeness=sample_completeness,
        epoch=epoch,
        effective_set_fingerprint=effective_set_fingerprint,
        proof=None,
        declared_blocks=tuple(all_decl_ids),
        expected_blocks=tuple(all_decl_ids),
        contributed_blocks=tuple(sample_contribs),
        missing_blocks=tuple(sample_missing),
        rhs_digest=rhs_digest_value,
        operator_version=config.operator_version,
    )

    # ---- 保留策略 -> 候选 -----------------------------------------------
    candidates, retention_report = _apply_retention(
        row_results, inp.policy, rhs_width=k_cols
    )

    return SampleResult(
        sample_id=sample_id,
        n_rows=n_rows,
        n_cols=inp.declarations and _safe_n_cols(inp.views) or 0,
        rhs_columns=k_cols,
        rows=tuple(row_results),
        candidates=tuple(candidates),
        contribution=sample_summary,
        retention=retention_report,
        order_index=order_index,
    )


def _safe_n_cols(views: Mapping[str, BlockView]) -> int:
    for v in views.values():
        if v.block is not None:
            return v.block.n_cols
    return 0


def _apply_retention(
    rows: Sequence[RowResult],
    policy: RetentionPolicy,
    *,
    rhs_width: int,
) -> tuple[list[Candidate], RetentionReport]:
    if policy.score_column >= rhs_width:
        from .errors import RetentionError

        raise RetentionError(
            f"score_column {policy.score_column} out of range for RHS width {rhs_width}"
        )
    if policy.columns:
        bad = [c for c in policy.columns if c >= rhs_width or c < 0]
        if bad:
            from .errors import RetentionError

            raise RetentionError(f"output columns out of range [0,{rhs_width}): {bad}")

    dropped: list[DroppedRow] = []
    eligible: list[RowResult] = []

    for row in rows:
        primary = row.scores[policy.score_column]
        if policy.only_complete and not row.is_complete:
            dropped.append(DroppedRow(row.row_index, "incomplete_source", primary, row.completeness))
            continue
        if policy.min_score is not None and primary < policy.min_score:
            dropped.append(DroppedRow(row.row_index, "below_min_score", primary, row.completeness))
            continue
        if policy.mode == "nonzero" and all(s == 0.0 for s in row.scores):
            dropped.append(DroppedRow(row.row_index, "all_zero_scores", primary, row.completeness))
            continue
        eligible.append(row)

    if policy.mode == "top_k":
        ranked = sorted(eligible, key=lambda r: (-r.scores[policy.score_column], r.row_index))
        kept = ranked[: policy.top_k]
        kept_ids = {r.row_index for r in kept}
        for row in eligible:
            if row.row_index not in kept_ids:
                dropped.append(
                    DroppedRow(
                        row.row_index,
                        "rank_below_top_k",
                        row.scores[policy.score_column],
                        row.completeness,
                    )
                )
    else:
        kept = eligible

    candidates: list[Candidate] = []
    retained_rows: list[int] = []
    for rank, row in enumerate(kept):
        retained_rows.append(row.row_index)
        candidates.append(
            Candidate(
                candidate_id=f"{row.sample_id}:r{row.row_index}",
                sample_id=row.sample_id,
                row_index=row.row_index,
                score=row.scores[policy.score_column],
                scores=project_scores(row.scores, policy),
                contribution=row.contribution,
                rank=rank,
                completeness=row.completeness,
                row_status=row.row_status,
                flags=row.flags,
                policy=policy.descriptor(),
            )
        )

    report = RetentionReport(
        policy=policy.descriptor(),
        retained_rows=tuple(retained_rows),
        dropped=tuple(dropped),
    )
    return candidates, report


# ---------------------------------------------------------------------------
# 面向调用方的无状态门面（内部使用 BatchProcessor 获得完整状态能力）
# ---------------------------------------------------------------------------


class SparseBatchMatMul:
    """可复用的稀疏批矩阵乘算子。

    可以在构造时固定 RHS/配置，也可以在 :meth:`compute` 时传入。
    无生命周期需求时直接使用模块函数 :func:`sparse_batch_matmul`。
    """

    def __init__(
        self,
        dense_rhs: Sequence[Sequence[float]] | None = None,
        *,
        rhs: Sequence[Sequence[float]] | None = None,
        config: OperatorConfig | None = None,
        retention: RetentionPolicy | Mapping[str, Any] | None = None,
    ):
        from .dense import coerce_rhs, rhs_digest as _rhs_digest

        rhs_value = dense_rhs if dense_rhs is not None else rhs
        self.rhs = coerce_rhs(rhs_value) if rhs_value is not None else None
        self.rhs_digest = _rhs_digest(self.rhs) if self.rhs is not None else ""
        self.config = config or OperatorConfig()
        self.retention = retention

    def compute(
        self,
        batches: Any = None,
        dense_rhs: Sequence[Sequence[float]] | None = None,
        *,
        source: Any = None,
        rhs: Sequence[Sequence[float]] | None = None,
        blocks: Sequence[Any] | None = None,
        retention: Any = None,
        policies: Mapping[str, Any] | None = None,
    ) -> BatchResult:
        from .processor import BatchProcessor

        src = source if source is not None else batches
        rhs_value = dense_rhs if dense_rhs is not None else rhs
        rhs_value = rhs_value if rhs_value is not None else self.rhs
        if rhs_value is None and isinstance(src, dict) and src.get("dense_rhs") is not None:
            rhs_value = src["dense_rhs"]
        if rhs_value is None:
            raise ValueError("dense_rhs must be provided to compute")

        policy_map = resolve_policy_map(policies if policies is not None else (retention if retention is not None else self.retention))
        default_policy = policy_map.get("*", DEFAULT_POLICY)

        processor = BatchProcessor(
            rhs_value,
            config=self.config,
            default_retention=default_policy,
            retention_map={k: v for k, v in policy_map.items() if k != "*"},
        )
        processor.load_source(src, blocks=blocks)
        report = processor.compute()
        return report.batch

    # 命名别名
    execute = compute
    run = compute

    __call__ = compute


def sparse_batch_matmul(
    batches: Any = None,
    dense_rhs: Sequence[Sequence[float]] | None = None,
    *args: Any,
    source: Any = None,
    rhs: Sequence[Sequence[float]] | None = None,
    blocks: Sequence[Any] | None = None,
    csr_blocks: Sequence[Any] | None = None,
    dense_matrix: Sequence[Sequence[float]] | None = None,
    retention: Any = None,
    policies: Mapping[str, Any] | None = None,
    config: OperatorConfig | None = None,
    **kwargs: Any,
) -> BatchResult:
    """函数式入口：CSR 批次 × 稠密 RHS，返回 :class:`BatchResult`。

    支持以下调用形态：

    * ``sparse_batch_matmul(payload)`` —— 读取 ``{"batches": ..., "dense_rhs": ...}``；
    * ``sparse_batch_matmul(batches, rhs)``；
    * ``sparse_batch_matmul(blocks=..., dense_matrix=...)``。

    非法 CSR 块不会导致整体失败：会被标记为 rejected，对应输出为
    incomplete；如需抛异常可自行先调用 ``block.require_valid()`` 或使用
    :class:`BatchProcessor` 的严格提交模式。
    """
    rhs_value = dense_rhs if dense_rhs is not None else rhs
    if rhs_value is None:
        rhs_value = dense_matrix
    if rhs_value is None and isinstance(batches, dict) and "dense_rhs" in batches:
        rhs_value = batches["dense_rhs"]

    block_value = blocks if blocks is not None else csr_blocks

    op = SparseBatchMatMul(rhs_value, config=config, retention=retention)
    src = source if source is not None else batches
    if block_value is not None and src is None:
        src = block_value
    return op.compute(src, policies=policies, retention=retention, blocks=block_value if isinstance(block_value, list) else None)


# 常见命名别名
batch_csr_matmul = sparse_batch_matmul
csr_batch_matmul = sparse_batch_matmul
sparse_batched_matmul = sparse_batch_matmul
sparse_csr_batch_matmul = sparse_batch_matmul
csr_matmul = sparse_batch_matmul
batch_matmul = sparse_batch_matmul
matmul_csr_dense = sparse_batch_matmul

SparseBatchedMatMul = SparseBatchMatMul
SparseCSRBatchMatMul = SparseBatchMatMul
CSRSparseBatchMatMul = SparseBatchMatMul
CSRBatchMatMul = SparseBatchMatMul
BatchCSRMatMul = SparseBatchMatMul
CSRBatchOperator = SparseBatchMatMul
SparseBatchMatMulOperator = SparseBatchMatMul

# 动词风格别名
compute_batch = sparse_batch_matmul
run_batch_matmul = sparse_batch_matmul
