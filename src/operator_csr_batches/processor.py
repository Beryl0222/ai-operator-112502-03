"""块注册中心 + 批处理器：装载复用、撤回/补交版本管理、失效评估与重算。

核心协议
--------
* 块的每次状态/声明集合变更都会令 ``epoch`` 加一，并重算全局
  ``effective_set_fingerprint``；
* 算术结果按 ``merge_fingerprint`` 缓存，已装载块的规范化结果直接复用，
  不重复解析/校验；
* :meth:`BatchProcessor.assess` 用注册中心当前状态无损重建每行的
  ``source_fingerprint``，判断旧候选分数是 ``reusable`` 还是
  ``must_recompute``；
* 变更后 :meth:`BatchProcessor.compute` / :meth:`recompute` 构造的都是
  **新的证明对象**；受影响行的 ``proof_id`` 必然改变，未受影响行的
  ``proof_id`` 因来源语句相同而保持稳定（旧证明仍可验证 -> 分数可继续用）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import functools
import threading
from typing import Any, Iterable, Iterator, Sequence

from .csr import CanonicalBlock, CsrBlock
from .declaration import Declaration
from .dense import coerce_rhs, rhs_digest
from .errors import UnknownBlockError, UnknownSampleError
from .hashing import digest
from .io import load_source
from .operator import (
    BlockView,
    OperatorConfig,
    SampleComputeInput,
    build_sample_result,
)
from .provenance import BlockBasis, compute_source_fingerprint
from .results import BatchResult, Candidate, RowResult
from .retention import DEFAULT_POLICY, RetentionPolicy, RetentionError
from .status import BlockStatus, ReasonCode, ScoreDecision


# ---------------------------------------------------------------------------
# 注册记录与回执
# ---------------------------------------------------------------------------


@dataclass
class SampleSpec:
    sample_id: str
    n_rows: int | None = None
    n_cols: int | None = None
    order_index: int = 0
    dimensions_explicit: bool = False


@dataclass
class BlockRecord:
    declaration: Declaration
    status: BlockStatus
    block: CsrBlock | None = None
    canonical: CanonicalBlock | None = None
    issues: tuple[str, ...] = ()
    digest: str | None = None
    revision: int = 1
    status_reason: str | None = None
    updated_epoch: int = 0


@dataclass(frozen=True)
class SubmissionReceipt:
    block_id: str
    sample_id: str
    status: str
    accepted: bool
    digest: str | None
    issues: tuple[str, ...] = ()
    epoch: int = 0
    revision: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "sample_id": self.sample_id,
            "status": self.status,
            "accepted": self.accepted,
            "digest": self.digest,
            "issues": list(self.issues),
            "epoch": self.epoch,
            "revision": self.revision,
        }


@dataclass(frozen=True)
class MutationReceipt:
    block_id: str
    action: str
    status: str
    epoch: int
    changed: bool = True
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "action": self.action,
            "status": self.status,
            "epoch": self.epoch,
            "changed": self.changed,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# 分数引用与重验证计划
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoreRef:
    """与具体对象解耦的分数身份句柄。"""

    sample_id: str
    row_index: int
    proof_id: str | None
    source_fingerprint: str | None
    merge_fingerprint: str | None = None
    score: float | None = None
    candidate_id: str | None = None


@dataclass(frozen=True)
class ScoreValidity:
    ref: ScoreRef
    decision: str
    reasons: tuple[str, ...]
    completeness: str
    item: Any = None

    @property
    def is_reusable(self) -> bool:
        return self.decision == ScoreDecision.REUSABLE

    @property
    def must_recompute(self) -> bool:
        return self.decision == ScoreDecision.MUST_RECOMPUTE

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref": {
                "sample_id": self.ref.sample_id,
                "row_index": self.ref.row_index,
                "proof_id": self.ref.proof_id,
                "source_fingerprint": self.ref.source_fingerprint,
                "merge_fingerprint": self.ref.merge_fingerprint,
                "score": self.ref.score,
                "candidate_id": self.ref.candidate_id,
            },
            "decision": self.decision,
            "reasons": list(self.reasons),
            "completeness": self.completeness,
        }


@dataclass
class RevalidationPlan:
    """旧分数在当前注册中心状态下的可用性计划。"""

    entries: tuple[ScoreValidity, ...]
    epoch: int
    effective_set_fingerprint: str

    @property
    def reusable(self) -> tuple[Any, ...]:
        return tuple(e.item for e in self.entries if e.is_reusable and e.item is not None)

    @property
    def must_recompute(self) -> tuple[Any, ...]:
        return tuple(e.item for e in self.entries if e.must_recompute and e.item is not None)

    # 显式别名
    stale = must_recompute
    invalid = must_recompute

    @property
    def valid(self) -> tuple[Any, ...]:
        return self.reusable

    @property
    def reusable_items(self) -> tuple[Any, ...]:
        return self.reusable

    @property
    def stale_items(self) -> tuple[Any, ...]:
        return self.must_recompute

    @property
    def reusable_candidates(self) -> tuple[Candidate, ...]:
        return tuple(e.item for e in self.entries if e.is_reusable and isinstance(e.item, Candidate))

    @property
    def stale_candidates(self) -> tuple[Candidate, ...]:
        return tuple(e.item for e in self.entries if e.must_recompute and isinstance(e.item, Candidate))

    @property
    def reusable_refs(self) -> tuple[ScoreRef, ...]:
        return tuple(e.ref for e in self.entries if e.is_reusable)

    @property
    def stale_refs(self) -> tuple[ScoreRef, ...]:
        return tuple(e.ref for e in self.entries if e.must_recompute)

    @property
    def affected_samples(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(e.ref.sample_id for e in self.entries if e.must_recompute))

    def __len__(self) -> int:
        return len(self.entries)

    def summary(self) -> dict[str, int]:
        reusable = sum(1 for e in self.entries if e.is_reusable)
        return {
            "total": len(self.entries),
            "reusable": reusable,
            "must_recompute": len(self.entries) - reusable,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "effective_set_fingerprint": self.effective_set_fingerprint,
            "summary": self.summary(),
            "entries": [e.to_dict() for e in self.entries],
        }


@dataclass
class ComputeReport:
    batch: BatchResult
    computed_keys: tuple[str, ...] = ()
    reused_keys: tuple[str, ...] = ()
    epoch: int = 0
    cache_size: int = 0

    @property
    def result(self) -> BatchResult:
        return self.batch

    @property
    def candidates(self) -> tuple[Candidate, ...]:
        return self.batch.candidates

    @property
    def cache_hits(self) -> int:
        return len(self.reused_keys)

    @property
    def cache_misses(self) -> int:
        return len(self.computed_keys)

    # 简短别名
    @property
    def hits(self) -> int:
        return self.cache_hits

    @property
    def misses(self) -> int:
        return self.cache_misses

    @property
    def cache_stats(self) -> dict[str, int]:
        return {
            "hits": len(self.reused_keys),
            "misses": len(self.computed_keys),
            "entries": self.cache_size,
        }

    def to_dict(self) -> dict[str, Any]:
        data = self.batch.to_dict()
        data["compute_report"] = {
            "computed_keys": list(self.computed_keys),
            "reused_keys": list(self.reused_keys),
            "cache_stats": self.cache_stats,
            "epoch": self.epoch,
        }
        return data


@dataclass
class RevalidationOutcome:
    """按计划执行重算后的结果：保留可用旧分数 + 替换失效分数。"""

    plan: RevalidationPlan
    report: ComputeReport
    kept: tuple[Any, ...]
    recomputed: tuple[Candidate, ...]
    replacements: dict[str, Candidate]

    def replacement_for(self, old: Candidate | ScoreRef | str) -> Candidate | None:
        if isinstance(old, str):
            key = old
        elif isinstance(old, ScoreRef):
            key = old.proof_id or ""
        else:
            key = old.proof_id or ""
        return self.replacements.get(key)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kept_count": len(self.kept),
            "recomputed_count": len(self.recomputed),
            "kept_proof_ids": [getattr(k, "proof_id", None) for k in self.kept],
            "recomputed": [c.to_dict() for c in self.recomputed],
            "replacements": {k: v.to_dict() for k, v in self.replacements.items()},
            "report": self.report.to_dict(),
        }


# ---------------------------------------------------------------------------
# BlockRegistry
# ---------------------------------------------------------------------------


def _locked(method: Any) -> Any:
    """串行化注册中心/处理器上的变更与计算调用；可重入锁允许嵌套转发。"""

    @functools.wraps(method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        lock = getattr(self, "_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._lock = lock
        with lock:
            return method(self, *args, **kwargs)

    return wrapper


class BlockRegistry:
    """声明、装载并跟踪 CSR 块生命周期的状态中心。"""

    def __init__(self, *, block_order: str = "block_id", rhs_digest_value: str | None = None):
        self._lock = threading.RLock()
        self.block_order = block_order
        self.rhs_digest_value = rhs_digest_value or ""
        self.samples: dict[str, SampleSpec] = {}
        self.declarations: dict[str, Declaration] = {}
        self.records: dict[str, BlockRecord] = {}
        self.history: dict[str, list[BlockRecord]] = {}
        self.epoch: int = 0
        self._sample_counter = 0
        self._decl_counter = 0
        self.effective_set_fingerprint: str = self._compute_set_fingerprint()

    # ---- 样本 -----------------------------------------------------------

    @_locked
    def add_sample(
        self,
        sample_id: str,
        *,
        n_rows: int | None = None,
        n_cols: int | None = None,
        explicit: bool | None = None,
    ) -> SampleSpec:
        sid = str(sample_id)
        if explicit is None:
            explicit = n_rows is not None or n_cols is not None
        spec = self.samples.get(sid)
        if spec is None:
            spec = SampleSpec(
                sid,
                n_rows=n_rows,
                n_cols=n_cols,
                order_index=self._sample_counter,
                dimensions_explicit=bool(explicit),
            )
            self._sample_counter += 1
            self.samples[sid] = spec
            self._bump()
        else:
            changed = False
            if spec.n_rows is None and n_rows is not None:
                spec.n_rows = n_rows
                changed = True
            if spec.n_cols is None and n_cols is not None:
                spec.n_cols = n_cols
                changed = True
            if explicit:
                spec.dimensions_explicit = True
            if changed:
                self._bump()
        return spec

    def _infer_sample_rows(self, block: CsrBlock) -> int:
        global_rows = block.global_rows()
        return (max(global_rows) + 1) if global_rows else block.n_rows

    # ---- 声明 -----------------------------------------------------------

    @_locked
    def declare(
        self,
        block_id: str,
        sample_id: str,
        *,
        row_start: int = 0,
        row_indices: Sequence[int] | None = None,
        rows: Sequence[int] | None = None,
        cover_all: bool = False,
        block_n_rows: int | None = None,
        required: bool = True,
    ) -> Declaration:
        bid = str(block_id)
        sid = str(sample_id)
        self.add_sample(sid)
        explicit_rows = tuple(row_indices) if row_indices is not None else (tuple(rows) if rows is not None else None)

        existing = self.declarations.get(bid)
        if existing is not None:
            if existing.sample_id != sid:
                raise ValueError(f"block {bid!r} already declared for sample {existing.sample_id!r}")
            return existing

        decl = Declaration(
            block_id=bid,
            sample_id=sid,
            order_index=self._decl_counter,
            row_start=int(row_start),
            row_indices=explicit_rows,
            cover_all=bool(cover_all),
            block_n_rows=block_n_rows,
            required=required,
        )
        self._decl_counter += 1
        self.declarations[bid] = decl
        self._bump()
        return decl

    # ---- 提交 / 拒绝 ----------------------------------------------------

    @_locked
    def submit(
        self,
        data: Any = None,
        *,
        block_id: str | None = None,
        sample_id: str | None = None,
        strict: bool = False,
        sample_n_cols: int | None = None,
        **kwargs: Any,
    ) -> SubmissionReceipt:
        block: CsrBlock
        issues: tuple[str, ...] = ()
        parse_issues: list[str] = []
        try:
            if isinstance(data, CsrBlock):
                block = data
                if block_id is not None and block.block_id != block_id:
                    block = CsrBlock(**{**block.__dict__, "block_id": str(block_id)})
            elif isinstance(data, dict):
                block = CsrBlock.from_dict(data, block_id=block_id, sample_id=sample_id)
            else:
                raise ValueError(f"unsupported submission type: {type(data).__name__}")
        except Exception as exc:  # noqa: BLE001 - 任何解析问题都转为 rejected 状态
            bid = str(block_id or (data.get("block_id") if isinstance(data, dict) else "unknown"))
            sid = str(sample_id or (data.get("sample_id") if isinstance(data, dict) else "unknown"))
            parse_issues = [f"{type(exc).__name__}: {exc}"]
            self.declare(bid, sid) if sid != "unknown" else None
            decl = self.declarations.get(bid) or Declaration(bid, sid)
            record = BlockRecord(
                declaration=decl,
                status=BlockStatus.REJECTED,
                issues=tuple(parse_issues),
                updated_epoch=self.epoch,
            )
            self.records[bid] = record
            self._bump()
            if strict:
                from .errors import BlockValidationError

                raise BlockValidationError(bid, parse_issues)
            return SubmissionReceipt(bid, sid, BlockStatus.REJECTED.value, False, None, tuple(parse_issues), self.epoch)

        bid = block.block_id
        sid = block.sample_id
        inferred_rows = self._infer_sample_rows(block)
        spec = self.add_sample(sid, n_rows=inferred_rows, n_cols=block.n_cols, explicit=False)
        if not spec.dimensions_explicit and spec.n_rows is not None and inferred_rows > spec.n_rows:
            spec.n_rows = inferred_rows
        decl = self.declarations.get(bid)
        if decl is None:
            decl = self.declare(
                bid,
                sid,
                row_start=block.row_start,
                row_indices=block.row_indices,
                block_n_rows=block.n_rows,
            )

        expected_cols = sample_n_cols if sample_n_cols is not None else spec.n_cols
        validation_issues = block.validate(spec.n_rows, expected_cols)
        if validation_issues:
            record = BlockRecord(
                declaration=decl,
                status=BlockStatus.REJECTED,
                block=block,
                canonical=None,
                issues=validation_issues,
                digest=block.content_digest,
                revision=block.revision,
                updated_epoch=self.epoch,
            )
            self.records[bid] = record
            self._bump()
            if strict:
                from .errors import BlockValidationError

                raise BlockValidationError(bid, validation_issues)
            return SubmissionReceipt(
                bid, sid, BlockStatus.REJECTED.value, False, block.content_digest,
                validation_issues, self.epoch, block.revision,
            )

        canonical = block.canonicalize()
        prior = self.records.get(bid)
        if prior is not None:
            self.history.setdefault(bid, []).append(prior)
        record = BlockRecord(
            declaration=decl,
            status=BlockStatus.ACCEPTED,
            block=block,
            canonical=canonical,
            issues=(),
            digest=block.content_digest,
            revision=block.revision,
            updated_epoch=self.epoch,
        )
        self.records[bid] = record
        self._bump()
        return SubmissionReceipt(
            bid, sid, BlockStatus.ACCEPTED.value, True, block.content_digest,
            (), self.epoch, block.revision,
        )

    @_locked
    def reject(self, block_id: str, reason: str = "rejected by authority") -> MutationReceipt:
        bid = str(block_id)
        decl = self.declarations.get(bid)
        if decl is None:
            raise UnknownBlockError(bid)
        prior = self.records.get(bid)
        record = BlockRecord(
            declaration=decl,
            status=BlockStatus.REJECTED,
            block=prior.block if prior else None,
            canonical=prior.canonical if prior else None,
            issues=(reason,),
            digest=prior.digest if prior else None,
            revision=prior.revision if prior else 1,
            status_reason=reason,
            updated_epoch=self.epoch + 1,
        )
        self.records[bid] = record
        self._bump()
        return MutationReceipt(bid, "reject", BlockStatus.REJECTED.value, self.epoch, True, reason)

    @_locked
    def withdraw(self, block_id: str, reason: str | None = None) -> MutationReceipt:
        bid = str(block_id)
        decl = self.declarations.get(bid)
        if decl is None:
            raise UnknownBlockError(bid)
        prior = self.records.get(bid)
        if prior is not None and prior.status == BlockStatus.WITHDRAWN:
            return MutationReceipt(bid, "withdraw", BlockStatus.WITHDRAWN.value, self.epoch, False, reason)
        record = BlockRecord(
            declaration=decl,
            status=BlockStatus.WITHDRAWN,
            block=prior.block if prior else None,
            canonical=prior.canonical if prior else None,
            issues=prior.issues if prior else (),
            digest=prior.digest if prior else None,
            revision=prior.revision if prior else 1,
            status_reason=reason,
            updated_epoch=self.epoch + 1,
        )
        self.records[bid] = record
        self._bump()
        return MutationReceipt(bid, "withdraw", BlockStatus.WITHDRAWN.value, self.epoch, True, reason)

    @_locked
    def supplement(self, data: Any = None, **kwargs: Any) -> SubmissionReceipt:
        """补交块或块的新修订版本。

        * 新 ``block_id``：等同于 :meth:`submit`；
        * 已存在的 ``block_id``：旧记录进入 history 并标记 superseded，
          新内容成为当前 accepted 记录。
        """
        bid = kwargs.get("block_id") or (data.get("block_id") if isinstance(data, dict) else None)
        prior = self.records.get(str(bid)) if bid else None
        receipt = self.submit(data, **kwargs)
        if receipt.accepted and prior is not None and bid is not None:
            # submit 已把 prior 放入 history；补充动作的语义在回执中体现
            pass
        return receipt

    @_locked
    def resubmit(self, data: Any = None, **kwargs: Any) -> SubmissionReceipt:
        return self.supplement(data, **kwargs)

    # ---- 查询 -----------------------------------------------------------

    def get(self, block_id: str) -> BlockRecord:
        if block_id not in self.records and block_id not in self.declarations:
            raise UnknownBlockError(block_id)
        if block_id not in self.records:
            # 已声明未提交
            return BlockRecord(self.declarations[block_id], BlockStatus.DECLARED, updated_epoch=self.epoch)
        return self.records[block_id]

    def status_of(self, block_id: str) -> str:
        return self.get(block_id).status.value

    def block_ids(self, sample_id: str | None = None) -> tuple[str, ...]:
        if sample_id is None:
            return tuple(self.declarations.keys())
        return tuple(bid for bid, d in self.declarations.items() if d.sample_id == sample_id)

    def effective_views(self, sample_id: str) -> dict[str, BlockView]:
        views: dict[str, BlockView] = {}
        for bid in self.block_ids(sample_id):
            rec = self.get(bid)
            views[bid] = BlockView(
                block_id=bid,
                status=rec.status.value if isinstance(rec.status, BlockStatus) else str(rec.status),
                digest=rec.digest,
                revision=rec.revision,
                block=rec.block,
                canonical=rec.canonical,
                issues=rec.issues,
            )
        return views

    def expected_block_ids_for_row(self, sample_id: str, row_index: int) -> tuple[str, ...]:
        spec = self.samples.get(sample_id)
        n_rows = spec.n_rows if spec else None
        result: list[str] = []
        for bid in self.block_ids(sample_id):
            decl = self.declarations[bid]
            rec = self.records.get(bid)
            block_n_rows = rec.block.n_rows if rec and rec.block else decl.block_n_rows
            rows = decl.resolve_global_rows(block_n_rows=block_n_rows, sample_n_rows=n_rows)
            if row_index in rows:
                result.append(bid)
        if self.block_order == "declaration":
            result.sort(key=lambda b: (self.declarations[b].order_index, b))
        else:
            result.sort()
        return tuple(result)

    # ---- 指纹 / epoch ---------------------------------------------------

    def _bump(self) -> None:
        self.epoch += 1
        self.effective_set_fingerprint = self._compute_set_fingerprint()

    def _compute_set_fingerprint(self) -> str:
        decl_obj = []
        for bid, d in sorted(self.declarations.items()):
            decl_obj.append(
                {
                    "block_id": bid,
                    "sample_id": d.sample_id,
                    "row_start": d.row_start,
                    "row_indices": list(d.row_indices) if d.row_indices else None,
                    "cover_all": d.cover_all,
                    "block_n_rows": d.block_n_rows,
                }
            )
        block_obj = []
        for bid, r in sorted(self.records.items()):
            block_obj.append([bid, r.status.value if isinstance(r.status, BlockStatus) else str(r.status), r.digest, r.revision])
        sample_obj = sorted(
            [sid, s.n_rows, s.n_cols] for sid, s in self.samples.items()
        )
        return digest({"samples": sample_obj, "declarations": decl_obj, "blocks": block_obj})

    # ---- 证明验证 -------------------------------------------------------

    def current_bases_for_row(self, sample_id: str, row_index: int) -> list[BlockBasis]:
        bases: list[BlockBasis] = []
        for bid in self.expected_block_ids_for_row(sample_id, row_index):
            decl = self.declarations[bid]
            rec = self.records.get(bid)
            if rec is None:
                status = BlockStatus.DECLARED.value
                digest_value = None
                revision = 1
                issues_fp = None
                issues: tuple[str, ...] = ()
            else:
                status = rec.status.value if isinstance(rec.status, BlockStatus) else str(rec.status)
                # 与算子保持同一规则：优先使用规范化归并摘要作为证明基础
                digest_value = rec.canonical.canonical_digest if rec.canonical is not None else rec.digest
                revision = rec.revision
                issues = rec.issues
                issues_fp = digest(list(issues)) if issues else None
            bases.append(
                BlockBasis(
                    block_id=bid,
                    status=status,
                    digest=digest_value,
                    revision=revision,
                    declaration_fingerprint=decl.declaration_fingerprint,
                    issues_fingerprint=issues_fp,
                )
            )
        return bases

    def verify_proof(self, proof: Any) -> tuple[bool, tuple[str, ...]]:
        """用当前注册中心状态重算来源指纹；返回是否仍成立及原因。"""
        reasons: list[str] = []
        spec = self.samples.get(proof.sample_id)
        if spec is None:
            return False, ("unknown_sample",)

        current_expected = self.expected_block_ids_for_row(proof.sample_id, proof.row_index)
        if tuple(proof.expected_blocks) != tuple(current_expected):
            new_blocks = [b for b in current_expected if b not in proof.expected_blocks]
            removed = [b for b in proof.expected_blocks if b not in current_expected]
            if new_blocks:
                reasons.append(f"{ReasonCode.NEW_BLOCK_DECLARED.value}:{','.join(new_blocks)}")
            if removed:
                reasons.append(f"{ReasonCode.EXPECTED_SET_CHANGED.value}:removed:{','.join(removed)}")
            if not reasons:
                reasons.append(ReasonCode.EXPECTED_SET_CHANGED.value)

        # 逐块给出可读原因
        for bid in proof.merged_blocks:
            rec = self.records.get(bid)
            if rec is None:
                reasons.append(f"{ReasonCode.BLOCK_WITHDRAWN.value}:{bid}")
                continue
            status = rec.status.value if isinstance(rec.status, BlockStatus) else str(rec.status)
            if status != BlockStatus.ACCEPTED.value:
                reasons.append(f"block_{status}:{bid}")
            elif any(d.startswith(f"{bid}@") for d in proof.block_digests):
                token = next(d for d in proof.block_digests if d.startswith(f"{bid}@"))
                old_short = token.split("@", 1)[1]
                if rec.digest and not rec.digest.startswith(old_short):
                    reasons.append(f"{ReasonCode.BLOCK_REVISED.value}:{bid}")

        if self.rhs_digest_value and proof.rhs_digest and self.rhs_digest_value != proof.rhs_digest:
            reasons.append(ReasonCode.RHS_CHANGED.value)

        current_fp = compute_source_fingerprint(
            operator_version=proof.operator_version,
            rhs_digest=proof.rhs_digest,  # RHS 变化已单独报告；这里比较块来源
            sample_id=proof.sample_id,
            row_index=proof.row_index,
            expected_bases=self.current_bases_for_row(proof.sample_id, proof.row_index),
        )
        # 若 RHS 变化，上面沿用 proof.rhs 以隔离 RHS 因素；最终仍需 RHS 一致。
        source_ok = current_fp == proof.source_fingerprint
        rhs_ok = not self.rhs_digest_value or self.rhs_digest_value == proof.rhs_digest
        if not source_ok and not reasons:
            reasons.append(ReasonCode.PROOF_MISMATCH.value)
        if source_ok and rhs_ok:
            return True, (ReasonCode.PROVENANCE_CURRENT.value,)
        return False, tuple(dict.fromkeys(reasons))


# ---------------------------------------------------------------------------
# BatchProcessor
# ---------------------------------------------------------------------------


class BatchProcessor:
    """面向调用方的批处理器：组合注册中心、RHS、缓存与重验证流程。"""

    def __init__(
        self,
        dense_rhs: Sequence[Sequence[float]] | None = None,
        *,
        rhs: Sequence[Sequence[float]] | None = None,
        config: OperatorConfig | None = None,
        default_retention: RetentionPolicy | dict[str, Any] | str | None = None,
        retention: RetentionPolicy | dict[str, Any] | str | None = None,
        retention_map: dict[str, RetentionPolicy] | None = None,
        strict_submit: bool = False,
    ):
        rhs_value = dense_rhs if dense_rhs is not None else rhs
        if rhs_value is None:
            raise ValueError("BatchProcessor requires dense_rhs")
        self.rhs = coerce_rhs(rhs_value)
        self.rhs_digest = rhs_digest(self.rhs)
        self.config = config or OperatorConfig()
        self.registry = BlockRegistry(block_order=self.config.block_order, rhs_digest_value=self.rhs_digest)
        self.strict_submit = strict_submit
        policy_value = default_retention if default_retention is not None else retention
        self.default_retention = (
            policy_value if isinstance(policy_value, RetentionPolicy) else RetentionPolicy.from_any(policy_value)
        )
        self.retention_map: dict[str, RetentionPolicy] = dict(retention_map or {})
        self.arith_cache: dict[str, tuple[float, ...]] = {}
        self._last_report: ComputeReport | None = None
        self._lock = threading.RLock()

    # -- 转发注册 API -----------------------------------------------------

    @property
    def epoch(self) -> int:
        return self.registry.epoch

    @property
    def effective_set_fingerprint(self) -> str:
        return self.registry.effective_set_fingerprint

    def __getattr__(self, name: str) -> Any:
        # 未在处理器层显式实现的注册中心 API 直接转发（status_of/get/...）
        registry = self.__dict__.get("registry")
        if registry is None:
            raise AttributeError(name)
        return getattr(registry, name)

    def add_sample(self, sample_id: str, **kwargs: Any) -> SampleSpec:
        return self.registry.add_sample(sample_id, **kwargs)

    def declare(self, block_id: str, sample_id: str, **kwargs: Any) -> Declaration:
        return self.registry.declare(block_id, sample_id, **kwargs)

    def submit(self, data: Any = None, **kwargs: Any) -> SubmissionReceipt:
        kwargs.setdefault("strict", self.strict_submit)
        kwargs.setdefault("sample_n_cols", len(self.rhs))
        return self.registry.submit(data, **kwargs)

    @_locked
    def load_block(self, data: Any = None, **kwargs: Any) -> SubmissionReceipt:
        return self.submit(data, **kwargs)

    def reject(self, block_id: str, reason: str = "rejected by authority") -> MutationReceipt:
        return self.registry.reject(block_id, reason)

    def withdraw(self, block_id: str, reason: str | None = None) -> MutationReceipt:
        return self.registry.withdraw(block_id, reason)

    def supplement(self, data: Any = None, **kwargs: Any) -> SubmissionReceipt:
        kwargs.setdefault("strict", self.strict_submit)
        kwargs.setdefault("sample_n_cols", len(self.rhs))
        return self.registry.supplement(data, **kwargs)

    def resubmit(self, data: Any = None, **kwargs: Any) -> SubmissionReceipt:
        return self.supplement(data, **kwargs)

    @_locked
    def load_source(self, source: Any = None, *, blocks: Iterable[Any] | None = None) -> list[str]:
        return load_source(self, source, blocks=blocks)

    @_locked
    def load_payload(self, payload: dict[str, Any]) -> list[str]:
        return self.load_source(payload)

    @_locked
    def set_retention(self, sample_id: str, policy: RetentionPolicy | dict[str, Any] | str) -> RetentionPolicy:
        resolved = policy if isinstance(policy, RetentionPolicy) else RetentionPolicy.from_any(policy)
        self.retention_map[str(sample_id)] = resolved
        return resolved

    @_locked
    def set_rhs(self, dense_rhs: Sequence[Sequence[float]]) -> str:
        """更换 RHS（会使所有行证明中的 rhs_digest 失效）。"""
        self.rhs = coerce_rhs(dense_rhs)
        self.rhs_digest = rhs_digest(self.rhs)
        self.registry.rhs_digest_value = self.rhs_digest
        self.registry._bump()
        return self.rhs_digest

    def verify_proof(self, proof: Any) -> tuple[bool, tuple[str, ...]]:
        return self.registry.verify_proof(proof)

    # ---- 计算 -----------------------------------------------------------

    def _sample_ids_ordered(self, sample_ids: Sequence[str] | None) -> list[str]:
        if sample_ids is not None:
            return [str(s) for s in sample_ids]
        ids = list(self.registry.samples.keys())
        if self.config.sample_order == "sample_id":
            ids.sort()
        else:
            ids.sort(key=lambda s: self.registry.samples[s].order_index)
        return ids

    @_locked
    def compute(
        self,
        sample_ids: Sequence[str] | None = None,
        *,
        use_cache: bool = True,
    ) -> ComputeReport:
        cache = self.arith_cache if use_cache else {}
        ordered_ids = self._sample_ids_ordered(sample_ids)

        results = []
        computed_keys: list[str] = []
        reused_keys: list[str] = []

        for order_index, sid in enumerate(ordered_ids):
            spec = self.registry.samples.get(sid)
            if spec is None:
                raise UnknownSampleError(sid)
            if spec.n_rows is None:
                # 没有任何块给出维度时无法计算
                raise ValueError(f"sample {sid!r} has no known n_rows; submit a block or call add_sample")

            policy = self.retention_map.get(sid, self.default_retention)
            inp = SampleComputeInput(
                sample_id=sid,
                n_rows=spec.n_rows,
                declarations=[d for d in self.registry.declarations.values() if d.sample_id == sid],
                views=self.registry.effective_views(sid),
                policy=policy,
            )
            sample_result = build_sample_result(
                inp,
                rhs=self.rhs,
                rhs_digest_value=self.rhs_digest,
                epoch=self.registry.epoch,
                effective_set_fingerprint=self.registry.effective_set_fingerprint,
                config=self.config,
                order_index=order_index,
                arith_cache=cache,
            )
            results.append(sample_result)
            for row in sample_result.rows:
                key = f"{sid}:{row.row_index}"
                if row.cache_status == ScoreDecision.REUSED_FROM_CACHE.value:
                    reused_keys.append(key)
                else:
                    computed_keys.append(key)

        batch = BatchResult(
            results=tuple(results),
            epoch=self.registry.epoch,
            effective_set_fingerprint=self.registry.effective_set_fingerprint,
            order=tuple(ordered_ids),
            source_description=f"BatchProcessor epoch={self.registry.epoch}",
        )
        report = ComputeReport(
            batch=batch,
            computed_keys=tuple(computed_keys),
            reused_keys=tuple(reused_keys),
            epoch=self.registry.epoch,
            cache_size=len(self.arith_cache),
        )
        self._last_report = report
        return report

    def run(self, *args: Any, **kwargs: Any) -> BatchResult:
        """兼容入口：``run(source)`` 装载并计算，或等同 :meth:`compute`。"""
        if args or "source" in kwargs:
            self.load_source(kwargs.pop("source", args[0] if args else None), blocks=kwargs.pop("blocks", None))
            args = args[1:]
        return self.compute(**kwargs).batch

    # 命名别名
    process = run
    process_batch = run

    __call__ = compute

    # ---- 评估 -----------------------------------------------------------

    def _to_ref_and_item(self, item: Any) -> tuple[ScoreRef, Any]:
        if isinstance(item, Candidate):
            p = item.proof
            return (
                ScoreRef(
                    item.sample_id, item.row_index, item.proof_id,
                    item.source_fingerprint,
                    p.merge_fingerprint if p else None,
                    item.score, item.candidate_id,
                ),
                item,
            )
        if isinstance(item, RowResult):
            p = item.proof
            return (
                ScoreRef(
                    item.sample_id, item.row_index, item.proof_id,
                    item.contribution.source_fingerprint,
                    p.merge_fingerprint if p else None,
                    item.score, f"{item.sample_id}:r{item.row_index}",
                ),
                item,
            )
        if isinstance(item, ScoreRef):
            return item, None
        raise TypeError(f"cannot assess object of type {type(item).__name__}")

    def _flatten_targets(self, targets: Any) -> list[Any]:
        if targets is None:
            if self._last_report is None:
                return []
            return list(self._last_report.batch.candidates)
        if isinstance(targets, (Candidate, RowResult, ScoreRef)):
            return [targets]
        if isinstance(targets, BatchResult):
            return list(targets.candidates)
        from .results import SampleResult

        if isinstance(targets, SampleResult):
            return list(targets.candidates)
        if isinstance(targets, ComputeReport):
            return list(targets.batch.candidates)
        if isinstance(targets, RevalidationPlan):
            return [e.item if e.item is not None else e.ref for e in targets.entries]
        if isinstance(targets, (list, tuple, set, frozenset)):
            out: list[Any] = []
            for t in targets:
                out.extend(self._flatten_targets(t))
            return out
        # 单次调用可能传入多个位置参数由 assess 处理；这里不认识
        raise TypeError(f"cannot assess targets of type {type(targets).__name__}")

    @_locked
    def assess(self, targets: Any = None, *more: Any) -> RevalidationPlan:
        """判断已生成的候选分数/行结果在当前块集合下是否仍可使用。"""
        items = self._flatten_targets(targets)
        for extra in more:
            items.extend(self._flatten_targets(extra))

        validity: list[ScoreValidity] = []
        for item in items:
            ref, original = self._to_ref_and_item(item) if not isinstance(item, ScoreRef) else (item, None)
            proof = getattr(original, "proof", None)
            if proof is None and original is not None:
                proof = getattr(getattr(original, "contribution", None), "proof", None)

            if proof is None or ref.source_fingerprint is None:
                validity.append(
                    ScoreValidity(
                        ref,
                        ScoreDecision.MUST_RECOMPUTE.value,
                        ("missing_proof",),
                        getattr(original, "completeness", "unknown"),
                        original,
                    )
                )
                continue

            ok, reasons = self.verify_proof(proof)
            decision = ScoreDecision.REUSABLE.value if ok else ScoreDecision.MUST_RECOMPUTE.value
            completeness = getattr(original, "completeness", proof.completeness)
            validity.append(ScoreValidity(ref, decision, tuple(reasons), completeness, original))

        return RevalidationPlan(
            entries=tuple(validity),
            epoch=self.registry.epoch,
            effective_set_fingerprint=self.registry.effective_set_fingerprint,
        )

    # ---- 重算 -----------------------------------------------------------

    @_locked
    def recompute(self, plan: RevalidationPlan | None = None) -> RevalidationOutcome:
        """执行计划：可用分数原样保留，失效分数重算并换发新证明。"""
        if plan is None:
            plan = self.assess()

        affected = plan.affected_samples
        report = self.compute(sample_ids=affected or None)

        new_candidates: dict[tuple[str, int], Candidate] = {}
        for cand in report.candidates:
            new_candidates[(cand.sample_id, cand.row_index)] = cand

        kept: list[Any] = []
        recomputed: list[Candidate] = []
        replacements: dict[str, Candidate] = {}

        for entry in plan.entries:
            old_key = entry.ref.proof_id or ""
            if entry.is_reusable:
                if entry.item is not None:
                    kept.append(entry.item)
                continue
            fresh = new_candidates.get((entry.ref.sample_id, entry.ref.row_index))
            if fresh is None:
                # 该行可能被保留策略过滤；直接从新批次行构造等价候选
                sample = report.batch.get(entry.ref.sample_id)
                row = sample.get_row(entry.ref.row_index)
                policy = self.retention_map.get(entry.ref.sample_id, self.default_retention)
                from .operator import _apply_retention

                cands, _ = _apply_retention([row], policy, rhs_width=sample.rhs_columns)
                fresh = cands[0] if cands else None  # type: ignore[assignment]
            if fresh is not None:
                recomputed.append(fresh)
                if old_key:
                    replacements[old_key] = fresh

        return RevalidationOutcome(
            plan=plan,
            report=report,
            kept=tuple(kept),
            recomputed=tuple(recomputed),
            replacements=replacements,
        )


# 兼容命名
BlockRegistryProcessor = BatchProcessor
BatchOperator = BatchProcessor
CSRBatchProcessor = BatchProcessor
Registry = BlockRegistry
