"""算子结果对象：行结果、候选分数、样本结果与批次结果。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

from .hashing import canonical_text
from .provenance import ContributionSummary, MergeProof
from .status import Completeness, RowStatus


@dataclass(frozen=True)
class RowResult:
    """单个样本输出行（一个候选位置）的完整计算结果与状态。"""

    sample_id: str
    row_index: int
    scores: tuple[float, ...]
    completeness: str
    row_status: str
    flags: tuple[str, ...]
    reasons: tuple[str, ...]
    nnz_raw: int
    nnz_merged: int
    contribution: ContributionSummary
    cache_status: str = "fresh"
    structural_status: str = "ok"

    # -- 常用布尔/别名 ----------------------------------------------------

    @property
    def status(self) -> str:
        """主状态：来源阻断优先，其次结构状态（等价于 :attr:`row_status`）。"""
        return self.row_status

    @property
    def state(self) -> str:
        """兼容命名：主状态。"""
        return self.row_status

    @property
    def status_code(self) -> str:
        """兼容命名：主状态码。"""
        return self.row_status

    @property
    def structural_state(self) -> str:
        """纯结构/规范化状态（``zero_row`` / ``duplicate_...`` / ...）。"""
        return self.structural_status

    @property
    def kind(self) -> str:
        """行类别（结构状态）。"""
        return self.structural_status

    @property
    def is_zero_row(self) -> bool:
        return RowStatus.ZERO_ROW in self.flags or self.row_status == RowStatus.ZERO_ROW

    @property
    def has_duplicate_columns(self) -> bool:
        return RowStatus.DUPLICATE_COLUMNS_MERGED in self.flags

    @property
    def is_unsorted(self) -> bool:
        return RowStatus.INDICES_UNSORTED in self.flags

    @property
    def is_complete(self) -> bool:
        return self.completeness == Completeness.COMPLETE

    @property
    def all_blocks_contributed(self) -> bool:
        return self.is_complete

    @property
    def proof(self) -> MergeProof | None:
        return self.contribution.proof

    @property
    def proof_id(self) -> str | None:
        return self.contribution.proof_id

    @property
    def score(self) -> float:
        """主分数（第 0 列），便于标量候选场景。"""
        return self.scores[0] if self.scores else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "row_index": self.row_index,
            "scores": list(self.scores),
            "score": self.score,
            "completeness": self.completeness,
            "complete": self.is_complete,
            "all_blocks_contributed": self.all_blocks_contributed,
            "row_status": self.row_status,
            "status": self.row_status,
            "structural_status": self.structural_status,
            "flags": list(self.flags),
            "reasons": list(self.reasons),
            "nnz_raw": self.nnz_raw,
            "nnz_merged": self.nnz_merged,
            "cache_status": self.cache_status,
            "proof_id": self.proof_id,
            "contribution": self.contribution.to_dict(),
        }


@dataclass(frozen=True)
class Candidate:
    """传递给候选筛选/排序的分数记录，强制携带样本身份与贡献摘要。"""

    candidate_id: str
    sample_id: str
    row_index: int
    score: float
    scores: tuple[float, ...]
    contribution: ContributionSummary
    rank: int | None = None
    completeness: str = Completeness.COMPLETE.value
    row_status: str = RowStatus.OK.value
    flags: tuple[str, ...] = ()
    policy: dict[str, Any] = field(default_factory=dict)

    @property
    def proof(self) -> MergeProof | None:
        return self.contribution.proof

    @property
    def proof_id(self) -> str | None:
        return self.contribution.proof_id

    @property
    def source_fingerprint(self) -> str | None:
        return self.contribution.source_fingerprint

    @property
    def provenance_key(self) -> tuple[str, int | None, str | None]:
        return self.contribution.provenance_key

    @property
    def source_key(self) -> tuple[str, str | None]:
        """粗粒度来源键：``(sample_id, source_fingerprint)``。"""
        return (self.sample_id, self.contribution.source_fingerprint)

    @property
    def is_complete(self) -> bool:
        return self.completeness == Completeness.COMPLETE

    @property
    def all_blocks_contributed(self) -> bool:
        return self.contribution.all_blocks_contributed

    @property
    def status(self) -> str:
        return self.row_status

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "sample_id": self.sample_id,
            "row_index": self.row_index,
            "rank": self.rank,
            "score": self.score,
            "scores": list(self.scores),
            "completeness": self.completeness,
            "complete": self.is_complete,
            "all_blocks_contributed": self.all_blocks_contributed,
            "row_status": self.row_status,
            "status": self.row_status,
            "flags": list(self.flags),
            "policy": self.policy,
            "proof_id": self.proof_id,
            "source_fingerprint": self.source_fingerprint,
            "provenance_key": list(self.provenance_key),
            "contribution": self.contribution.to_dict(),
        }


@dataclass(frozen=True)
class DroppedRow:
    row_index: int
    reason: str
    score: float | None
    completeness: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_index": self.row_index,
            "reason": self.reason,
            "score": self.score,
            "completeness": self.completeness,
        }


@dataclass(frozen=True)
class RetentionReport:
    policy: dict[str, Any]
    retained_rows: tuple[int, ...]
    dropped: tuple[DroppedRow, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "retained_rows": list(self.retained_rows),
            "dropped": [d.to_dict() for d in self.dropped],
        }


@dataclass(frozen=True)
class SampleResult:
    sample_id: str
    n_rows: int
    n_cols: int
    rhs_columns: int
    rows: tuple[RowResult, ...]
    candidates: tuple[Candidate, ...]
    contribution: ContributionSummary
    retention: RetentionReport | None = None
    order_index: int = 0

    # -- 访问 -------------------------------------------------------------

    def get_row(self, row_index: int) -> RowResult:
        return self.rows[row_index]

    def __getitem__(self, row_index: int) -> RowResult:
        return self.rows[row_index]

    def __iter__(self) -> Iterator[RowResult]:
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def scores_matrix(self) -> tuple[tuple[float, ...], ...]:
        return tuple(r.scores for r in self.rows)

    @property
    def outputs(self) -> tuple[RowResult, ...]:
        """兼容命名：行输出集合。"""
        return self.rows

    @property
    def status_counts(self) -> dict[str, int]:
        """完整性与主行状态的计数（每行各计一次，不重复计 flag）。"""
        counts: dict[str, int] = {}
        for row in self.rows:
            status_key = getattr(row.row_status, "value", row.row_status)
            comp_key = getattr(row.completeness, "value", row.completeness)
            counts[status_key] = counts.get(status_key, 0) + 1
            counts[comp_key] = counts.get(comp_key, 0) + 1
        return dict(sorted(counts.items()))

    @property
    def flag_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.rows:
            for flag in row.flags:
                key = getattr(flag, "value", flag)
                counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "rhs_columns": self.rhs_columns,
            "status_counts": self.status_counts,
            "flag_counts": self.flag_counts,
            "rows": [r.to_dict() for r in self.rows],
            "candidates": [c.to_dict() for c in self.candidates],
            "retention": self.retention.to_dict() if self.retention else None,
            "contribution": self.contribution.to_dict(),
        }


@dataclass
class BatchResult:
    """一次批次计算的完整结果。"""

    results: tuple[SampleResult, ...]
    epoch: int = 0
    effective_set_fingerprint: str = ""
    order: tuple[str, ...] = ()
    source_description: str = ""

    # -- 容器接口 ---------------------------------------------------------

    def by_id(self) -> dict[str, SampleResult]:
        return {r.sample_id: r for r in self.results}

    def get(self, sample_id: str) -> SampleResult:
        for r in self.results:
            if r.sample_id == sample_id:
                return r
        raise KeyError(sample_id)

    def __getitem__(self, key: int | str) -> SampleResult:
        if isinstance(key, int):
            return self.results[key]
        return self.get(key)

    def __iter__(self) -> Iterator[SampleResult]:
        return iter(self.results)

    def __len__(self) -> int:
        return len(self.results)

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(r.sample_id for r in self.results)

    @property
    def samples(self) -> tuple[SampleResult, ...]:
        """兼容命名：样本结果元组。"""
        return self.results

    @property
    def candidates(self) -> tuple[Candidate, ...]:
        out: list[Candidate] = []
        for sample in self.results:
            out.extend(sample.candidates)
        return tuple(out)

    @property
    def all_rows(self) -> tuple[RowResult, ...]:
        out: list[RowResult] = []
        for sample in self.results:
            out.extend(sample.rows)
        return tuple(out)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "operator_csr_batches.result/v1",
            "epoch": self.epoch,
            "effective_set_fingerprint": self.effective_set_fingerprint,
            "order": list(self.order),
            "source_description": self.source_description,
            "samples": [s.to_dict() for s in self.results],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        import json

        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def write_json(self, path: str) -> None:
        from pathlib import Path

        Path(path).write_text(self.to_json(), encoding="utf-8")

    def explain(self) -> str:
        """生成人类可读的结果说明摘要。"""
        lines = [
            "CSR batch matmul result",
            f"  epoch={self.epoch} set={self.effective_set_fingerprint[:16]}",
            f"  samples={list(self.sample_ids)}",
        ]
        for sample in self.results:
            lines.append(
                f"  - {sample.sample_id}: rows={sample.n_rows} "
                f"candidates={len(sample.candidates)} counts={sample.status_counts}"
            )
            for row in sample.rows:
                flag_values = [getattr(flag, "value", flag) for flag in row.flags]
                lines.append(
                    f"      row {row.row_index}: scores={list(row.scores)} "
                    f"completeness={getattr(row.completeness, 'value', row.completeness)} "
                    f"status={getattr(row.row_status, 'value', row.row_status)} "
                    f"flags={flag_values} proof={row.proof_id}"
                )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.explain()
