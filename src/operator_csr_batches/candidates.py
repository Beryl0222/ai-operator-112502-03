"""候选筛选与安全合并：样本身份 + 贡献摘要优先于数值比较。

核心规则：**分数相等绝不意味着来源相同**。两个候选只有在
``(sample_id, source_fingerprint/proof_id, row/candidate identity)``
一致时才允许去重或合并；数值相同但来源不同的记录会被保留为独立候选，
并在 ``score_collisions`` 中报告，防止下游把不同来源的结果误并。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Sequence

from .errors import ProvenanceMismatchError
from .results import Candidate
from .status import Completeness


def provenance_identity(candidate: Candidate | Any) -> tuple[str, Any, Any]:
    """完整来源+位置身份：``(sample_id, source_fingerprint, row_index)``。"""
    return (
        getattr(candidate, "sample_id"),
        getattr(candidate, "source_fingerprint", None) or getattr(getattr(candidate, "contribution", None), "source_fingerprint", None),
        getattr(candidate, "row_index", None),
    )


def source_identity(candidate: Candidate | Any) -> tuple[str, Any]:
    """粗来源身份：``(sample_id, source_fingerprint)``。"""
    return (
        getattr(candidate, "sample_id"),
        getattr(candidate, "source_fingerprint", None),
    )


def same_provenance(a: Candidate | Any, b: Candidate | Any) -> bool:
    """两个候选是否来自同一样本且共享同一行级来源指纹。"""
    return source_identity(a) == source_identity(b)


def same_identity(a: Candidate | Any, b: Candidate | Any) -> bool:
    """是否为同一来源下的同一输出位置（可安全去重）。"""
    return provenance_identity(a) == provenance_identity(b)


def assert_same_provenance(a: Candidate | Any, b: Candidate | Any) -> None:
    if not same_provenance(a, b):
        raise ProvenanceMismatchError(
            "refusing to merge scores with different sample/provenance; "
            f"left={source_identity(a)!r} right={source_identity(b)!r}",
            left=a,
            right=b,
        )


def merge_scores(a: Candidate | Any, b: Candidate | Any) -> Candidate | Any:
    """合并两条候选记录；来源不同则抛出 :class:`ProvenanceMismatchError`。

    同一身份的重复记录视为幂等，返回较新的一条（若带 epoch 信息）。
    这里不做数值平均——浮点合并顺序应由算子内部的确定契约负责。
    """
    assert_same_provenance(a, b)
    if not same_identity(a, b):
        raise ProvenanceMismatchError(
            "same source but different output identity; merge by row explicitly",
            left=a,
            right=b,
        )
    epoch_a = getattr(getattr(a, "contribution", None), "epoch", -1)
    epoch_b = getattr(getattr(b, "contribution", None), "epoch", -1)
    return b if epoch_b >= epoch_a else a


@dataclass(frozen=True)
class DroppedCandidate:
    candidate: Candidate
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"candidate_id": self.candidate.candidate_id, "reason": self.reason}


@dataclass
class DeduplicationResult:
    candidates: tuple[Candidate, ...]
    duplicates: tuple[Candidate, ...] = ()
    score_collisions: tuple[tuple[Candidate, Candidate], ...] = ()
    """数值相同但来源/身份不同的候选对——明确不合并。"""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kept": [c.to_dict() for c in self.candidates],
            "duplicates_removed": [c.to_dict() for c in self.duplicates],
            "score_value_collisions_kept_separate": [
                [a.candidate_id, b.candidate_id] for a, b in self.score_collisions
            ],
        }


def deduplicate_candidates(candidates: Iterable[Candidate]) -> DeduplicationResult:
    """按完整来源身份去重，并记录跨来源的同分碰撞。"""
    kept: list[Candidate] = []
    duplicates: list[Candidate] = []
    seen: dict[tuple[Any, ...], Candidate] = {}

    for cand in candidates:
        key = provenance_identity(cand)
        if key in seen:
            duplicates.append(cand)
        else:
            seen[key] = cand
            kept.append(cand)

    collisions: list[tuple[Candidate, Candidate]] = []
    for i, a in enumerate(kept):
        for b in kept[i + 1 :]:
            if a.score == b.score and not same_identity(a, b):
                collisions.append((a, b))

    return DeduplicationResult(
        candidates=tuple(kept),
        duplicates=tuple(duplicates),
        score_collisions=tuple(collisions),
    )


def group_by_source(candidates: Iterable[Candidate]) -> dict[tuple[str, Any], list[Candidate]]:
    """按 ``(sample_id, source_fingerprint)`` 分组。"""
    groups: dict[tuple[str, Any], list[Candidate]] = {}
    for cand in candidates:
        groups.setdefault(source_identity(cand), []).append(cand)
    return groups


@dataclass
class FilterOutcome:
    kept: tuple[Candidate, ...]
    dropped: tuple[DroppedCandidate, ...] = ()
    source_groups: dict[tuple[str, Any], tuple[Candidate, ...]] = field(default_factory=dict)
    collisions: tuple[tuple[Candidate, Candidate], ...] = ()

    def __iter__(self) -> Iterator[Candidate]:
        return iter(self.kept)

    def __len__(self) -> int:
        return len(self.kept)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kept": [c.to_dict() for c in self.kept],
            "dropped": [d.to_dict() for d in self.dropped],
            "source_groups": {
                f"{sid}|{fp}": [c.candidate_id for c in group]
                for (sid, fp), group in self.source_groups.items()
            },
            "score_collisions_not_merged": [[a.candidate_id, b.candidate_id] for a, b in self.collisions],
        }


class ProvenanceAwareFilter:
    """下游候选筛选器：所有输出都保留样本身份与贡献摘要。

    Parameters
    ----------
    only_complete:
        仅保留 ``all_blocks_contributed is True`` 的候选。
    min_score / max_score:
        分数区间（按候选主分数）。
    allowed_samples:
        白名单 sample_id。
    predicate:
        额外谓词 ``Candidate -> bool``。
    dedupe:
        是否按完整来源身份去重。
    keep_score_collisions:
        同分别来源时是否保留双方（默认 True，也是安全要求）。
    """

    def __init__(
        self,
        *,
        only_complete: bool = False,
        require_current_proof_with: Any = None,
        min_score: float | None = None,
        max_score: float | None = None,
        allowed_samples: Sequence[str] | None = None,
        predicate: Callable[[Candidate], bool] | None = None,
        dedupe: bool = True,
    ):
        self.only_complete = only_complete
        self.verifier = require_current_proof_with
        self.min_score = min_score
        self.max_score = max_score
        self.allowed_samples = set(allowed_samples) if allowed_samples else None
        self.predicate = predicate
        self.dedupe = dedupe

    def apply(self, candidates: Iterable[Candidate]) -> FilterOutcome:
        dropped: list[DroppedCandidate] = []
        survivors: list[Candidate] = []

        for cand in candidates:
            if self.only_complete and not cand.all_blocks_contributed:
                dropped.append(DroppedCandidate(cand, "incomplete_source"))
                continue
            if self.min_score is not None and cand.score < self.min_score:
                dropped.append(DroppedCandidate(cand, "below_min_score"))
                continue
            if self.max_score is not None and cand.score > self.max_score:
                dropped.append(DroppedCandidate(cand, "above_max_score"))
                continue
            if self.allowed_samples is not None and cand.sample_id not in self.allowed_samples:
                dropped.append(DroppedCandidate(cand, "sample_not_allowed"))
                continue
            if self.verifier is not None and cand.proof is not None:
                ok, _ = self.verifier.verify_proof(cand.proof) if hasattr(self.verifier, "verify_proof") else cand.proof.verify(self.verifier)
                if not ok:
                    dropped.append(DroppedCandidate(cand, "stale_proof"))
                    continue
            if self.predicate is not None and not self.predicate(cand):
                dropped.append(DroppedCandidate(cand, "predicate_rejected"))
                continue
            survivors.append(cand)

        collisions: tuple[tuple[Candidate, Candidate], ...] = ()
        if self.dedupe:
            result = deduplicate_candidates(survivors)
            survivors = list(result.candidates)
            collisions = result.score_collisions

        groups = {key: tuple(group) for key, group in group_by_source(survivors).items()}
        return FilterOutcome(
            kept=tuple(survivors),
            dropped=tuple(dropped),
            source_groups=groups,
            collisions=collisions,
        )

    def __call__(self, candidates: Iterable[Candidate]) -> FilterOutcome:
        return self.apply(candidates)


def filter_candidates(candidates: Iterable[Candidate], **kwargs: Any) -> FilterOutcome:
    """函数式快捷入口，参数同 :class:`ProvenanceAwareFilter`。"""
    return ProvenanceAwareFilter(**kwargs).apply(candidates)


def merge_candidate_streams(*streams: Iterable[Candidate]) -> DeduplicationResult:
    """合并多个候选流，但绝不跨来源合并；按身份去重并报告同分碰撞。"""
    all_candidates: list[Candidate] = []
    for stream in streams:
        all_candidates.extend(stream)
    return deduplicate_candidates(all_candidates)


# 命名别名
CandidateFilter = ProvenanceAwareFilter
safe_merge = merge_scores
