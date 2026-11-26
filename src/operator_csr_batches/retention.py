"""样本级输出保留策略。

不同样本可以使用不同策略；策略本身参与候选结果的描述（但不改变底层
归并分数与证明），因此调整保留策略不会污染算术缓存键。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .errors import RetentionError


@dataclass(frozen=True)
class RetentionPolicy:
    """决定哪些行作为候选向下游传递，以及保留哪些分数列。

    Parameters
    ----------
    mode:
        * ``"all"``     —— 保留全部行（默认）；
        * ``"nonzero"`` —— 丢弃分数向量全为 0 的行（结构零行仍保留在
          ``SampleResult.rows`` 的状态说明中）；
        * ``"top_k"``   —— 按 ``score_column`` 降序取前 ``top_k``，
          平局按 ``row_index`` 升序，保证确定性。
    columns:
        候选 ``scores`` 中保留的输出列；``None`` 表示全部保留。
    only_complete:
        为 ``True`` 时，来源不完整的行不进入候选列表。
    min_score:
        若设置，主分数低于该值的行被丢弃。
    """

    mode: str = "all"
    top_k: int | None = None
    score_column: int = 0
    columns: tuple[int, ...] | None = None
    only_complete: bool = False
    min_score: float | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("all", "nonzero", "top_k"):
            raise RetentionError(f"unsupported retention mode: {self.mode!r}")
        if self.mode == "top_k":
            if self.top_k is None or self.top_k <= 0:
                raise RetentionError("top_k mode requires positive top_k")
        if self.columns is not None:
            if len(set(self.columns)) != len(self.columns):
                raise RetentionError(f"columns must be unique: {self.columns}")
            if any(c < 0 for c in self.columns):
                raise RetentionError("column indices must be non-negative")
        if self.min_score is not None:
            try:
                float(self.min_score)
            except (TypeError, ValueError) as exc:
                raise RetentionError("min_score must be numeric") from exc

    # ---- 工厂 -----------------------------------------------------------

    @classmethod
    def all(cls, **kwargs: Any) -> "RetentionPolicy":
        return cls(mode="all", **kwargs)

    @classmethod
    def nonzero(cls, **kwargs: Any) -> "RetentionPolicy":
        return cls(mode="nonzero", **kwargs)

    @classmethod
    def top_k(cls, k: int, *, score_column: int = 0, **kwargs: Any) -> "RetentionPolicy":
        return cls(mode="top_k", top_k=int(k), score_column=score_column, **kwargs)

    @classmethod
    def from_any(cls, value: "RetentionPolicy | Mapping[str, Any] | str | None") -> "RetentionPolicy":
        if value is None:
            return DEFAULT_POLICY
        if isinstance(value, RetentionPolicy):
            return value
        if isinstance(value, str):
            if value == "all":
                return DEFAULT_POLICY
            if value == "nonzero":
                return cls.nonzero()
            if value.startswith("top_k"):
                k = int(value.split(":", 1)[1]) if ":" in value else 10
                return cls.top_k(k)
            raise RetentionError(f"unknown retention policy string: {value!r}")
        if isinstance(value, Mapping):
            data = dict(value)
            if "columns" in data and data["columns"] is not None:
                data["columns"] = tuple(data["columns"])
            return cls(**data)
        raise RetentionError(f"unsupported retention policy: {value!r}")

    # ---- 序列化 ---------------------------------------------------------

    def descriptor(self) -> dict[str, Any]:
        out: dict[str, Any] = {"mode": self.mode}
        if self.mode == "top_k":
            out["top_k"] = self.top_k
            out["score_column"] = self.score_column
        if self.columns is not None:
            out["columns"] = list(self.columns)
        if self.only_complete:
            out["only_complete"] = True
        if self.min_score is not None:
            out["min_score"] = self.min_score
        return out


DEFAULT_POLICY = RetentionPolicy()


def resolve_policy_map(
    retention: "RetentionPolicy | Mapping[str, Any] | None",
) -> dict[str, RetentionPolicy]:
    """把统一策略或 ``sample_id -> policy`` 映射规范化。"""
    if retention is None:
        return {}
    if isinstance(retention, RetentionPolicy):
        return {"*": retention}
    if isinstance(retention, Mapping):
        # 可能是单个策略的字典（含 mode 键），也可能是样本映射
        if "mode" in retention and not _looks_like_sample_map(retention):
            return {"*": RetentionPolicy.from_any(retention)}
        out: dict[str, RetentionPolicy] = {}
        for sid, policy in retention.items():
            out[str(sid)] = RetentionPolicy.from_any(policy)
        return out
    raise RetentionError(f"unsupported retention argument type: {type(retention).__name__}")


def _looks_like_sample_map(mapping: Mapping[str, Any]) -> bool:
    policy_keys = {"mode", "top_k", "score_column", "columns", "only_complete", "min_score"}
    return not set(mapping.keys()) <= policy_keys


def policy_for_sample(policy_map: Mapping[str, RetentionPolicy], sample_id: str) -> RetentionPolicy:
    return policy_map.get(sample_id) or policy_map.get("*") or DEFAULT_POLICY


def project_scores(scores: Sequence[float], policy: RetentionPolicy) -> tuple[float, ...]:
    if policy.columns is None:
        return tuple(scores)
    return tuple(scores[c] for c in policy.columns)
