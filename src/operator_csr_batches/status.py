"""状态码定义。

算子把四个容易被混淆的情况拆成互相独立、可比较的状态：

* 零行（结构上没有任何非零条目）        -> :class:`RowStatus.ZERO_ROW`
* 重复列索引（需要按定义归并）           -> :class:`RowStatus.DUPLICATE_COLUMNS_MERGED`
* 未排序索引（先规范化再计算）           -> :class:`RowStatus.INDICES_UNSORTED`
* 被拒绝的权重块（来源不完整）           -> :class:`BlockStatus.REJECTED`

状态对象本身仍是字符串枚举（JSON 可直接序列化为规范值），同时对常见的
大小写、连字符以及近义命名做了兼容，例如
``"ZERO_ROW" == RowStatus.ZERO_ROW`` 与
``"block_rejected" == BlockStatus.REJECTED`` 都成立。
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any


def _compact(value: Any) -> str:
    """生成大小写/分隔符不敏感的比较键。"""
    text = str(value).strip().lower()
    return "".join(ch for ch in text if ch not in "_- ")


# "枚举类名.成员名" -> 该成员的额外兼容拼写（compact 形式在比较时现算）
_MEMBER_ALIASES: dict[str, tuple[str, ...]] = {}


def _member_key(member: "StatusStrEnum") -> str:
    return f"{type(member).__name__}.{member.name}"


def register_aliases(
    enum_cls: type["StatusStrEnum"],
    aliases: dict["StatusStrEnum", tuple[str, ...]],
) -> None:
    """为某个状态枚举的成员注册额外近义拼写。"""
    for member, extra in aliases.items():
        _MEMBER_ALIASES[f"{enum_cls.__name__}.{member.name}"] = tuple(extra)


class StatusStrEnum(StrEnum):
    """带兼容相等性的字符串枚举基类。

    相等判断对大小写、下划线/连字符/空格不敏感，并接受通过
    :func:`register_aliases` 声明的近义词。哈希使用规范化键，
    遵守 "相等对象哈希相等"。
    """

    @classmethod
    def _missing_(cls, value: object) -> "StatusStrEnum | None":
        if not isinstance(value, str):
            return None
        target = _compact(value)
        for member in cls:
            candidates = [_compact(member.value), _compact(member.name)]
            candidates.extend(_compact(a) for a in _MEMBER_ALIASES.get(f"{cls.__name__}.{member.name}", ()))
            if target in candidates:
                return member
        return None

    def _matches_text(self, other: str) -> bool:
        target = _compact(other)
        if target == _compact(self.value) or target == _compact(self.name):
            return True
        aliases = _MEMBER_ALIASES.get(_member_key(self), ())
        return any(target == _compact(alias) for alias in aliases)

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        if isinstance(other, StatusStrEnum):
            if other is self:
                return True
            # 不同枚举类之间，按字符串规范值/别名比较
            return self._matches_text(str(other.value)) or other._matches_text(self.value)
        if isinstance(other, str):
            return self._matches_text(other)
        return NotImplemented

    def __hash__(self) -> int:  # type: ignore[override]
        return hash(_compact(self.value))


class BlockStatus(StatusStrEnum):
    """权重块在注册中心中的生命周期状态。"""

    DECLARED = "declared"
    """已声明、数据尚未装载。"""

    ACCEPTED = "accepted"
    """数据通过校验并进入有效块集合（别名 ``loaded``）。"""

    REJECTED = "rejected"
    """提交的数据未通过校验，或被权威方显式拒绝；不参与归并。"""

    WITHDRAWN = "withdrawn"
    """曾经接受、后来被撤回；离开当前有效块集合。"""

    SUPERSEDED = "superseded"
    """已被更新修订版本替换（补交场景的旧版本）。"""


register_aliases(
    BlockStatus,
    {
        BlockStatus.DECLARED: ("pending", "awaiting_data", "declared_not_loaded"),
        BlockStatus.ACCEPTED: ("loaded", "active", "available", "ok"),
        BlockStatus.REJECTED: ("block_rejected", "rejected_block", "invalid"),
        BlockStatus.WITHDRAWN: ("withdraw", "revoked", "removed", "block_withdrawn"),
        BlockStatus.SUPERSEDED: ("replaced", "old_revision", "superseded_by_supplement", "block_superseded"),
    },
)


class Completeness(StatusStrEnum):
    """输出来源完整性维度。

    完整性与数值是否为零完全正交：零行也可以是 ``complete``（所有声明块
    都已按定义归并，只是结构为空）；非零结果也可能 ``incomplete``。
    """

    COMPLETE = "complete"
    """覆盖该行的全部声明块均已接受并参与归并。"""

    INCOMPLETE = "incomplete"
    """存在已声明但被拒绝/撤回/未装载/被替换的块。"""

    UNSOURCED = "unsourced"
    """该行没有任何声明来源（既不是完整，也不是普通零行）。"""


register_aliases(
    Completeness,
    {
        Completeness.COMPLETE: ("all_blocks_contributed", "fully_sourced", "full"),
        Completeness.INCOMPLETE: ("missing_contributions", "partial", "stale_sources"),
        Completeness.UNSOURCED: ("no_source", "undeclared", "no_declared_block"),
    },
)


class RowStatus(StatusStrEnum):
    """行状态（主状态综合来源阻断与结构规范化情况）。"""

    OK = "ok"
    """有普通条目，无重复、无乱序。"""

    ZERO_ROW = "zero_row"
    """结构空行：``indptr[r] == indptr[r+1]``，没有任何存储条目。

    仅当该行所有声明块均被接受并归并后仍无条目时使用；被拒绝块导致的
    无输出会标记为 :attr:`BLOCK_REJECTED`，绝不会误报成零行。
    """

    DUPLICATE_COLUMNS_MERGED = "duplicate_columns_merged"
    """同一行出现重复列索引，已按定义顺序左折叠归并。"""

    INDICES_UNSORTED = "indices_unsorted"
    """列索引未按升序排列，计算前已做稳定规范化。"""

    EXPLICIT_ZERO = "explicit_zero_value"
    """行内存在显式存储的 0.0 值（不是结构空行）。"""

    BLOCK_REJECTED = "rejected"
    """该行至少有一个应贡献的声明块处于 rejected 状态。"""

    BLOCK_WITHDRAWN = "withdrawn"
    """该行至少有一个应贡献的声明块已被撤回。"""

    BLOCK_PENDING = "declared"
    """该行至少有一个声明块尚未装载数据。"""

    BLOCK_SUPERSEDED = "superseded"
    """该行依赖的块修订已被补交的新版本替换。"""

    UNSOURCED = "unsourced"
    """没有任何声明块覆盖该行。"""


register_aliases(
    RowStatus,
    {
        RowStatus.OK: ("normal", "fine"),
        RowStatus.ZERO_ROW: ("empty_row", "empty", "structural_zero", "all_zero"),
        RowStatus.DUPLICATE_COLUMNS_MERGED: (
            "duplicate_columns",
            "duplicate_column",
            "duplicate_indices_merged",
            "duplicate_indices",
            "duplicate_index",
            "has_duplicates",
            "duplicate",
        ),
        RowStatus.INDICES_UNSORTED: (
            "unsorted_indices",
            "unsorted_index",
            "indices_not_sorted",
            "indices_unsorted_reordered",
            "out_of_order_indices",
            "unordered_indices",
        ),
        RowStatus.EXPLICIT_ZERO: ("stored_zero", "zero_value", "explicit_zero"),
        RowStatus.BLOCK_REJECTED: ("block_rejected", "rejected_block", "reject"),
        RowStatus.BLOCK_WITHDRAWN: ("block_withdrawn", "withdraw", "revoked"),
        RowStatus.BLOCK_PENDING: ("pending", "awaiting_data", "declared_block_awaiting_data"),
        RowStatus.BLOCK_SUPERSEDED: ("block_superseded", "superseded", "replaced"),
        RowStatus.UNSOURCED: ("no_source", "undeclared"),
    },
)


class ReasonCode(StatusStrEnum):
    """机器可读的详细原因码（可叠加，不互斥）。"""

    OK = "ok"
    ZERO_ROW = "zero_row"
    EXPLICIT_ZERO_VALUE = "explicit_zero_value"
    DUPLICATE_COLUMNS = "duplicate_columns_merged"
    UNSORTED_INDICES = "indices_unsorted_reordered"
    RESULT_ZERO = "result_zero"
    ACCEPTED_SOURCES_EMPTY = "accepted_sources_structurally_empty"
    DECLARED_AWAITING_DATA = "declared_block_awaiting_data"
    BLOCK_REJECTED = "block_rejected"
    BLOCK_WITHDRAWN = "block_withdrawn"
    BLOCK_SUPERSEDED = "block_superseded"
    BLOCK_SUPPLEMENTED = "block_supplemented"
    BLOCK_REVISED = "block_revised"
    NEW_BLOCK_DECLARED = "new_block_declared"
    EXPECTED_SET_CHANGED = "expected_block_set_changed"
    RHS_CHANGED = "rhs_changed"
    RETENTION_CHANGED = "retention_changed"
    PROVENANCE_CURRENT = "provenance_current"
    PROOF_MISMATCH = "proof_fingerprint_mismatch"


register_aliases(
    ReasonCode,
    {
        ReasonCode.BLOCK_SUPPLEMENTED: ("supplement", "supplemented"),
        ReasonCode.BLOCK_REVISED: ("revision_changed", "digest_changed"),
    },
)


class ScoreDecision(StatusStrEnum):
    """变更有效块集合后，旧分数的处置结论。"""

    REUSABLE = "reusable"
    """来源证明仍被当前注册中心验证，数值可继续使用（保留旧证明）。"""

    MUST_RECOMPUTE = "must_recompute"
    """来源基础已变化，旧归并证明失效，必须重算并领取新证明。"""

    FRESH = "fresh"
    """本次新计算。"""

    REUSED_FROM_CACHE = "reused"
    """算术结果按 ``merge_fingerprint`` 命中缓存，未重复浮点运算。"""


register_aliases(
    ScoreDecision,
    {
        ScoreDecision.REUSABLE: ("keep", "still_valid", "valid"),
        ScoreDecision.MUST_RECOMPUTE: ("stale", "invalidated", "recompute", "needs_recompute"),
        ScoreDecision.REUSED_FROM_CACHE: ("cache_hit", "cached"),
    },
)


# ---- 纯字符串常量，方便常量风格的调用方 ----------------------------------


class STATUS:
    """全部规范状态值的字符串常量集合。"""

    DECLARED = BlockStatus.DECLARED.value
    ACCEPTED = BlockStatus.ACCEPTED.value
    LOADED = ACCEPTED
    REJECTED = BlockStatus.REJECTED.value
    WITHDRAWN = BlockStatus.WITHDRAWN.value
    SUPERSEDED = BlockStatus.SUPERSEDED.value

    COMPLETE = Completeness.COMPLETE.value
    INCOMPLETE = Completeness.INCOMPLETE.value
    UNSOURCED = Completeness.UNSOURCED.value

    OK = RowStatus.OK.value
    ZERO_ROW = RowStatus.ZERO_ROW.value
    DUPLICATE_COLUMNS = RowStatus.DUPLICATE_COLUMNS_MERGED.value
    UNSORTED_INDICES = RowStatus.INDICES_UNSORTED.value
    EXPLICIT_ZERO = RowStatus.EXPLICIT_ZERO.value
    BLOCK_REJECTED = RowStatus.BLOCK_REJECTED.value
    BLOCK_WITHDRAWN = RowStatus.BLOCK_WITHDRAWN.value

    REUSABLE = ScoreDecision.REUSABLE.value
    MUST_RECOMPUTE = ScoreDecision.MUST_RECOMPUTE.value
    FRESH = ScoreDecision.FRESH.value
    REUSED = ScoreDecision.REUSED_FROM_CACHE.value
