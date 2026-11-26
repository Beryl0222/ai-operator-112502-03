"""CSR 权重块：校验、稳定规范化与内容摘要。

规范化规则（确定性归并的第一部分契约）
--------------------------------------
对每一行：

1. 按原始位置切片 ``indptr[r]:indptr[r+1]``；
2. 若列索引不是非降序，标记 ``indices_unsorted``，并对 ``(column, 原始位置)``
   做**稳定升序排序**——同列重复项保持它们在输入中的相对顺序；
3. 连续同列条目归并：从第一个值开始做 IEEE-754 左折叠
   ``((v0 + v1) + v2) ...``，记录重复次数与原始位置；
4. 规范化后的条目严格按列升序排列。

整个过程不依赖线程到达顺序或 Python 进程的哈希种子。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .errors import BlockValidationError
from .hashing import digest

Number = int | float


# ---------------------------------------------------------------------------
# 规范化结果类型
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CanonicalEntry:
    """同一列归并后的一个条目。"""

    column: int
    value: float
    multiplicity: int = 1
    original_positions: tuple[int, ...] = ()

    def to_term(self) -> list[Any]:
        """参与算术指纹的最小表示：列 + 精确浮点十六进制 + 重复次数。"""
        return [self.column, float(self.value).hex(), self.multiplicity]


@dataclass(frozen=True)
class CanonicalRow:
    local_row: int
    entries: tuple[CanonicalEntry, ...]
    raw_nnz: int
    was_empty: bool
    was_unsorted: bool
    had_duplicates: bool
    had_explicit_zero: bool

    @property
    def merged_nnz(self) -> int:
        return len(self.entries)


@dataclass(frozen=True)
class CanonicalBlock:
    rows: tuple[CanonicalRow, ...]
    content_digest: str
    """原始提交内容（精确到浮点比特）的摘要。"""

    canonical_digest: str
    """规范化归并结果的摘要。"""


# ---------------------------------------------------------------------------
# CSR 块
# ---------------------------------------------------------------------------


def _as_int_tuple(seq: Sequence[Any] | None) -> tuple[int, ...]:
    if seq is None:
        return ()
    out: list[int] = []
    for x in seq:
        if isinstance(x, bool):
            out.append(int(x))
        elif isinstance(x, int):
            out.append(x)
        elif isinstance(x, float) and x.is_integer():
            out.append(int(x))
        else:
            raise ValueError(f"index/indptr entry is not an integer: {x!r}")
    return tuple(out)


def _as_float_tuple(seq: Sequence[Any] | None) -> tuple[float, ...]:
    if seq is None:
        return ()
    out: list[float] = []
    for x in seq:
        if isinstance(x, bool):
            out.append(float(int(x)))
        elif isinstance(x, (int, float)):
            out.append(float(x))
        else:
            raise ValueError(f"CSR value is not numeric: {x!r}")
    return tuple(out)


@dataclass(frozen=True)
class CsrBlock:
    """一个不可变的小型 CSR 权重块。

    ``shape`` 为块自身的 ``(n_local_rows, n_columns)``。当块只覆盖样本的
    一段输出行时，使用 ``row_start`` 或显式 ``row_indices`` 声明局部行到
    样本全局行的映射。
    """

    block_id: str
    sample_id: str
    indptr: tuple[int, ...]
    indices: tuple[int, ...]
    values: tuple[float, ...]
    shape: tuple[int, int]
    row_start: int = 0
    row_indices: tuple[int, ...] | None = None
    revision: int = 1
    metadata: tuple[tuple[str, Any], ...] = field(default_factory=tuple)

    # ---- 构造 -----------------------------------------------------------

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any] | "CsrBlock",
        *,
        block_id: str | None = None,
        sample_id: str | None = None,
        default_block_id: str | None = None,
    ) -> "CsrBlock":
        if isinstance(data, CsrBlock):
            return data
        if not isinstance(data, dict):
            raise BlockValidationError(block_id, [f"unsupported block payload type: {type(data).__name__}"])

        bid = block_id or data.get("block_id") or data.get("id") or default_block_id
        sid = sample_id or data.get("sample_id")
        if sid is None:
            raise BlockValidationError(bid, ["missing sample_id"])
        if bid is None:
            raise BlockValidationError(None, ["missing block_id"])

        shape = data.get("shape")
        try:
            if shape is None:
                n_rows = len(data["indptr"]) - 1
                n_cols = max(data["indices"], default=-1) + 1
                shape = (n_rows, n_cols)
            shape_t = (int(shape[0]), int(shape[1]))
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise BlockValidationError(bid, [f"invalid shape: {exc}"]) from exc

        try:
            indptr = _as_int_tuple(data.get("indptr"))
            indices = _as_int_tuple(data.get("indices"))
            values = _as_float_tuple(data.get("values"))
        except ValueError as exc:
            raise BlockValidationError(bid, [str(exc)]) from exc

        row_indices = data.get("row_indices") or data.get("rows")
        if row_indices is not None:
            try:
                row_indices = _as_int_tuple(row_indices)
            except ValueError as exc:
                raise BlockValidationError(bid, [str(exc)]) from exc

        metadata = data.get("metadata")
        if isinstance(metadata, dict):
            metadata_t = tuple(sorted((str(k), v) for k, v in metadata.items()))
        else:
            metadata_t = ()

        return cls(
            block_id=str(bid),
            sample_id=str(sid),
            indptr=indptr,
            indices=indices,
            values=values,
            shape=shape_t,
            row_start=int(data.get("row_start", 0) or 0),
            row_indices=row_indices,
            revision=int(data.get("revision", 1) or 1),
            metadata=metadata_t,
        )

    # ---- 行覆盖 ---------------------------------------------------------

    @property
    def n_rows(self) -> int:
        return self.shape[0]

    @property
    def n_cols(self) -> int:
        return self.shape[1]

    def global_rows(self) -> tuple[int, ...]:
        """块的每个局部行对应的样本全局行号。"""
        if self.row_indices is not None:
            return self.row_indices
        return tuple(self.row_start + j for j in range(self.n_rows))

    def local_row_of(self, global_row: int) -> int | None:
        mapping = self.global_rows()
        try:
            return mapping.index(global_row)
        except ValueError:
            return None

    # ---- 校验 -----------------------------------------------------------

    def validate(self, sample_n_rows: int | None = None, sample_n_cols: int | None = None) -> tuple[str, ...]:
        """返回问题列表；空元组表示合法。"""
        issues: list[str] = []
        n_rows, n_cols = self.shape
        if n_rows < 0 or n_cols < 0:
            issues.append(f"shape dimensions must be non-negative, got {self.shape}")
        if len(self.indptr) != n_rows + 1:
            issues.append(f"indptr length {len(self.indptr)} != shape[0]+1 {n_rows + 1}")
        if len(self.indices) != len(self.values):
            issues.append(f"indices length {len(self.indices)} != values length {len(self.values)}")

        if self.indptr:
            if any(p < 0 for p in self.indptr):
                issues.append("indptr contains negative offset")
            if self.indptr[0] != 0:
                issues.append(f"indptr[0] must be 0, got {self.indptr[0]}")
            if any(a > b for a, b in zip(self.indptr, self.indptr[1:])):
                issues.append("indptr must be non-decreasing")
            if self.indptr[-1] != len(self.indices):
                issues.append(f"indptr[-1] {self.indptr[-1]} != len(indices) {len(self.indices)}")
            if self.indptr[-1] != len(self.values):
                issues.append(f"indptr[-1] {self.indptr[-1]} != len(values) {len(self.values)}")

        for pos, col in enumerate(self.indices):
            if col < 0 or col >= n_cols:
                issues.append(f"indices[{pos}]={col} out of range [0,{n_cols})")
                if len(issues) >= 12:
                    issues.append("too many index issues; truncating")
                    break

        for pos, value in enumerate(self.values):
            if not math.isfinite(value):
                issues.append(f"values[{pos}]={value!r} is not finite (NaN/Inf forbidden for deterministic output)")

        if self.row_indices is not None:
            if len(self.row_indices) != n_rows:
                issues.append(
                    f"row_indices length {len(self.row_indices)} != block local rows {n_rows}"
                )
            if any(r < 0 for r in self.row_indices):
                issues.append("row_indices contains negative global row index")
            if len(set(self.row_indices)) != len(self.row_indices):
                issues.append("row_indices must not map two local rows to the same global row")
        elif self.row_start < 0:
            issues.append(f"row_start must be non-negative, got {self.row_start}")

        if sample_n_rows is not None:
            for gr in self.global_rows():
                if gr >= sample_n_rows:
                    issues.append(
                        f"block covers global row {gr} outside sample row count {sample_n_rows}"
                    )
                    break
        if sample_n_cols is not None and n_cols != sample_n_cols:
            issues.append(f"block columns {n_cols} != sample/dense columns {sample_n_cols}")

        return tuple(issues)

    def require_valid(self, sample_n_rows: int | None = None, sample_n_cols: int | None = None) -> None:
        issues = self.validate(sample_n_rows, sample_n_cols)
        if issues:
            raise BlockValidationError(self.block_id, issues)

    # ---- 摘要与规范化 ----------------------------------------------------

    def raw_content_obj(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "sample_id": self.sample_id,
            "shape": list(self.shape),
            "indptr": list(self.indptr),
            "indices": list(self.indices),
            # 用浮点十六进制保证 -0.0 等比特差异也进入摘要
            "values": [float(v).hex() for v in self.values],
            "revision": self.revision,
        }

    @property
    def content_digest(self) -> str:
        return digest(self.raw_content_obj())

    def canonicalize(self) -> CanonicalBlock:
        issues = self.validate()
        if issues:
            raise BlockValidationError(self.block_id, issues)

        rows: list[CanonicalRow] = []
        canonical_obj: list[list[list[Any]]] = []

        for r in range(self.n_rows):
            start, end = self.indptr[r], self.indptr[r + 1]
            cols = list(self.indices[start:end])
            vals = list(self.values[start:end])
            positions = list(range(start, end))

            was_empty = start == end
            was_unsorted = any(cols[i] > cols[i + 1] for i in range(len(cols) - 1))
            had_explicit_zero = any(v == 0.0 for v in vals)

            # 稳定排序：键为列号；重复列保持输入相对顺序。
            order = sorted(range(len(cols)), key=lambda i: cols[i])

            entries: list[CanonicalEntry] = []
            groups: list[list[int]] = []
            for i in order:
                if groups and cols[groups[-1][0]] == cols[i]:
                    groups[-1].append(i)
                else:
                    groups.append([i])

            had_duplicates = any(len(g) > 1 for g in groups)
            row_obj: list[list[Any]] = []

            for group in groups:
                column = cols[group[0]]
                # 从第一个存储值开始左折叠，避免额外的 0.0+v 改变 -0.0 语义。
                merged: float = vals[group[0]]
                for i in group[1:]:
                    merged = merged + vals[i]
                original_positions = tuple(positions[i] for i in group)
                entry = CanonicalEntry(
                    column=column,
                    value=merged,
                    multiplicity=len(group),
                    original_positions=original_positions,
                )
                entries.append(entry)
                row_obj.append(entry.to_term())

            crow = CanonicalRow(
                local_row=r,
                entries=tuple(entries),
                raw_nnz=end - start,
                was_empty=was_empty,
                was_unsorted=was_unsorted,
                had_duplicates=had_duplicates,
                had_explicit_zero=had_explicit_zero,
            )
            rows.append(crow)
            canonical_obj.append(row_obj)

        return CanonicalBlock(
            rows=tuple(rows),
            content_digest=self.content_digest,
            canonical_digest=digest({"block": self.block_id, "rows": canonical_obj}),
        )

    # ---- 序列化 ---------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "block_id": self.block_id,
            "sample_id": self.sample_id,
            "indptr": list(self.indptr),
            "indices": list(self.indices),
            "values": list(self.values),
            "shape": list(self.shape),
            "revision": self.revision,
        }
        if self.row_start:
            data["row_start"] = self.row_start
        if self.row_indices is not None:
            data["row_indices"] = list(self.row_indices)
        if self.metadata:
            data["metadata"] = {k: v for k, v in self.metadata}
        return data


def coerce_blocks(blocks: Iterable[Any], *, sample_id: str | None = None) -> list[CsrBlock]:
    """把字典 / :class:`CsrBlock` 混合序列转换为块列表。"""
    out: list[CsrBlock] = []
    for i, item in enumerate(blocks):
        if isinstance(item, CsrBlock):
            out.append(item)
        elif isinstance(item, dict):
            bid = item.get("block_id") or item.get("id") or f"block-{i}"
            out.append(CsrBlock.from_dict(item, block_id=str(bid), sample_id=sample_id))
        else:
            raise BlockValidationError(None, [f"unsupported block type at index {i}: {type(item).__name__}"])
    return out
