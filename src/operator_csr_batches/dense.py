"""稠密右矩阵的规范化与摘要。"""

from __future__ import annotations

import math
from typing import Any, Sequence

from .hashing import digest


def coerce_rhs(dense_rhs: Sequence[Sequence[float]] | None) -> tuple[tuple[float, ...], ...]:
    if dense_rhs is None:
        raise ValueError("dense_rhs is required")
    rows: list[tuple[float, ...]] = []
    width: int | None = None
    for r, row in enumerate(dense_rhs):
        if not isinstance(row, (list, tuple)):
            raise ValueError(f"dense_rhs row {r} must be a list/tuple")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError(f"dense_rhs is not rectangular: row {r} has {len(row)}, expected {width}")
        vals: list[float] = []
        for c, v in enumerate(row):
            if isinstance(v, bool) or isinstance(v, (int, float)):
                fv = float(v)
            else:
                raise ValueError(f"dense_rhs[{r}][{c}] is not numeric: {v!r}")
            if not math.isfinite(fv):
                raise ValueError(f"dense_rhs[{r}][{c}] must be finite, got {v!r}")
            vals.append(fv)
        rows.append(tuple(vals))
    if width is None or width == 0:
        raise ValueError("dense_rhs must have at least one column")
    return tuple(rows)


def rhs_digest(rhs: Sequence[Sequence[float]]) -> str:
    return digest([[float(v).hex() for v in row] for row in rhs])
