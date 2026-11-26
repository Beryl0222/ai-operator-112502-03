"""块声明与行覆盖描述。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .hashing import digest


@dataclass(frozen=True)
class Declaration:
    """声明某个块属于某个样本，以及它覆盖哪些输出行。

    覆盖解析顺序：

    1. 显式 ``row_indices``（块局部行 j -> 样本全局行 row_indices[j]）；
    2. ``cover_all=True``：覆盖样本全部行；
    3. 连续映射 ``row_start .. row_start + block_n_rows - 1``；
    4. 若数据尚未装载且未给出 ``block_n_rows``，保守地按覆盖全部行处理，
       以免在信息不足时谎称输出完整。
    """

    block_id: str
    sample_id: str
    order_index: int = 0
    row_start: int = 0
    row_indices: tuple[int, ...] | None = None
    cover_all: bool = False
    block_n_rows: int | None = None
    required: bool = True

    def resolve_global_rows(
        self,
        *,
        block_n_rows: int | None = None,
        sample_n_rows: int | None = None,
    ) -> tuple[int, ...]:
        if self.row_indices is not None:
            return self.row_indices
        if self.cover_all and sample_n_rows is not None:
            return tuple(range(sample_n_rows))
        n_rows = block_n_rows if block_n_rows is not None else self.block_n_rows
        if n_rows is not None:
            return tuple(self.row_start + j for j in range(n_rows))
        if sample_n_rows is not None:
            # 信息不足时的保守策略：视为覆盖所有行。
            return tuple(range(sample_n_rows))
        return ()

    @property
    def declaration_fingerprint(self) -> str:
        return digest(
            {
                "block_id": self.block_id,
                "sample_id": self.sample_id,
                "row_start": self.row_start,
                "row_indices": list(self.row_indices) if self.row_indices else None,
                "cover_all": self.cover_all,
                "block_n_rows": self.block_n_rows,
                "required": self.required,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "sample_id": self.sample_id,
            "order_index": self.order_index,
            "row_start": self.row_start,
            "row_indices": list(self.row_indices) if self.row_indices else None,
            "cover_all": self.cover_all,
            "block_n_rows": self.block_n_rows,
            "required": self.required,
            "declaration_fingerprint": self.declaration_fingerprint,
        }
