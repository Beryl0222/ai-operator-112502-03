"""把多种输入形态装载进 :class:`~operator_csr_batches.processor.BatchProcessor`。

支持：

* 资料文件形态 ``{"batches": [ {sample_id, indptr, ...} ], "dense_rhs": [...]}``；
* 样本对象内含多个块：``{"batches": [{"sample_id": ..., "blocks": [...] }]}``；
* ``{"blocks": [ ... ]}``（每个块自带 ``sample_id``）；
* CSR 字典 / :class:`CsrBlock` 列表；
* ``{sample_id: csr_dict}`` 映射。
"""

from __future__ import annotations

from typing import Any, Iterable

from .csr import CsrBlock


def _as_block_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _sample_n_rows_from(block_data: Any) -> int | None:
    if isinstance(block_data, CsrBlock):
        return block_data.n_rows
    if isinstance(block_data, dict):
        shape = block_data.get("shape")
        if shape and len(shape) == 2:
            return int(shape[0])
        indptr = block_data.get("indptr")
        if indptr is not None:
            return len(indptr) - 1
    return None


def _sample_n_cols_from(block_data: Any) -> int | None:
    if isinstance(block_data, CsrBlock):
        return block_data.n_cols
    if isinstance(block_data, dict):
        shape = block_data.get("shape")
        if shape and len(shape) == 2:
            return int(shape[1])
    return None


def _normalize_sample_entry(entry: Any, *, fallback_index: int) -> dict[str, Any]:
    """返回 ``{sample_id, n_rows, n_cols, retention, blocks}``。"""
    if isinstance(entry, dict) and ("blocks" in entry or "sample_id" in entry and "indptr" not in entry and "shape" not in entry):
        sample_id = str(entry.get("sample_id") or f"sample-{fallback_index}")
        blocks = _as_block_list(entry.get("blocks"))
        n_rows = entry.get("n_rows") or (entry.get("shape")[0] if entry.get("shape") else None)
        n_cols = entry.get("n_cols") or (entry.get("shape")[1] if entry.get("shape") and len(entry["shape"]) == 2 else None)
        return {
            "sample_id": sample_id,
            "n_rows": int(n_rows) if n_rows is not None else None,
            "n_cols": int(n_cols) if n_cols is not None else None,
            "retention": entry.get("retention"),
            "blocks": blocks,
        }

    # 单个 CSR 块即一个样本（data/input.json 的形态）
    block = entry
    if isinstance(block, CsrBlock):
        sample_id = block.sample_id
    elif isinstance(block, dict):
        sample_id = str(block.get("sample_id") or f"sample-{fallback_index}")
    else:
        sample_id = f"sample-{fallback_index}"
    return {
        "sample_id": sample_id,
        "n_rows": _sample_n_rows_from(block),
        "n_cols": _sample_n_cols_from(block),
        "retention": block.get("retention") if isinstance(block, dict) else None,
        "blocks": [block],
    }


def extract_sample_entries(source: Any, *, extra_blocks: Iterable[Any] | None = None) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    if isinstance(source, dict) and ("batches" in source or "blocks" in source or "dense_rhs" in source):
        raw_batches = source.get("batches")
        if raw_batches is not None:
            for i, entry in enumerate(raw_batches):
                entries.append(_normalize_sample_entry(entry, fallback_index=i))
        raw_blocks = source.get("blocks")
        if raw_blocks is not None:
            for i, block in enumerate(raw_blocks):
                norm = _normalize_sample_entry(block, fallback_index=len(entries) + i)
                entries.append(norm)
        # 顶层 samples 元信息（可选）
        samples_meta = source.get("samples")
        if isinstance(samples_meta, list):
            existing = {e["sample_id"] for e in entries}
            for i, meta in enumerate(samples_meta):
                sid = str(meta.get("sample_id") or meta.get("id") or f"sample-{i}")
                if sid not in existing:
                    entries.append(
                        {
                            "sample_id": sid,
                            "n_rows": meta.get("n_rows"),
                            "n_cols": meta.get("n_cols"),
                            "retention": meta.get("retention"),
                            "blocks": [],
                        }
                    )
    elif isinstance(source, dict):
        # {sample_id: csr_like} 映射
        for i, (sid, value) in enumerate(source.items()):
            if isinstance(value, dict) and "indptr" in value:
                value = dict(value)
                value.setdefault("sample_id", sid)
            entries.append(_normalize_sample_entry(value, fallback_index=i))
    elif isinstance(source, (list, tuple)):
        for i, entry in enumerate(source):
            entries.append(_normalize_sample_entry(entry, fallback_index=i))
    elif source is not None:
        raise TypeError(f"unsupported source type: {type(source).__name__}")

    if extra_blocks:
        for i, block in enumerate(extra_blocks):
            entries.append(_normalize_sample_entry(block, fallback_index=len(entries) + i))

    # 合并同样本条目（保持首次出现顺序）
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for entry in entries:
        sid = entry["sample_id"]
        if sid not in merged:
            merged[sid] = {
                "sample_id": sid,
                "n_rows": entry["n_rows"],
                "n_cols": entry["n_cols"],
                "retention": entry["retention"],
                "blocks": list(entry["blocks"]),
            }
            order.append(sid)
        else:
            cur = merged[sid]
            cur["blocks"].extend(entry["blocks"])
            if cur["n_rows"] is None and entry["n_rows"] is not None:
                cur["n_rows"] = entry["n_rows"]
            if cur["n_cols"] is None and entry["n_cols"] is not None:
                cur["n_cols"] = entry["n_cols"]
            if cur["retention"] is None and entry["retention"] is not None:
                cur["retention"] = entry["retention"]

    # 为多块样本补默认 block_id
    for sid in order:
        e = merged[sid]
        for j, block in enumerate(e["blocks"]):
            if isinstance(block, dict) and not block.get("block_id") and not block.get("id"):
                block["block_id"] = f"{sid}#block-{j}"
    return [merged[sid] for sid in order]


def load_source(processor: Any, source: Any, *, blocks: Iterable[Any] | None = None) -> list[str]:
    """把 source 中的全部样本/块声明并提交到 processor，返回 block IDs。"""
    entries = extract_sample_entries(source, extra_blocks=blocks)
    block_ids: list[str] = []

    for entry in entries:
        sid = entry["sample_id"]
        processor.add_sample(
            sid,
            n_rows=entry["n_rows"],
            n_cols=entry["n_cols"],
        )
        if entry["retention"] is not None:
            processor.set_retention(sid, entry["retention"])

        for j, raw_block in enumerate(entry["blocks"]):
            if isinstance(raw_block, CsrBlock):
                bid = raw_block.block_id
                block_sample_id = raw_block.sample_id or sid
            elif isinstance(raw_block, dict):
                bid = str(raw_block.get("block_id") or raw_block.get("id") or f"{sid}#block-{j}")
                block_sample_id = str(raw_block.get("sample_id") or sid)
            else:
                continue

            row_start = 0
            row_indices = None
            if isinstance(raw_block, dict):
                row_start = int(raw_block.get("row_start", 0) or 0)
                row_indices = raw_block.get("row_indices") or raw_block.get("rows")
            elif isinstance(raw_block, CsrBlock):
                row_start = raw_block.row_start
                row_indices = raw_block.row_indices

            processor.declare(
                bid,
                block_sample_id,
                row_start=row_start,
                row_indices=tuple(row_indices) if row_indices is not None else None,
                block_n_rows=_sample_n_rows_from(raw_block),
            )
            receipt = processor.submit(raw_block, block_id=bid, sample_id=block_sample_id)
            block_ids.append(receipt.block_id)

    return block_ids
