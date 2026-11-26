"""跨进程稳定的规范化 JSON 与摘要工具（仅使用标准库 hashlib）。"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any


def to_jsonable(obj: Any) -> Any:
    """递归把 dataclass / 枚举 / 元组转换为 JSON 可表示对象。"""
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return to_jsonable(obj.to_dict())
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted((to_jsonable(v) for v in obj), key=lambda x: json.dumps(x, sort_keys=True))
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, float):
        # 允许非有限值进入结构转换，但摘要路径会显式拒绝它们。
        return obj
    return obj


def canonical_text(obj: Any) -> str:
    """生成键排序、无多余空白的规范化 JSON 文本。"""
    return json.dumps(
        to_jsonable(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def digest(obj: Any, *, length: int | None = None) -> str:
    """返回任意可规范化对象的 sha256 十六进制摘要。

    不使用 Python 内建 :func:`hash`（受 ``PYTHONHASHSEED`` 影响），
    因此在不同机器、进程和线程调度下结果一致。
    """
    blob = canonical_text(obj).encode("utf-8")
    hex_digest = hashlib.sha256(blob).hexdigest()
    return hex_digest if length is None else hex_digest[:length]


def prefixed_digest(prefix: str, obj: Any, length: int = 16) -> str:
    return f"{prefix}-{digest(obj, length=length)}"
