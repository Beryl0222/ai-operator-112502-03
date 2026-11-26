"""算子相关异常类型。"""

from __future__ import annotations

from typing import Any


class OperatorError(Exception):
    """所有算子层异常的基类。"""


class BlockValidationError(OperatorError):
    """CSR 块结构或取值不合法。

    在宽松模式下该异常不会抛出：块会被标记为
    :attr:`BlockStatus.REJECTED <operator_csr_batches.status.BlockStatus.REJECTED>`
    并把 ``issues`` 记录在案。
    """

    def __init__(self, block_id: str | None = None, issues: list[str] | tuple[str, ...] = ()):
        self.block_id = block_id
        self.issues = tuple(issues)
        prefix = f"CSR block {block_id!r} " if block_id else "CSR block "
        super().__init__(prefix + "validation failed: " + "; ".join(self.issues))


class UnknownBlockError(OperatorError, KeyError):
    """引用了注册中心中不存在的块 ID。"""

    def __init__(self, block_id: str):
        self.block_id = block_id
        super().__init__(f"unknown block_id: {block_id!r}")


class UnknownSampleError(OperatorError, KeyError):
    """引用了不存在的样本 ID。"""

    def __init__(self, sample_id: str):
        self.sample_id = sample_id
        super().__init__(f"unknown sample_id: {sample_id!r}")


class SampleBoundaryError(OperatorError):
    """样本边界、形状或行覆盖映射不一致。"""


class ProvenanceMismatchError(OperatorError):
    """试图合并样本身份或贡献摘要不同的候选结果。"""

    def __init__(self, message: str, *, left: Any = None, right: Any = None):
        self.left = left
        self.right = right
        super().__init__(message)


class RetentionError(OperatorError):
    """保留策略参数不合法。"""


class DimensionMismatchError(OperatorError):
    """CSR 列维度与稠密右矩阵维度不一致。"""
