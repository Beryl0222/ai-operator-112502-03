"""operator_csr_batches —— 确定性、可追溯的稀疏 CSR 批矩阵乘算子。

公开入口
--------
函数式：
    :func:`sparse_batch_matmul`（别名 ``batch_csr_matmul`` /
    ``csr_batch_matmul`` / ``sparse_batched_matmul``）

有状态生命周期（装载复用、撤回/补交、失效评估、重算）：
    :class:`BatchProcessor`、:class:`BlockRegistry`

候选筛选：
    :class:`ProvenanceAwareFilter`、:func:`deduplicate_candidates`、
    :func:`same_provenance`、:func:`assert_same_provenance`

状态码：
    :class:`BlockStatus`、:class:`Completeness`、:class:`RowStatus`、
    :class:`ReasonCode`、:class:`ScoreDecision`、:class:`STATUS`

结果模型：
    :class:`BatchResult`、:class:`SampleResult`、:class:`RowResult`、
    :class:`Candidate`、:class:`ContributionSummary`、:class:`MergeProof`
"""

from __future__ import annotations

__version__ = "1.0.0"

from .candidates import (
    CandidateFilter,
    DeduplicationResult,
    FilterOutcome,
    ProvenanceAwareFilter,
    assert_same_provenance,
    deduplicate_candidates,
    filter_candidates,
    group_by_source,
    merge_candidate_streams,
    merge_scores,
    provenance_identity,
    same_identity,
    same_provenance,
    source_identity,
)
from .csr import CanonicalBlock, CanonicalRow, CsrBlock
from .errors import (
    BlockValidationError,
    DimensionMismatchError,
    OperatorError,
    ProvenanceMismatchError,
    RetentionError,
    SampleBoundaryError,
    UnknownBlockError,
    UnknownSampleError,
)
from .operator import (
    OperatorConfig,
    SparseBatchMatMul,
    SparseBatchedMatMul,
    SparseCSRBatchMatMul,
    CSRSparseBatchMatMul,
    BatchCSRMatMul,
    CSRBatchMatMul,
    batch_csr_matmul,
    batch_matmul,
    compute_batch,
    csr_batch_matmul,
    csr_matmul,
    matmul_csr_dense,
    run_batch_matmul,
    sparse_batch_matmul,
    sparse_batched_matmul,
    sparse_csr_batch_matmul,
)
from .processor import (
    BatchOperator,
    BatchProcessor,
    BlockRegistry,
    ComputeReport,
    MutationReceipt,
    Registry,
    RevalidationOutcome,
    RevalidationPlan,
    ScoreRef,
    ScoreValidity,
    SubmissionReceipt,
)
from .provenance import (
    OPERATOR_VERSION,
    BlockContribution,
    ContributionSummary,
    MergeProof,
)
from .results import (
    BatchResult,
    Candidate,
    DroppedRow,
    RetentionReport,
    RowResult,
    SampleResult,
)
from .retention import RetentionPolicy
from .status import (
    STATUS,
    BlockStatus,
    Completeness,
    ReasonCode,
    RowStatus,
    ScoreDecision,
)

# 裸状态常量，方便 ``from operator_csr_batches import ZERO_ROW`` 风格断言
DECLARED = BlockStatus.DECLARED
ACCEPTED = BlockStatus.ACCEPTED
LOADED = BlockStatus.ACCEPTED
REJECTED = BlockStatus.REJECTED
WITHDRAWN = BlockStatus.WITHDRAWN
SUPERSEDED = BlockStatus.SUPERSEDED

COMPLETE = Completeness.COMPLETE
INCOMPLETE = Completeness.INCOMPLETE
UNSOURCED = Completeness.UNSOURCED

OK = RowStatus.OK
ZERO_ROW = RowStatus.ZERO_ROW
DUPLICATE_COLUMNS_MERGED = RowStatus.DUPLICATE_COLUMNS_MERGED
DUPLICATE_COLUMNS = RowStatus.DUPLICATE_COLUMNS_MERGED
INDICES_UNSORTED = RowStatus.INDICES_UNSORTED
EXPLICIT_ZERO = RowStatus.EXPLICIT_ZERO

REUSABLE = ScoreDecision.REUSABLE
MUST_RECOMPUTE = ScoreDecision.MUST_RECOMPUTE

__all__ = [
    # version
    "__version__",
    "OPERATOR_VERSION",
    # functional operator
    "sparse_batch_matmul",
    "sparse_csr_batch_matmul",
    "batch_csr_matmul",
    "csr_batch_matmul",
    "sparse_batched_matmul",
    "csr_matmul",
    "batch_matmul",
    "matmul_csr_dense",
    "compute_batch",
    "run_batch_matmul",
    "SparseBatchMatMul",
    "SparseBatchedMatMul",
    "SparseCSRBatchMatMul",
    "CSRSparseBatchMatMul",
    "CSRBatchMatMul",
    "BatchCSRMatMul",
    "OperatorConfig",
    # lifecycle
    "BatchProcessor",
    "BatchOperator",
    "BlockRegistry",
    "Registry",
    "ComputeReport",
    "RevalidationPlan",
    "RevalidationOutcome",
    "ScoreRef",
    "ScoreValidity",
    "SubmissionReceipt",
    "MutationReceipt",
    # csr
    "CsrBlock",
    "CanonicalBlock",
    "CanonicalRow",
    "RetentionPolicy",
    # provenance
    "MergeProof",
    "ContributionSummary",
    "BlockContribution",
    # results
    "BatchResult",
    "SampleResult",
    "RowResult",
    "Candidate",
    "DroppedRow",
    "RetentionReport",
    # candidates
    "ProvenanceAwareFilter",
    "CandidateFilter",
    "FilterOutcome",
    "DeduplicationResult",
    "deduplicate_candidates",
    "filter_candidates",
    "merge_candidate_streams",
    "merge_scores",
    "same_provenance",
    "same_identity",
    "source_identity",
    "provenance_identity",
    "assert_same_provenance",
    "group_by_source",
    # status
    "BlockStatus",
    "Completeness",
    "RowStatus",
    "ReasonCode",
    "ScoreDecision",
    "STATUS",
    # bare status constants
    "DECLARED",
    "ACCEPTED",
    "LOADED",
    "REJECTED",
    "WITHDRAWN",
    "SUPERSEDED",
    "COMPLETE",
    "INCOMPLETE",
    "UNSOURCED",
    "OK",
    "ZERO_ROW",
    "DUPLICATE_COLUMNS",
    "DUPLICATE_COLUMNS_MERGED",
    "INDICES_UNSORTED",
    "EXPLICIT_ZERO",
    "REUSABLE",
    "MUST_RECOMPUTE",
    # errors
    "OperatorError",
    "BlockValidationError",
    "UnknownBlockError",
    "UnknownSampleError",
    "SampleBoundaryError",
    "ProvenanceMismatchError",
    "RetentionError",
    "DimensionMismatchError",
]
