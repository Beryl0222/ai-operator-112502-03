"""归并证明（MergeProof）与贡献摘要（ContributionSummary）。

设计要点
--------
* ``merge_fingerprint`` 只描述实际参与浮点累加的有序项序列，相同则算术
  结果必然相同，可安全命中计算缓存；
* ``source_fingerprint`` 描述该行当前的声明块集合、每个块的状态/摘要/
  修订版本以及 RHS 摘要。注册中心可以在不重算的情况下重新构造它，从而
  判断一张旧证明是否仍然成立；
* ``proof_id`` 是完整证明语句的短摘要；只要该行的有效来源集合发生变化
  （撤回 / 拒绝 / 补交 / 修订 / 新声明覆盖），``proof_id`` 必然改变；
* ``effective_set_fingerprint`` + ``epoch`` 是注册中心全局有效块集合的
  版本信息，仅用于展示与审计，不参与行级证明哈希——因此与某行无关的
  块变更不会让该行的来源指纹失效，旧分数仍可被判定为 reusable。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from .hashing import digest

OPERATOR_VERSION = "csr-batch-deterministic-1.1.0"


class ProofVerifier(Protocol):
    def verify_proof(self, proof: "MergeProof") -> tuple[bool, tuple[str, ...]]: ...


@dataclass(frozen=True)
class BlockBasis:
    """某个声明块对某一行的当前来源基础。"""

    block_id: str
    status: str
    digest: str | None
    revision: int
    declaration_fingerprint: str
    issues_fingerprint: str | None = None

    def key(self) -> list[Any]:
        return [
            self.block_id,
            self.status,
            self.digest,
            self.revision,
            self.declaration_fingerprint,
            self.issues_fingerprint,
        ]


def compute_source_fingerprint(
    *,
    operator_version: str,
    rhs_digest: str,
    sample_id: str,
    row_index: int,
    expected_bases: Sequence[BlockBasis],
) -> str:
    """行级来源指纹：注册中心可根据当前状态无损重建。"""
    statement = {
        "kind": "row_source",
        "v": operator_version,
        "rhs": rhs_digest,
        "sample": sample_id,
        "row": row_index,
        "expected": [b.key() for b in expected_bases],
    }
    return digest(statement)


def compute_merge_fingerprint(
    *,
    operator_version: str,
    rhs_digest: str,
    sample_id: str,
    row_index: int,
    ordered_terms: Sequence[Sequence[Any]],
) -> str:
    """算术指纹：精确描述累加顺序上的每个项。

    项的顺序即累加顺序：块按规范顺序（默认 block_id 升序），块内列号
    升序，重复列已按定义左折叠。
    """
    statement = {
        "kind": "row_merge",
        "v": operator_version,
        "rhs": rhs_digest,
        "sample": sample_id,
        "row": row_index,
        "terms": [list(t) for t in ordered_terms],
        "accumulation": "acc += merged_weight * rhs[column][k], block-major then column-ascending",
    }
    return digest(statement)


@dataclass(frozen=True)
class MergeProof:
    """一次确定性归并的可验证证明。"""

    proof_id: str
    source_fingerprint: str
    merge_fingerprint: str
    epoch: int
    effective_set_fingerprint: str
    operator_version: str
    rhs_digest: str
    sample_id: str
    row_index: int
    completeness: str
    expected_blocks: tuple[str, ...]
    merged_blocks: tuple[str, ...]
    block_digests: tuple[str, ...] = ()
    issued_cache_status: str = "fresh"

    @property
    def basis_fingerprint(self) -> str:
        """兼容命名：basis/source fingerprint 指向同一来源指纹。"""
        return self.source_fingerprint

    def verify(self, verifier: ProofVerifier) -> bool:
        ok, _ = verifier.verify_proof(self)
        return ok

    def verify_with_reasons(self, verifier: ProofVerifier) -> tuple[bool, tuple[str, ...]]:
        return verifier.verify_proof(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "proof_id": self.proof_id,
            "source_fingerprint": self.source_fingerprint,
            "basis_fingerprint": self.source_fingerprint,
            "merge_fingerprint": self.merge_fingerprint,
            "epoch": self.epoch,
            "effective_set_fingerprint": self.effective_set_fingerprint,
            "operator_version": self.operator_version,
            "rhs_digest": self.rhs_digest,
            "sample_id": self.sample_id,
            "row_index": self.row_index,
            "completeness": self.completeness,
            "expected_blocks": list(self.expected_blocks),
            "merged_blocks": list(self.merged_blocks),
            "block_digests": list(self.block_digests),
            "issued_cache_status": self.issued_cache_status,
        }


def build_proof(
    *,
    sample_id: str,
    row_index: int,
    epoch: int,
    effective_set_fingerprint: str,
    rhs_digest_value: str,
    expected_bases: Sequence[BlockBasis],
    merged_blocks: Sequence[str],
    block_digests: Sequence[str],
    merge_fingerprint: str,
    completeness: str,
    issued_cache_status: str = "fresh",
    operator_version: str = OPERATOR_VERSION,
) -> MergeProof:
    source_fp = compute_source_fingerprint(
        operator_version=operator_version,
        rhs_digest=rhs_digest_value,
        sample_id=sample_id,
        row_index=row_index,
        expected_bases=expected_bases,
    )
    proof_statement = {
        "source": source_fp,
        "merge": merge_fingerprint,
        "completeness": completeness,
        "merged": list(merged_blocks),
    }
    proof_id = "proof-" + digest(proof_statement)[:20]
    return MergeProof(
        proof_id=proof_id,
        source_fingerprint=source_fp,
        merge_fingerprint=merge_fingerprint,
        epoch=epoch,
        effective_set_fingerprint=effective_set_fingerprint,
        operator_version=operator_version,
        rhs_digest=rhs_digest_value,
        sample_id=sample_id,
        row_index=row_index,
        completeness=completeness,
        expected_blocks=tuple(b.block_id for b in expected_bases),
        merged_blocks=tuple(merged_blocks),
        block_digests=tuple(block_digests),
        issued_cache_status=issued_cache_status,
    )


# ---------------------------------------------------------------------------
# 贡献摘要
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlockContribution:
    """单个块对某一行（或整个样本）的贡献摘要。"""

    block_id: str
    status: str
    digest: str | None = None
    revision: int = 1
    merge_order: int | None = None
    local_row: int | None = None
    global_row: int | None = None
    raw_nnz: int = 0
    merged_nnz: int = 0
    duplicate_columns: tuple[int, ...] = ()
    unsorted: bool = False
    explicit_zero_entries: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "status": self.status,
            "digest": self.digest,
            "revision": self.revision,
            "merge_order": self.merge_order,
            "local_row": self.local_row,
            "global_row": self.global_row,
            "raw_nnz": self.raw_nnz,
            "merged_nnz": self.merged_nnz,
            "duplicate_columns": list(self.duplicate_columns),
            "unsorted": self.unsorted,
            "explicit_zero_entries": self.explicit_zero_entries,
        }


@dataclass(frozen=True)
class MissingBlock:
    """未参与归并的声明块及其原因。"""

    block_id: str
    status: str
    reasons: tuple[str, ...]
    digest: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "status": self.status,
            "reasons": list(self.reasons),
            "digest": self.digest,
        }


@dataclass(frozen=True)
class ContributionSummary:
    """随候选分数一起传递给下游筛选的贡献摘要。

    下游不应只比较分数数值；``sample_id`` + ``proof_id``（或
    ``source_fingerprint``）共同确定分数的来源。
    """

    sample_id: str
    row_index: int | None
    completeness: str
    epoch: int
    effective_set_fingerprint: str
    proof: MergeProof | None
    declared_blocks: tuple[str, ...] = ()
    expected_blocks: tuple[str, ...] = ()
    contributed_blocks: tuple[BlockContribution, ...] = ()
    missing_blocks: tuple[MissingBlock, ...] = ()
    rhs_digest: str = ""
    operator_version: str = OPERATOR_VERSION

    @property
    def all_blocks_contributed(self) -> bool:
        return self.completeness == "complete"

    @property
    def complete(self) -> bool:
        return self.completeness == "complete"

    @property
    def proof_id(self) -> str | None:
        return self.proof.proof_id if self.proof else None

    @property
    def source_fingerprint(self) -> str | None:
        return self.proof.source_fingerprint if self.proof else None

    @property
    def merge_fingerprint(self) -> str | None:
        return self.proof.merge_fingerprint if self.proof else None

    @property
    def provenance_key(self) -> tuple[str, int | None, str | None]:
        """候选来源身份：样本 + 行 + 行级来源指纹。"""
        return (self.sample_id, self.row_index, self.source_fingerprint)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "row_index": self.row_index,
            "completeness": self.completeness,
            "all_blocks_contributed": self.all_blocks_contributed,
            "complete": self.complete,
            "epoch": self.epoch,
            "effective_set_fingerprint": self.effective_set_fingerprint,
            "proof_id": self.proof_id,
            "source_fingerprint": self.source_fingerprint,
            "merge_fingerprint": self.merge_fingerprint,
            "provenance_key": list(self.provenance_key),
            "proof": self.proof.to_dict() if self.proof else None,
            "declared_blocks": list(self.declared_blocks),
            "expected_blocks": list(self.expected_blocks),
            "contributed_blocks": [c.to_dict() for c in self.contributed_blocks],
            "missing_blocks": [m.to_dict() for m in self.missing_blocks],
            "rhs_digest": self.rhs_digest,
            "operator_version": self.operator_version,
        }
