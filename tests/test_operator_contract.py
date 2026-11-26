"""算子数值契约与状态区分测试。"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
import json
import unittest

from operator_csr_batches import (
    BlockStatus,
    Completeness,
    CsrBlock,
    RetentionPolicy,
    RowStatus,
    sparse_batch_matmul,
)

RHS = [
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0],
    [7.0, 8.0],
]


def block(sample_id="s", rows=2, cols=4, **over):
    data = {
        "block_id": "b0",
        "sample_id": sample_id,
        "indptr": [0, 0, 0],
        "indices": [],
        "values": [],
        "shape": [rows, cols],
    }
    data.update(over)
    return data


class FixtureNumericContractTest(unittest.TestCase):
    """与 data/input.json 对齐的精确数值。"""

    @classmethod
    def setUpClass(cls):
        from pathlib import Path

        payload = json.loads(
            (Path(__file__).resolve().parents[1] / "data" / "input.json").read_text(encoding="utf-8")
        )
        cls.result = sparse_batch_matmul(payload)

    def test_exact_scores(self):
        s0, s1 = self.result["s0"], self.result["s1"]
        self.assertEqual(s0[0].scores, (11.0, 14.0))   # 1*r0 + 2*r2
        self.assertEqual(s0[1].scores, (15.0, 18.0))   # 3*r2
        self.assertEqual(s1[0].scores, (0.0, 0.0))     # 结构空行
        self.assertEqual(s1[1].scores, (12.0, 16.0))   # 4*r1

    def test_empty_row_is_complete_but_marked_zero_row(self):
        row = self.result["s1"][0]
        self.assertEqual(row.completeness, Completeness.COMPLETE)
        self.assertTrue(row.all_blocks_contributed)
        self.assertEqual(row.status, RowStatus.ZERO_ROW)
        self.assertTrue(row.is_zero_row)
        self.assertIn("zero_row", row.flags)

    def test_normal_rows_are_ok(self):
        self.assertEqual(self.result["s0"][0].status, RowStatus.OK)
        self.assertEqual(self.result["s1"][1].status, RowStatus.OK)

    def test_result_is_json_serializable_with_proof(self):
        data = json.loads(self.result.to_json())
        self.assertEqual(data["schema"], "operator_csr_batches.result/v1")
        row0 = data["samples"][0]["rows"][0]
        self.assertTrue(row0["proof_id"].startswith("proof-"))
        self.assertIn("contribution", row0)


class DuplicateAndUnsortedTest(unittest.TestCase):
    def test_duplicate_columns_are_merged_in_defined_order(self):
        # 同一行 col1 出现两次：2.0 与 0.5，稳定顺序左折叠为 2.5
        data = block(
            indptr=[0, 2, 2],
            indices=[1, 1],
            values=[2.0, 0.5],
            shape=[2, 3],
        )
        result = sparse_batch_matmul([data], RHS[:3])
        row = result["s"][0]
        self.assertEqual(row.scores, (7.5, 10.0))  # 左折叠 2.0+0.5=2.5，再乘 rhs[1]
        self.assertTrue(row.has_duplicate_columns)
        self.assertEqual(row.status, RowStatus.DUPLICATE_COLUMNS_MERGED)
        self.assertIn("duplicate_columns_merged", row.flags)
        self.assertEqual(row.nnz_raw, 2)
        self.assertEqual(row.nnz_merged, 1)

    def test_unsorted_indices_are_flagged_and_sorted(self):
        unsorted = block(indptr=[0, 2], indices=[2, 0], values=[1.0, 2.0], shape=[1, 3])
        sorted_like = block(indptr=[0, 2], indices=[0, 2], values=[2.0, 1.0], shape=[1, 3])
        r1 = sparse_batch_matmul([unsorted], RHS[:3])["s"][0]
        r2 = sparse_batch_matmul([sorted_like], RHS[:3])["s"][0]
        self.assertTrue(r1.is_unsorted)
        self.assertEqual(r1.status, RowStatus.INDICES_UNSORTED)
        self.assertIn("indices_unsorted", r1.flags)
        self.assertFalse(r2.is_unsorted)
        self.assertEqual(r1.scores, r2.scores)
        self.assertEqual(r1.proof_id, r2.proof_id)

    def test_duplicate_and_unsorted_can_coexist_with_distinct_flags(self):
        data = block(
            indptr=[0, 3],
            indices=[2, 1, 1],
            values=[1.0, 2.0, 0.5],
            shape=[1, 3],
        )
        row = sparse_batch_matmul([data], RHS[:3])["s"][0]
        self.assertIn("duplicate_columns_merged", row.flags)
        self.assertIn("indices_unsorted", row.flags)
        self.assertNotEqual(RowStatus.DUPLICATE_COLUMNS_MERGED, RowStatus.INDICES_UNSORTED)

    def test_explicit_zero_is_not_zero_row(self):
        data = block(indptr=[0, 1, 1], indices=[0], values=[0.0], shape=[2, 2])
        row0 = sparse_batch_matmul([data], RHS[:2])["s"][0]
        self.assertFalse(row0.is_zero_row)
        self.assertEqual(row0.status, RowStatus.EXPLICIT_ZERO)
        self.assertIn("explicit_zero_value", row0.flags)
        # 第二行才是结构空行
        row1 = sparse_batch_matmul([data], RHS[:2])["s"][1]
        self.assertEqual(row1.status, RowStatus.ZERO_ROW)


class RejectedBlockStatusTest(unittest.TestCase):
    def test_invalid_block_is_rejected_and_marks_incomplete_not_zero_row(self):
        bad = block(
            block_id="bad",
            indptr=[0, 2],
            indices=[0, 9],  # 越界
            values=[1.0, 1.0],
            shape=[1, 2],
        )
        result = sparse_batch_matmul({"batches": [bad], "dense_rhs": RHS[:2]})
        row = result["s"][0]
        self.assertEqual(row.completeness, Completeness.INCOMPLETE)
        self.assertFalse(row.all_blocks_contributed)
        self.assertEqual(row.status, BlockStatus.REJECTED)
        self.assertNotEqual(row.status, RowStatus.ZERO_ROW)
        self.assertFalse(row.is_zero_row)
        self.assertTrue(any("block_rejected" in r for r in row.reasons))
        missing = row.contribution.missing_blocks
        self.assertEqual(missing[0].status, BlockStatus.REJECTED)

    def test_validation_exception_lists_issues(self):
        blk = CsrBlock.from_dict(block(indptr=[0, 2], indices=[0], values=[1.0, 2.0], shape=[1, 2]))
        issues = blk.validate()
        self.assertTrue(issues)


class RetentionPolicyTest(unittest.TestCase):
    def _three_rows(self):
        return [
            {"sample_id": "s", "indptr": [0, 1, 2, 3], "indices": [0, 1, 0],
             "values": [1.0, 1.0, 10.0], "shape": [3, 2]}
        ]

    def test_top_k_is_deterministic_with_tie_break(self):
        rhs = [[1.0, 0.0], [1.0, 0.0]]
        # scores: row0=1, row1=1, row2=10
        policy = RetentionPolicy.top_k(2, score_column=0)
        result = sparse_batch_matmul(self._three_rows(), rhs, retention=policy)
        ids = [c.row_index for c in result["s"].candidates]
        self.assertEqual(ids, [2, 0])  # 10 then tie row0 before row1
        self.assertEqual([c.rank for c in result["s"].candidates], [0, 1])

    def test_only_complete_filters_incomplete(self):
        bad = block(block_id="bad", indptr=[0, 1], indices=[9], values=[1.0], shape=[1, 2])
        policy = RetentionPolicy.all(only_complete=True)
        result = sparse_batch_matmul([bad], RHS[:2], retention=policy)
        self.assertEqual(result["s"].candidates, ())
        self.assertTrue(result["s"].retention.dropped)

    def test_per_sample_policy_mapping(self):
        data = [
            {"sample_id": "a", "indptr": [0, 1], "indices": [0], "values": [1.0], "shape": [1, 2]},
            {"sample_id": "b", "indptr": [0, 1], "indices": [0], "values": [1.0], "shape": [1, 2]},
        ]
        policies = {"a": RetentionPolicy.all(), "b": {"mode": "top_k", "top_k": 1}}
        result = sparse_batch_matmul(data, RHS[:2], policies=policies)
        self.assertEqual(len(result["a"].candidates), 1)
        self.assertEqual(len(result["b"].candidates), 1)


if __name__ == "__main__":
    unittest.main()
