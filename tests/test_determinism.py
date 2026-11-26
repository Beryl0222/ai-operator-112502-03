"""浮点确定性测试：到达顺序、排列、线程调度都不得改变结果或证明。"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
import itertools
import unittest
from concurrent.futures import ThreadPoolExecutor

from operator_csr_batches import BatchProcessor, OperatorConfig, sparse_batch_matmul

RHS = [[1.0, 0.5], [2.0, 1.5], [3.0, 2.5]]


def make_blocks():
    return [
        {
            "block_id": "B0", "sample_id": "s",
            "indptr": [0, 2, 2], "indices": [2, 1],
            "values": [1.5, 2.5], "shape": [2, 3],
        },
        {
            "block_id": "B1", "sample_id": "s",
            "indptr": [0, 1, 3], "indices": [0, 1, 1],
            "values": [4.0, 1.0, -0.25], "shape": [2, 3],
        },
        {
            "block_id": "B2", "sample_id": "s",
            "indptr": [0, 3, 3], "indices": [2, 2, 0],
            "values": [2.0, 2.0, -1.0], "shape": [2, 3],
        },
    ]


def compute_with_order(order, *, threaded=False):
    processor = BatchProcessor(RHS, config=OperatorConfig())
    processor.add_sample("s", n_rows=2, n_cols=3)
    for b in make_blocks():
        processor.declare(b["block_id"], "s")
    blocks = make_blocks()
    if threaded:
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(lambda i: processor.submit(dict(blocks[i])), order))
    else:
        for i in order:
            processor.submit(dict(blocks[i]))
    result = processor.compute().batch["s"]
    return [(r.scores, r.proof_id, r.flags, r.reasons) for r in result.rows]


class OrderIndependenceTest(unittest.TestCase):
    def test_all_submission_permutations_match(self):
        baseline = compute_with_order((0, 1, 2))
        for perm in itertools.permutations(range(3)):
            with self.subTest(perm=perm):
                self.assertEqual(compute_with_order(perm), baseline)

    def test_thread_arrival_order_matches(self):
        baseline = compute_with_order((0, 1, 2))
        for _ in range(5):
            self.assertEqual(compute_with_order((2, 0, 1), threaded=True), baseline)

    def test_repeated_compute_is_identical(self):
        blocks = make_blocks()
        r1 = sparse_batch_matmul(blocks, RHS)
        r2 = sparse_batch_matmul([dict(b) for b in reversed(blocks)], RHS)
        ids1 = [(s.sample_id, r.row_index, r.scores, r.proof_id) for s in r1 for r in s.rows]
        ids2 = [(s.sample_id, r.row_index, r.scores, r.proof_id) for s in r2 for r in s.rows]
        self.assertEqual(ids1, ids2)


class FloatingPointOrderContractTest(unittest.TestCase):
    def test_duplicate_fold_uses_documented_left_association(self):
        # 构造对结合顺序敏感的值：从第一个值左折叠。
        big, small = 1.0e16, 1.0
        data = {
            "block_id": "fp", "sample_id": "f",
            "indptr": [0, 3], "indices": [0, 0, 0],
            "values": [big, small, -big], "shape": [1, 1],
        }
        rhs = [[1.0]]
        row = sparse_batch_matmul([data], rhs)["f"][0]
        # 文档化归并：(big + small) 在此量级被舍入为 big，再加 -big => 0.0
        merged = (big + small) + (-big)
        self.assertEqual(row.scores[0], merged)
        self.assertEqual(row.scores[0], 0.0)

    def test_digest_is_stable_across_instances(self):
        from operator_csr_batches.hashing import digest

        d1 = digest({"a": [1, 2, 3], "b": "x"})
        d2 = digest({"b": "x", "a": [1, 2, 3]})  # 键序不影响
        self.assertEqual(d1, d2)
        self.assertEqual(len(d1), 64)


if __name__ == "__main__":
    unittest.main()
