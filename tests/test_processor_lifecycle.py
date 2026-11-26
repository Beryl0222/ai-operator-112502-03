"""块撤回/补交场景下的证明失效、复用评估与缓存测试。"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
import unittest

from operator_csr_batches import BatchProcessor, BlockStatus, ScoreDecision

RHS = [
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0],
    [7.0, 8.0],
]


def fixture_processor():
    processor = BatchProcessor(RHS)
    processor.load_source(
        {
            "batches": [
                {"sample_id": "s0", "indptr": [0, 2, 3], "indices": [0, 2, 2],
                 "values": [1.0, 2.0, 3.0], "shape": [2, 4]},
                {"sample_id": "s1", "indptr": [0, 0, 1], "indices": [1],
                 "values": [4.0], "shape": [2, 4]},
            ]
        }
    )
    # 只覆盖 s0 第 1 行的附加块
    processor.declare("C", "s0", row_indices=(1,), block_n_rows=1)
    processor.submit(
        {"block_id": "C", "sample_id": "s0", "indptr": [0, 1], "indices": [0],
         "values": [10.0], "shape": [1, 4], "row_indices": [1]}
    )
    return processor


class WithdrawalAssessmentTest(unittest.TestCase):
    def setUp(self):
        self.processor = fixture_processor()
        self.report = self.processor.compute()
        self.candidates = {c.candidate_id: c for c in self.report.candidates}

    def test_initial_proofs_all_verify(self):
        plan = self.processor.assess(tuple(self.candidates.values()))
        self.assertEqual(plan.summary()["must_recompute"], 0)
        self.assertEqual(plan.summary()["reusable"], 4)

    def test_withdrawal_invalidates_only_covered_row(self):
        self.processor.withdraw("C")
        plan = self.processor.assess(tuple(self.candidates.values()))
        stale_ids = {e.ref.candidate_id for e in plan.entries if e.must_recompute}
        reusable_ids = {e.ref.candidate_id for e in plan.entries if e.is_reusable}
        self.assertEqual(stale_ids, {"s0:r1"})
        self.assertEqual(reusable_ids, {"s0:r0", "s1:r0", "s1:r1"})
        stale_entry = next(e for e in plan.entries if e.must_recompute)
        self.assertTrue(any("withdrawn" in r for r in stale_entry.reasons))

    def test_recompute_replaces_proof_and_marks_incomplete(self):
        old = self.candidates["s0:r1"]
        old_proof_id = old.proof_id
        self.processor.withdraw("C")
        plan = self.processor.assess(tuple(self.candidates.values()))
        outcome = self.processor.recompute(plan)

        self.assertEqual([c.candidate_id for c in outcome.recomputed], ["s0:r1"])
        new = outcome.recomputed[0]
        self.assertNotEqual(new.proof_id, old_proof_id)
        self.assertFalse(new.all_blocks_contributed)
        self.assertEqual(new.row_status, BlockStatus.WITHDRAWN)
        self.assertEqual(new.score, 15.0)  # 失去 C 的 +10

        # 旧证明在当前注册中心不再成立
        ok, _ = self.processor.verify_proof(old.proof)
        self.assertFalse(ok)
        # 替换映射可用
        self.assertIs(outcome.replacement_for(old_proof_id), new)

        # 未受影响的候选原样保留，且旧证明仍可验证
        kept_ids = {c.candidate_id for c in outcome.kept}
        self.assertEqual(kept_ids, {"s0:r0", "s1:r0", "s1:r1"})
        for cand in outcome.kept:
            self.assertTrue(cand.proof.verify(self.processor))

    def test_new_output_never_reuses_old_proof_object(self):
        self.processor.withdraw("C")
        plan = self.processor.assess((self.candidates["s0:r1"],))
        outcome = self.processor.recompute(plan)
        new_proof = outcome.recomputed[0].proof
        old_proof = self.candidates["s0:r1"].proof
        self.assertIsNot(new_proof, old_proof)
        self.assertNotEqual(new_proof.proof_id, old_proof.proof_id)
        self.assertNotEqual(new_proof.source_fingerprint, old_proof.source_fingerprint)

    def test_supplement_restores_completeness_with_fresh_proof(self):
        old = self.candidates["s0:r1"]
        self.processor.withdraw("C")
        stale = self.processor.compute().candidates
        stale_r1 = next(c for c in stale if c.candidate_id == "s0:r1")

        self.processor.supplement(
            {"block_id": "C", "sample_id": "s0", "indptr": [0, 1], "indices": [0],
             "values": [10.0], "shape": [1, 4], "row_indices": [1], "revision": 2}
        )
        plan = self.processor.assess(stale)
        self.assertEqual(plan.affected_samples, ("s0",))
        outcome = self.processor.recompute(plan)
        restored = next(c for c in outcome.recomputed if c.row_index == 1 and c.sample_id == "s0")
        self.assertTrue(restored.all_blocks_contributed)
        self.assertEqual(restored.scores, (25.0, 38.0))
        # revision=2 改变来源基础，因此证明不同于最初 revision=1 的证明
        self.assertNotEqual(restored.proof_id, old.proof_id)

    def test_resubmit_identical_content_recovers_original_proof(self):
        old = self.candidates["s0:r1"]
        self.processor.withdraw("C")
        self.processor.supplement(
            {"block_id": "C", "sample_id": "s0", "indptr": [0, 1], "indices": [0],
             "values": [10.0], "shape": [1, 4], "row_indices": [1], "revision": 1}
        )
        restored = self.processor.compute().batch["s0"][1]
        self.assertTrue(restored.is_complete)
        self.assertEqual(restored.proof_id, old.proof_id)


class RejectedBlockLifecycleTest(unittest.TestCase):
    def test_invalid_submission_is_rejected_then_resubmission_repairs(self):
        processor = BatchProcessor([[1.0, 2.0], [3.0, 4.0]])
        processor.add_sample("x", n_rows=1, n_cols=2)
        processor.declare("b", "x")
        receipt = processor.submit(
            {"block_id": "b", "sample_id": "x", "indptr": [0, 1], "indices": [9],
             "values": [1.0], "shape": [1, 2]}
        )
        self.assertEqual(receipt.status, BlockStatus.REJECTED.value)
        row = processor.compute().batch["x"][0]
        self.assertEqual(row.completeness, "incomplete")
        self.assertEqual(row.status, BlockStatus.REJECTED)

        bad_candidate = processor.compute().candidates[0]
        # 补交合法版本
        processor.supplement(
            {"block_id": "b", "sample_id": "x", "indptr": [0, 1], "indices": [0],
             "values": [3.0], "shape": [1, 2], "revision": 2}
        )
        plan = processor.assess((bad_candidate,))
        self.assertEqual(plan.summary()["must_recompute"], 1)
        outcome = processor.recompute(plan)
        self.assertTrue(outcome.recomputed[0].all_blocks_contributed)
        self.assertEqual(outcome.recomputed[0].scores, (3.0, 6.0))


class CacheReuseTest(unittest.TestCase):
    def test_unchanged_compute_hits_arithmetic_cache(self):
        processor = fixture_processor()
        first = processor.compute()
        self.assertEqual(first.cache_misses, 4)  # s0 2 行 + s1 2 行
        second = processor.compute()
        self.assertEqual(second.cache_hits, 4)
        self.assertEqual(second.cache_misses, 0)

    def test_withdrawal_recomputes_only_affected_arithmetic(self):
        processor = fixture_processor()
        processor.compute()
        processor.withdraw("C")
        report = processor.compute(sample_ids=("s0",))
        # s0:r1 未命中；s0:r0 的 merge_fingerprint 不变 -> 命中
        self.assertIn("s0:1", report.computed_keys)
        self.assertIn("s0:0", report.reused_keys)

    def test_rhs_change_invalidates_proof(self):
        processor = fixture_processor()
        candidates = processor.compute().candidates
        old = candidates[0]
        processor.set_rhs([[0.0] * 2 for _ in range(4)])
        plan = processor.assess(candidates)
        self.assertTrue(all(e.must_recompute for e in plan.entries))
        self.assertIn("rhs_changed", " ".join(plan.entries[0].reasons))


if __name__ == "__main__":
    unittest.main()
