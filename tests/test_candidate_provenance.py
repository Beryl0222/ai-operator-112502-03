"""下游候选筛选的来源安全测试。"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
import unittest

from operator_csr_batches import (
    BatchProcessor,
    ProvenanceAwareFilter,
    ProvenanceMismatchError,
    assert_same_provenance,
    deduplicate_candidates,
    filter_candidates,
    merge_candidate_streams,
    same_provenance,
    sparse_batch_matmul,
)

RHS = [[1.0, 0.0], [2.0, 0.0]]


def candidate(sample_id, value, *, block_id="b", revision=1):
    return sparse_batch_matmul(
        [
            {
                "block_id": block_id,
                "sample_id": sample_id,
                "indptr": [0, 1],
                "indices": [0],
                "values": [value],
                "shape": [1, 2],
                "revision": revision,
            }
        ],
        RHS,
    ).candidates[0]


class ProvenanceSafetyTest(unittest.TestCase):
    def test_equal_scores_different_samples_are_not_duplicates(self):
        a = candidate("sample-a", 2.0)
        b = candidate("sample-b", 2.0)
        self.assertEqual(a.score, b.score)
        self.assertFalse(same_provenance(a, b))
        result = deduplicate_candidates([a, b])
        self.assertEqual(len(result.candidates), 2)
        self.assertEqual(len(result.score_collisions), 1)

    def test_same_sample_different_revision_has_different_proof(self):
        old = candidate("s", 2.0, revision=1)
        new = candidate("s", 2.0, revision=2)
        self.assertNotEqual(old.source_fingerprint, new.source_fingerprint)
        self.assertFalse(same_provenance(old, new))

    def test_identical_candidates_deduplicate(self):
        a = candidate("s", 3.0)
        result = deduplicate_candidates([a, a])
        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(len(result.duplicates), 1)

    def test_assert_same_provenance_raises_on_cross_source_merge(self):
        with self.assertRaises(ProvenanceMismatchError):
            assert_same_provenance(candidate("a", 1.0), candidate("b", 1.0))

    def test_merge_streams_keeps_cross_source_equal_scores(self):
        stream1 = [candidate("a", 1.0), candidate("a", 1.0)]
        stream2 = [candidate("b", 1.0)]
        result = merge_candidate_streams(stream1, stream2)
        self.assertEqual(len(result.candidates), 2)  # a 去重为 1，b 保留

    def test_filter_only_complete_and_sample_whitelist(self):
        good = candidate("a", 5.0)
        outcome = filter_candidates([good], only_complete=True, allowed_samples=["a"])
        self.assertEqual(len(outcome.kept), 1)
        outcome2 = filter_candidates([good], allowed_samples=["other"])
        self.assertEqual(len(outcome2.kept), 0)
        self.assertEqual(outcome2.dropped[0].reason, "sample_not_allowed")

    def test_filter_can_drop_stale_proofs_against_processor(self):
        processor = BatchProcessor(RHS)
        processor.load_source(
            [
                {"sample_id": "s", "indptr": [0, 1], "indices": [0],
                 "values": [1.0], "shape": [1, 2]}
            ]
        )
        old = processor.compute().candidates[0]
        processor.withdraw(processor.block_ids("s")[0])

        f = ProvenanceAwareFilter(require_current_proof_with=processor)
        outcome = f.apply([old])
        self.assertEqual(outcome.kept, ())
        self.assertEqual(outcome.dropped[0].reason, "stale_proof")

    def test_candidate_carries_sample_and_contribution_summary(self):
        c = candidate("s", 4.0)
        self.assertEqual(c.sample_id, "s")
        self.assertIsNotNone(c.contribution)
        self.assertTrue(c.contribution.proof_id.startswith("proof-"))
        self.assertIn("b", c.contribution.contributed_blocks[0].block_id)
        payload = c.to_dict()
        self.assertEqual(payload["sample_id"], "s")
        self.assertIn("contribution", payload)


if __name__ == "__main__":
    unittest.main()
