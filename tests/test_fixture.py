import json
from pathlib import Path
import unittest


class FixtureContractTest(unittest.TestCase):
    def test_csr_fixture_contains_duplicates_and_an_empty_row(self):
        base = Path(__file__).parents[1]
        payload = json.loads((base / "data/input.json").read_text(encoding="utf-8"))
        first, second = payload["batches"]
        self.assertEqual(first["indices"].count(2), 2)
        self.assertEqual(second["indptr"][0], second["indptr"][1])
        self.assertEqual(len(payload["dense_rhs"]), first["shape"][1])

if __name__ == "__main__":
    unittest.main()
