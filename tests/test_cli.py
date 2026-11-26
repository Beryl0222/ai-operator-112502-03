"""CLI 与打包入口的冒烟测试。"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import json
import tempfile
import unittest

from operator_csr_batches.__main__ import main

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "data" / "input.json"


class CliTest(unittest.TestCase):
    def test_write_result_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "result.json"
            code = main(["--input", str(FIXTURE), "--output", str(out)])
            self.assertEqual(code, 0)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], "operator_csr_batches.result/v1")
            self.assertEqual([s["sample_id"] for s in payload["samples"]], ["s0", "s1"])

    def test_stdout_explain_does_not_fail(self):
        code = main(["--input", str(FIXTURE)])
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
