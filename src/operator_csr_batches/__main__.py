"""命令行入口：读取资料 JSON，输出带状态与贡献摘要的结果 JSON。

用法::

    python -m operator_csr_batches --input data/input.json --output result.json
    python -m operator_csr_batches -i data/input.json            # 打印摘要
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .operator import sparse_batch_matmul


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="operator_csr_batches")
    parser.add_argument("--input", "-i", required=True, help="输入 JSON 路径（CSR 批次 + dense_rhs）")
    parser.add_argument("--output", "-o", help="结果 JSON 输出路径；省略时打印人类可读摘要")
    parser.add_argument("--json", action="store_true", help="即使不写文件也打印完整 JSON")
    args = parser.parse_args(argv)

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    result = sparse_batch_matmul(payload)

    if args.output:
        result.write_json(args.output)
        print(f"wrote {args.output}")
    elif args.json:
        print(result.to_json())
    else:
        print(result.explain())
    return 0


if __name__ == "__main__":
    sys.exit(main())
