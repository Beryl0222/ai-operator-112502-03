"""撤回 / 补交 / 重验证 / 重算的完整生命周期示例。

运行::

    python examples/lifecycle_demo.py [docs/lifecycle_example.json]

输出一个可直接阅读的 JSON，展示：
1. 初始完整结果与证明；
2. 撤回 C 后，哪些分数 reusable、哪些 must_recompute；
3. 重算得到的不完整输出与新证明；
4. 补交 C 后恢复完整来源。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# 允许直接 `python examples/lifecycle_demo.py` 运行
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from operator_csr_batches import BatchProcessor  # noqa: E402

RHS = [
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0],
    [7.0, 8.0],
]


def build_processor() -> BatchProcessor:
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
    processor.declare("C", "s0", row_indices=(1,), block_n_rows=1)
    processor.submit(
        {"block_id": "C", "sample_id": "s0", "indptr": [0, 1], "indices": [0],
         "values": [10.0], "shape": [1, 4], "row_indices": [1]}
    )
    return processor


def run_demo() -> dict:
    processor = build_processor()
    initial = processor.compute()
    initial_candidates = initial.candidates

    phase_1 = {
        "phase": "initial",
        "epoch": processor.epoch,
        "cache_stats": initial.cache_stats,
        "candidates": [
            {
                "candidate_id": c.candidate_id,
                "score": c.score,
                "complete": c.all_blocks_contributed,
                "proof_id": c.proof_id,
            }
            for c in initial_candidates
        ],
    }

    processor.withdraw("C", reason="weight block recalled by owner")
    plan_after_withdraw = processor.assess(initial_candidates)
    phase_2 = {
        "phase": "after_withdraw_C_assessment",
        "epoch": processor.epoch,
        "summary": plan_after_withdraw.summary(),
        "decisions": [e.to_dict() for e in plan_after_withdraw.entries],
    }

    outcome = processor.recompute(plan_after_withdraw)
    phase_3 = {
        "phase": "recomputed_after_withdraw",
        "kept": [getattr(c, "candidate_id", None) for c in outcome.kept],
        "recomputed": [
            {
                "candidate_id": c.candidate_id,
                "score": c.score,
                "complete": c.all_blocks_contributed,
                "status": c.row_status,
                "old_proof_replaced_by": c.proof_id,
            }
            for c in outcome.recomputed
        ],
        "cache_stats": outcome.report.cache_stats,
    }

    # 调用方继续使用：保留的旧候选 + 重算的新候选
    current_scores = list(outcome.kept) + list(outcome.recomputed)

    processor.supplement(
        {"block_id": "C", "sample_id": "s0", "indptr": [0, 1], "indices": [0],
         "values": [10.0], "shape": [1, 4], "row_indices": [1], "revision": 2}
    )
    plan_after_supplement = processor.assess(current_scores)
    repaired = processor.recompute(plan_after_supplement)
    phase_4 = {
        "phase": "after_supplement_C_revision_2",
        "epoch": processor.epoch,
        "assessment": plan_after_supplement.summary(),
        "recomputed": [
            {
                "candidate_id": c.candidate_id,
                "scores": list(c.scores),
                "complete": c.all_blocks_contributed,
                "proof_id": c.proof_id,
            }
            for c in repaired.recomputed
        ],
    }

    return {
        "description": "撤回/补交如何改变有效块集合、证明与分数可用性",
        "phases": [phase_1, phase_2, phase_3, phase_4],
    }


def main() -> int:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs/lifecycle_example.json")
    doc = run_demo()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
