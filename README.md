# 稀疏 CSR 批矩阵乘的确定性归并

`operator_csr_batches` 是一个仅依赖 Python 标准库的可复用算子，按**样本边界**
计算 CSR × 稠密矩阵，提供：

- 定义明确的重复列归并、索引稳定排序与浮点累加顺序，结果不随线程到达顺序、
  提交排列、机器或 Python 哈希种子变化；
- 每个输出的来源完整性标记：是否由覆盖该行的**全部声明块**贡献；
- `zero_row`、`duplicate_columns_merged`、`indices_unsorted`、
  `explicit_zero_value`、`rejected`/`withdrawn`/`declared` 等**互相区分的状态**，
  不把异常来源统统当成零；
- 块注册中心：一次装载/规范化后复用，算术按 `merge_fingerprint` 缓存；
- 撤回 / 拒绝 / 补交 / 修订后，用归并证明评估哪些候选分数 `reusable`、
  哪些 `must_recompute`，重算只处理受影响输出并换发新证明；
- 随候选向下游筛选传递 `sample_id` 与 `ContributionSummary`，
  禁止仅凭相等数值合并不同来源的结果。

完整结果契约、状态表、证明/失效协议与手算样例见
**[`RESULTS.md`](RESULTS.md)**；机器可读示例：
[`docs/example_result.json`](docs/example_result.json)、
[`docs/lifecycle_example.json`](docs/lifecycle_example.json)。

## 快速开始

```python
from operator_csr_batches import sparse_batch_matmul, BatchProcessor

# 函数式：直接消费 data/input.json 形态
result = sparse_batch_matmul(payload)
result["s1"][0].row_status              # "zero_row"
result["s1"][0].all_blocks_contributed  # True
result.to_json()

# 有状态：撤回/补交 + 失效评估
proc = BatchProcessor(payload["dense_rhs"])
proc.load_source(payload)
candidates = proc.compute().candidates
proc.withdraw("s0#block-0")
plan = proc.assess(candidates)          # plan.reusable / plan.must_recompute
outcome = proc.recompute(plan)          # kept + 带新证明的 recomputed
```

命令行：

```bash
python -m operator_csr_batches --input data/input.json --output docs/example_result.json
```

## 环境

项目运行和资料检查仅依赖 Python 标准库（要求 Python >= 3.11）。运行测试与
打包验证前安装锁定的测试工具：

```bash
python -m pip install -e .
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -v
python -m build --wheel --no-isolation
```

## 资料

`data/input.json` 是可重复使用的本地合成输入，`data/schema.json` 记录字段含义
和允许的状态。`tests/` 包含数值契约、确定性、生命周期与候选来源安全测试；
`examples/lifecycle_demo.py` 演示撤回/补交/重算的完整流程。
