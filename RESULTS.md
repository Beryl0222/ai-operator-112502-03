# 稀疏 CSR 批矩阵乘算子 — 结果与契约说明

包名：`operator_csr_batches`（仅依赖 Python 3.11+ 标准库）。

本文件说明算子的计算契约、**状态语义**、结果对象、归并证明与缓存失效协议，
以及下游候选筛选应如何使用样本身份与贡献摘要。可复现的示例输出见
[`docs/example_result.json`](docs/example_result.json)，
撤回/补交生命周期见 [`docs/lifecycle_example.json`](docs/lifecycle_example.json)。

---

## 1. 快速使用

```python
from operator_csr_batches import sparse_batch_matmul

# 直接消费 data/input.json 形态：{"batches": [...], "dense_rhs": [...]}
result = sparse_batch_matmul(payload)

result["s0"][0].scores                 # (11.0, 14.0)
result["s1"][0].row_status             # "zero_row"
result["s1"][0].all_blocks_contributed # True（空行也由全部声明块完整贡献）
result.to_json()                       # 可直接 json.dumps 的完整结果
```

有撤回/补交、缓存复用与重验证需求时使用 `BatchProcessor`：

```python
from operator_csr_batches import BatchProcessor

proc = BatchProcessor(dense_rhs)
proc.load_source(payload)             # 或 proc.declare(...) + proc.submit(...)
report = proc.compute()               # ComputeReport（含 cache_stats）

plan = proc.assess(report.candidates) # 变更前/后都可评估
plan.reusable                         # 可继续使用的旧候选（旧证明仍成立）
plan.must_recompute                   # 必须重算的旧候选

proc.withdraw("block-c")
plan = proc.assess(report.candidates)
outcome = proc.recompute(plan)        # kept 原样保留；recomputed 换发新证明
outcome.replacements[old_proof_id]    # 旧候选 -> 新候选
```

命令行：

```bash
python -m operator_csr_batches --input data/input.json --output docs/example_result.json
python -m operator_csr_batches --input data/input.json   # 打印人类可读摘要
```

---

## 2. 计算边界与确定性归并契约

### 2.1 样本边界

输入批次中的每个样本有独立的 `sample_id`、形状与输出行集合。算子严格按样本
分组，块通过以下方式映射到样本输出行：

* 默认连续映射：块局部行 `j` → 全局行 `row_start + j`（`row_start` 默认 0）；
* 显式 `row_indices=[...]`：局部行 `j` → `row_indices[j]`（支持行分片/重叠累加）；
* `cover_all=True`：覆盖样本全部行。

一行的“应贡献块集合”= 当前声明中覆盖该行的全部块。默认声明覆盖所有行，
此时它就是该样本的全部声明块。

### 2.2 浮点累加顺序（与线程到达顺序无关）

对每一行，有序累加项序列被完全定义：

1. **块顺序**：默认按 `block_id` 升序（`OperatorConfig(block_order="declaration")`
   可改为确定的声明顺序），不使用提交/线程完成顺序；
2. **块内索引**：稳定排序为列升序。未排序输入会被标记
   `indices_unsorted`，但排序后计算；
3. **重复列归并**：同列条目保持输入相对顺序，从第一个存储值开始左折叠
   `((v0 + v1) + v2) ...`，并记录 `multiplicity` 与原始位置；
4. **累加**：按“块 → 列升序”的项序列依次执行
   `acc[k] = acc[k] + merged_weight * rhs[column][k]`。

摘要使用 `hashlib.sha256` 与键排序的规范化 JSON，**不使用**
`PYTHONHASHSEED` 影响的内建 `hash()`，因此跨进程、跨机器、任意提交排列与
线程调度都得到比特一致的分数和证明。测试覆盖了 3 个块的全部 6 种提交排列
与线程池乱序提交。

---

## 3. 状态必须区分，不能都当成零

状态分两个正交维度，另有块生命周期状态。所有状态都是字符串枚举
（`StrEnum`），可直接与规范字符串比较，并兼容大写/近义拼写，例如
`RowStatus.ZERO_ROW == "ZERO_ROW" == "empty_row"`、
`BlockStatus.REJECTED == "block_rejected"`。

### 3.1 块状态 `BlockStatus`

| 规范值 | 含义 |
| --- | --- |
| `declared` | 已声明，数据尚未装载 |
| `accepted` | 校验通过并进入有效块集合（别名 `loaded`） |
| `rejected` | 数据非法或被权威方拒绝，不参与归并 |
| `withdrawn` | 曾被接受，后被撤回 |
| `superseded` | 旧修订版本，已被补交的新版本替换 |

### 3.2 来源完整性 `Completeness`（每行/每样本）

| 规范值 | 布尔标记 | 含义 |
| --- | --- | --- |
| `complete` | `all_blocks_contributed=True` | 覆盖该行的全部声明块均已接受并归并 |
| `incomplete` | `all_blocks_contributed=False` | 存在 rejected/withdrawn/superseded/declared 块 |
| `unsourced` | — | 没有任何声明块覆盖该行 |

### 3.3 行状态 `RowStatus`（`row.status` / `row.row_status` 为主状态）

主状态优先报告来源阻断（`rejected` / `withdrawn` / `superseded` /
`declared` / `unsourced`），来源完整时报告结构状态：

| 规范值 | 触发条件 |
| --- | --- |
| `zero_row` | **结构空行**：所有应贡献块均 accepted，且该行 `indptr[r]==indptr[r+1]`、`nnz_raw==0`。完整性仍为 `complete` |
| `duplicate_columns_merged` | 同一行出现重复列，已按定义左折叠；`flags` 含同值标记 |
| `indices_unsorted` | 列索引非升序，已稳定重排；`flags` 含 `indices_unsorted` |
| `explicit_zero_value` | 行内有显式存储的 `0.0` 条目（`nnz_raw>0`），**不是**零行 |
| `ok` | 普通行 |

关键不变量：

* **被拒绝/撤回的块永远不会被误报为 `zero_row`**。来源不完整时，即使已接受
  部分结构为空，主状态也是 `rejected`/`withdrawn`/`declared`，并在 `reasons`
  中给出 `accepted_sources_structurally_empty` 咨询原因，而不是打 `zero_row`；
* 数值全为 0 不代表零行：显式零值、正负抵消、结构空行分别由
  `explicit_zero_value`、`reasons` 中的 `result_zero`、`zero_row` 区分；
* 一行可同时带 `duplicate_columns_merged` 与 `indices_unsorted` 两个 flag。

`reasons` 元组保存可叠加的机器可读原因码（`ReasonCode`），包括
`block_rejected`（附具体校验问题）、`block_withdrawn`、
`declared_block_awaiting_data`、`rhs_changed` 等。

---

## 4. 结果对象

```
BatchResult
 ├─ epoch                         注册中心有效集合版本号
 ├─ effective_set_fingerprint     当前声明+块状态/摘要的全局指纹
 ├─ order / sample_ids
 ├─ results: [SampleResult]
 │   ├─ rows: [RowResult]         每个输出行的分数、状态、证明
 │   ├─ candidates: [Candidate]   经保留策略后传给下游的候选
 │   ├─ retention: RetentionReport
 │   ├─ status_counts / flag_counts
 │   └─ contribution: ContributionSummary（样本级）
 └─ candidates / all_rows（跨样本扁平访问）
```

`RowResult` 主要字段：

* `scores: tuple[float, ...]`（长度 = RHS 列宽 K），`.score` 为第 0 列；
* `completeness` / `is_complete` / `all_blocks_contributed`；
* `row_status`（主状态）、`structural_status`（纯结构状态）、`flags`、`reasons`；
* `nnz_raw` / `nnz_merged`、`cache_status`（`fresh`/`reused`）；
* `contribution: ContributionSummary` 与 `proof: MergeProof`、`proof_id`。

`Candidate` 额外携带 `candidate_id`、`rank`、保留策略描述，并**内嵌同一个
`ContributionSummary`**，保证样本身份与贡献摘要随候选一起进入筛选器。

所有对象都有 `to_dict()`；`BatchResult.to_json()` / `write_json(path)` 可直接
序列化，枚举会落为规范字符串。

---

## 5. 贡献摘要与归并证明

`ContributionSummary` 字段：`sample_id`、`row_index`、`completeness`、
`all_blocks_contributed`、`epoch`、`effective_set_fingerprint`、
`declared_blocks`、`expected_blocks`、`contributed_blocks`
（每块的摘要、修订、归并顺序、原始/归并 nnz、重复列等）、
`missing_blocks`（ID + 状态 + 原因）、`rhs_digest`、`proof`。

`MergeProof` 提供三层身份：

| 字段 | 覆盖内容 | 用途 |
| --- | --- | --- |
| `merge_fingerprint` | 算子版本、RHS 摘要、样本/行、**有序累加项序列** | 算术缓存键；相同则分数必然相同 |
| `source_fingerprint` | 该行期望块集合、每块状态/规范化摘要/修订/覆盖声明、RHS | 注册中心可无损重建，验证旧证明是否仍成立 |
| `proof_id` | 上述两者 + 完整性 + 归并块列表的短摘要 | 对外展示/替换映射的证明身份 |

设计要点：进入 `source_fingerprint` 的块摘要是**规范化归并摘要**，因此仅仅
乱序但规范化后完全相同的输入共享同一归并证明；原始提交的 `content_digest`
仍保留在 `BlockContribution.digest` 中供审计。块修订号 `revision` 也在基线上，
所以即使内容相同，显式的新修订也会换发证明。

---

## 6. 复用预算与撤回/补交协议

### 6.1 已装载块优先复用

* `submit()` 时只做一次校验与规范化（canonicalize），之后保存 canonical 结果
  与摘要；重复 `compute()` 不重新解析输入；
* 行算术结果按 `merge_fingerprint` 缓存在 `processor.arith_cache`。
  `ComputeReport.cache_stats = {"hits", "misses", "entries"}`，
  每行的 `cache_status` 标明 `fresh`/`reused`；
* 撤回一个块后重算时，只有来源基线变化的行缓存未命中；未被该块覆盖的行
  仍然命中缓存。

### 6.2 变更有效集合后，旧证明不能被新输出沿用

`declare / submit / reject / withdraw / supplement / set_rhs` 都会令 `epoch`
加一，并重算全局 `effective_set_fingerprint`。

`processor.assess(targets)` 对每个旧 `Candidate`/`RowResult`/集合类型：

1. 用注册中心**当前**状态重建该行的 `source_fingerprint`；
2. 与旧证明比较，并检查 RHS 摘要；
3. 输出 `RevalidationPlan`：
   * `reusable`：证明仍验证通过，调用方可以继续使用原分数与原证明；
   * `must_recompute`：基线已变化，附原因
     （`block_withdrawn:<id>`、`block_rejected:<id>`、`block_revised:<id>`、
     `new_block_declared`、`rhs_changed` 等）。

`processor.recompute(plan)`：

* `kept`：reusable 的原对象，原样保留；
* `recomputed`：仅针对 stale 行重算得到的**新候选 + 新证明对象/新 `proof_id`**，
  绝不会把旧证明贴到新输出上；
* `replacements: {old_proof_id: new_candidate}` 便于调用方替换缓存条目。

粒度是**输出行**而不是整个样本：撤回只覆盖第 1 行的块时，只有
`s0:r1` 进入 `must_recompute`，`s0:r0` 与其他样本仍然 reusable。

注意：一张 `incomplete` 证明在有效集合没有进一步变化时仍会被判定为
`reusable`——它准确描述了“当前缺少来源”的状态，可继续展示但应被下游
`only_complete` 过滤；补交/修复改变基线后它立即变为 `must_recompute`。

---

## 7. 下游候选筛选的来源规则

`Candidate` 强制携带 `sample_id` 与 `ContributionSummary`。
`operator_csr_batches.candidates` 提供来源安全工具：

* `same_provenance(a, b)`：`(sample_id, source_fingerprint)` 相同；
* `same_identity(a, b)`：再加上 `row_index`；
* `assert_same_provenance(a, b)` / `merge_scores(a, b)`：来源不同即抛
  `ProvenanceMismatchError`，**不允许仅凭相等的 score 合并**；
* `deduplicate_candidates(list)`：只按完整来源身份去重；数值相同但样本或证明
  不同的候选全部保留，并在 `score_collisions` 中报告；
* `group_by_source(candidates)`：按来源分组；
* `ProvenanceAwareFilter(only_complete=..., min_score=..., allowed_samples=...,
  require_current_proof_with=processor, ...)`：过滤结果保留来源，可丢弃证明
  已失效（`stale_proof`）的候选。

这避免了“两个样本恰好分数相同”或“撤回前后数值巧合相同”被错误合并。

---

## 8. 样例数据的手算结果

`data/input.json`（RHS 为 4×2）：

| 样本:行 | CSR 条目 | 分数 | 完整性 | 状态 |
| --- | --- | --- | --- | --- |
| `s0:0` | `(c0,1)` `(c2,2)` | `(11,14)` | complete | ok |
| `s0:1` | `(c2,3)` | `(15,18)` | complete | ok |
| `s1:0` | 无（`indptr[0]==indptr[1]`） | `(0,0)` | complete | **zero_row** |
| `s1:1` | `(c1,4)` | `(12,16)` | complete | ok |

注意 fixture 中列索引 `2` 出现两次，但位于**不同行**；行内重复列、未排序
与显式零值由专门的测试与 `examples/` 场景覆盖。

---

## 9. 保留策略

`RetentionPolicy`（可按 `sample_id` 分别传入映射）：

* `all()`：保留全部行（默认）；
* `nonzero()`：过滤分数向量全零的候选（行状态仍保留在 `SampleResult.rows`）；
* `top_k(k, score_column=0)`：按主分数降序，平局按 `row_index` 升序，确定性截断；
* 通用参数：`columns=(...)` 投影候选分数列、`only_complete=True`、`min_score`。

被保留策略丢弃的行记录在 `RetentionReport.dropped` 及原因中。

---

## 10. 验证与构建

```bash
python -m pip install -e .
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -v
python -m build --wheel --no-isolation
```

测试覆盖：fixture 数值契约、四类状态区分、重复列/乱序/显式零、全部提交排列
与线程确定性、撤回/拒绝/补交的评估与重算、缓存命中、RHS 变更、候选来源
安全、CLI 与 JSON 序列化。
