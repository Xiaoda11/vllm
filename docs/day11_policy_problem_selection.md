# Day 11：从数据选择调度策略问题

## 结论

Day 10 数据选出了一个问题：

> vLLM v0.26 在 Decode 请求与长 Prefill 共存时，Prefill 默认可以消耗除
> Decode token 以外的全部 step token budget。Day 10 数据显示，增大全局
> budget 没有在同一 prompt 长度内带来可见的单次 B TTFT 改善，却显著放大
> A 在 B Prefill 阶段的 ITL 尾部。需要验证较小的 Prefill quantum 是否能
> 改善 Decode 尾延迟，以及 v0.26 现有配置是否已经足够。

追加配置 Gate 后的决定是：**暂不设计新的 mixed-only Scheduler patch**。
现有 `long_prefill_token_threshold=2048` 已在三次重复中把 A 的 Prefill 阶段
ITL P95 中位数降低 52.5%，B TTFT 中位数只增加 0.7%，output token/s 中位数
变化为 -0.1%。四次单请求 8K Prefill 对照也没有观察到明确退化。当前证据不
足以证明新增一条 mixed-only 分支相对现有配置具有额外工程价值。

这不是声称现有阈值在所有模型和 workload 上都最优。它只说明：在当前固定
模型、硬件、backend 和两个最小 workload 下，重复实现 Prefill quantum 会
制造低价值重复功能。Day 12 应先扩展配置策略的边界验证；只有数据暴露出现有
全局阈值的明确退化，才重新打开新策略设计。

## 既有阈值配置实验 Gate

在选题去重后追加一个不修改 Scheduler 的配置实验，用来判断 Day 12 是否真的
需要新策略逻辑。

固定 workload：

- A：1024 prompt / 512 output，先进入 Decode；
- B：8192 prompt / 32 output，延迟 1 秒到达；
- 全局 token budget：8192；
- 对照变量：`long_prefill_token_threshold=0` 与 `2048`；
- 每组运行 3 次，开启 Scheduler/MRV2 trace 和逐 token timing。

实验前假设：阈值 2048 会把完整 mixed step 中 B 的 Prefill tokens 从 8191
降低到不超过 2048，增加 partial-prefill step 数；它可能降低 A 的 Prefill
阶段 ITL P95，但可能增加 B TTFT/E2E 或降低总吞吐。

观测量与判定标准：

- 两个请求必须生成准确 token 数，且 A 必须在 B 提交前进入 Decode；
- Scheduler scheduled tokens 必须与 MRV2 input shape 一致；
- 记录 mixed-step shape、A 分阶段 ITL、B TTFT/E2E、output token/s、
  preemption 和 allocation failure；
- 若现有阈值能稳定降低 A ITL 尾部，Day 12 必须先证明 mixed-only 新逻辑相对
  全局阈值的额外价值；否则不新增 Scheduler patch；
- 若三次重复方向不一致，则不以单次结果决定策略，先扩大重复或检查热状态、
  温度和功耗。

### Gate 实测结果

混合 workload 每组独立启动 3 次，所有 run 都满足准确输出、A 在 B 提交前
进入 Decode、无 preemption/allocation failure，以及 Scheduler/MRV2 input
shape 一致。

| Threshold | Mixed Prefill pattern | Max mixed batch | A Prefill ITL mean median | A Prefill ITL P95 median | B TTFT median | B E2E median | Output token/s median |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | `1×1 + 8191×1` | 8192 | 1386.030 ms | 4646.064 ms | 5537.983 ms | 6323.612 ms | 29.486 |
| 2048 | `2048×4` | 2049 | 932.127 ms | 2204.628 ms | 5576.545 ms | 6370.859 ms | 29.455 |

阈值相对 baseline 的中位数变化：

- A Prefill 阶段 ITL mean：-32.7%；
- A Prefill 阶段 ITL P95：-52.5%；
- B TTFT：+0.7%；
- B E2E：+0.7%；
- output token/s：-0.1%。

为检查全局阈值在非混合场景的副作用，又对 S1 单请求 8K Prefill 运行每组 3
次，并追加一对交叉回测，共每组 4 次。Trace 证明 baseline 使用 `8192×1`，
阈值组使用 `2048×4`；MRV2 input shape mismatch 均为 0。

| Threshold | Prefill pattern | TTFT median | TPOT median | E2E median |
|---:|---|---:|---:|---:|
| 0 | `8192×1` | 5529.662 ms | 24.362 ms | 6285.214 ms |
| 2048 | `2048×4` | 5460.465 ms | 23.545 ms | 6190.348 ms |

后跑的阈值组数值略低，但交叉回测的最后一对 TTFT 分别为 5436.059 和
5437.315 ms，几乎相同。因此只报告“当前没有观察到纯 Prefill 明确退化”，
不把顺序相关的小差异写成阈值带来的性能提升。

## 版本、执行路径与证据

- 分支：`exp/mrv2-scheduler-trace`。
- 固定基线：v0.26.0 tag，commit
  `f2654939e69b4069b13977e9aef3e31d4dcaf051`。
- Day 10 实测代码 commit：`7c3664f0ce`。
- Day 11 开始时 HEAD：`7cac890e0f`。
- 阈值实验工具与实测 commit：`3e301bd939`。
- 环境：`/home/xiaoda/vllm-lab/.venv-v026`。
- 实测路径：Path A，MRV2 + `TRITON_ATTN`，需要显式设置
  `VLLM_WSL2_ENABLE_PIN_MEMORY=1`。
- 原始数据：`/home/xiaoda/vllm-lab/outputs/day10-s5-*-20260801`。
- 汇总数据：
  `/home/xiaoda/vllm-lab/outputs/day10-matrix-analysis-20260801/day10_matrix_summary.csv`。
- Day 11 新原始数据：
  `/home/xiaoda/vllm-lab/outputs/day11-threshold-gate-*-20260803`。
- Day 11 新汇总数据：
  `/home/xiaoda/vllm-lab/outputs/day11-threshold-gate-analysis-20260803`。

沙箱内 NVML 访问被操作系统阻止；经批准在沙箱外运行 `nvidia-smi` 和 GPU
workload。实测 GPU 为 RTX 2060，runner/backend 为 MRV2 + `TRITON_ATTN`。

Day 11 验证：

- 重新对六个 Day 10 原始目录运行 `scripts/lab_day10_analyze.py`，六组全部
  通过输出数量、到达顺序、mixed step、budget、preemption、allocation 和
  MRV2 shape 检查；重新生成的两份 CSV 与归档 CSV 逐字节一致；
- `tests/lab/test_day10_analyze.py`：2 passed；
- workload 新增 `--long-prefill-token-threshold`，相关 workload 测试共
  22 passed，Ruff passed；
- 新增 14 个 clean GPU runs：混合 workload 6 个，单请求 Prefill 8 个；
  所有请求均准确完成，所有 forward step 的 MRV2 shape 检查通过；
- `test_schedule_concurrent_partial_requests` 的两个参数化 case 在
  `ModelConfig` 初始化时因沙箱无法解析 `facebook/opt-125m` 而失败，尚未进入
  Scheduler；因此该测试只作为已存在的源码测试证据，不报告为本次动态通过；
- `git diff --check` 通过。

## 候选问题筛选

| 候选 | 已有直接证据 | 一周可完成 | 与现有机制重复风险 | 决定 |
|---|---|---|---|---|
| 两个长 partial prefill 的公平性 | Day 5 证明默认配置下后到长请求等待 | 是 | 高：已有 `long_prefill_token_threshold` 和 concurrent partial prefill | 不选 |
| 长 Prefill 使后到短请求 TTFT 恶化 | 没有长请求在前、短请求后到的正式矩阵 | 可能 | 中 | 不选 |
| 持续 Decode 造成 Prefill starvation | Day 10 中 B 每个混合 step 都得到 Prefill tokens，没有 starvation | 可能 | 中 | 不选 |
| 固定 budget 对 Decode/Prefill 混合 workload 不够灵活 | Day 10 六组均直接复现 ITL 粒度取舍 | 是 | 中，可收窄为仅混合阶段生效 | **选择** |

第一个候选不是不存在问题。Day 5 的九组 trace 确实证明默认配置具有 FCFS
头部效应：先到长 Prefill 完成后，后到长 Prefill 才开始推进。但 v0.26 已有
以下能力：

- `SchedulerConfig.long_prefill_token_threshold` 可以截断一个请求单 step 的
  Prefill tokens；
- 该阈值同时应用于 running 和 waiting 请求的调度路径；
- `max_num_partial_prefills` 和 `max_long_partial_prefills` 已暴露为配置；
- `test_schedule_concurrent_partial_requests` 已验证多个 partial prefill 在同一
  step 推进。

因此，直接实现“通用 per-request quantum + 多长 Prefill 轮转”会与已有代码
重叠。它仍可以作为配置对照，但不适合作为本周唯一的新策略修改。

第二和第三个候选缺少直接 baseline。没有数据时为它们写策略，违反“只从数据
选题”的 Day 11 约束。

## 被选问题的 baseline

Day 10 workload：

- A：1024 prompt tokens、512 output tokens，先到并进入 Decode；
- B：8192 或 16384 prompt tokens、32 output tokens，计划延迟 1 秒到达；
- token budget：2048、4096、8192；
- 并发 2，chunked prefill 开启，prefix caching 关闭，eager mode；
- 六组均无 preemption、allocation failure 或 MRV2 input shape mismatch。

每个完整混合 step 的 Scheduler/MRV2 shape 都是：

```text
A Decode: 1 token
B Prefill: budget - 1 tokens
总 batch: budget tokens
```

实测延迟如下。每个点只有一次独立冷启动，因此表格描述观察到的趋势，不是稳定
性能 benchmark。

| B prompt | Budget | B TTFT ms | A 在 B Prefill 阶段 ITL mean ms | ITL P95 ms |
|---:|---:|---:|---:|---:|
| 8K | 2048 | 5336.749 | 764.127 | 2067.815 |
| 8K | 4096 | 5327.839 | 892.135 | 3245.029 |
| 8K | 8192 | 5341.096 | 1339.300 | 4488.222 |
| 16K | 2048 | 20718.605 | 1884.249 | 4472.850 |
| 16K | 4096 | 20670.714 | 2586.731 | 8024.546 |
| 16K | 8192 | 20689.070 | 4142.045 | 13304.196 |

同一 prompt 长度内：

- 8K 的 B TTFT 极差只有 13.257 ms，但 A ITL P95 从 2067.815 ms 增至
  4488.222 ms，为 2.17 倍；
- 16K 的 B TTFT 极差只有 47.891 ms，但 A ITL P95 从 4472.850 ms 增至
  13304.196 ms，为 2.97 倍；
- B Prefill 阶段之外，A 的平均 ITL 接近 24--25 ms。

这支持“Prefill chunk 粒度与 Decode 尾延迟相关”的工程问题，但当前 trace
没有 kernel 时间，不能把 ITL spike 归因于某个具体 Triton kernel。

## 与现有阈值的区别

现有 `long_prefill_token_threshold` 是全局 Prefill cap。只要阈值大于零且小于
本次 `num_new_tokens`，无论该 step 是否有 Decode，它都会截断 Prefill。

本次选择的问题更窄：

- 只关心 Decode 与长 Prefill 共存的 mixed step；
- 目标指标是已有 Decode 请求的分阶段 ITL/P95；
- 纯 Prefill 和单请求 workload 是必须检查的退化边界；
- 不同时引入 priority、deadline、aging、动态全局 budget 或 KV 策略。

配置 Gate 已证明在当前最小范围内可以直接复用现有阈值语义。Day 12 不增加
独立 mixed-only 上限，除非新的边界数据证明全局阈值在非混合 workload 上有
明确退化。

## Day 12 的最小复现与判定标准

最小 baseline 继续使用 Day 10 S5，不增加新 workload 维度：

1. 先用 B=8K、budget=8192 作为最小高信号点；
2. 用 B=16K、budget=8192 检查更长交错区间；
3. 用 budget=2048 作为较细 chunk 的既有对照；
4. 另外保留一个纯 Prefill/单请求 case 检查退化。

后续 Baseline vs Modified 至少观察：

- A 在 B Prefill 阶段的 single-token ITL mean/P50/P95/max；
- B TTFT 和 E2E；
- output token/s 和 workload makespan；
- 每 step scheduled tokens、mixed batch shape 和 partial-prefill step 数；
- preemption、allocation failure 和 peak KV usage；
- Scheduler scheduled tokens 与 MRV2 input shape 不变量。

进入实现前的判定标准：

- 默认关闭时必须逐 step 保持 baseline 行为；
- 开启时 trace 必须证明只在目标 mixed step 改变 Prefill 粒度；
- 不能只降低 ITL P95而不报告 B TTFT、E2E 或吞吐代价；
- warm repeated runs 必须确认 Day 10 单次冷启动趋势是否稳定；
- 现有 `long_prefill_token_threshold` 已在当前 Gate 中达到目标，因此停止新的
  mixed-only patch，将下一步改为配置策略边界实验并如实记录。

## 源码事实、实测结果与推断

- 源码事实：默认 `long_prefill_token_threshold=0`；非零阈值会在 running 和
  waiting 调度路径中限制 `num_new_tokens`。
- 源码事实：默认 mixed step 先调度 running Decode，再让 waiting Prefill 使用
  剩余 token budget。
- 实测结果：Day 10 每个完整 mixed step 都是 `[1, budget-1]`，六组没有
  preemption、allocation failure 或 shape mismatch。
- 实测结果：在两个 prompt 长度内，更大 budget 对应更高的 A Prefill 阶段
  ITL mean/P95，而 B 的单次 TTFT 近似不变。
- 实测结果：阈值 2048 将 mixed Prefill 从 `8191×1` 改为 `2048×4`，A 的
  Prefill 阶段 ITL P95 中位数降低 52.5%，而 B TTFT/E2E 中位数各增加约
  0.7%，output token/s 中位数变化为 -0.1%。
- 实测结果：单请求 8K Prefill 被从 `8192×1` 拆为 `2048×4`，但四次重复中
  没有观察到明确 TTFT/TPOT/E2E 退化。
- 工程判断：当前没有证据支持新增 mixed-only Scheduler 分支；应先把已有
  配置作为 baseline 扩展到更长 prompt、并发和 burst workload。

## 未解释问题

1. 在 B=16K、并发 4/8 和 burst workload 中，阈值的收益/代价是否仍稳定？
2. 多个纯 Prefill 并发时，更多 step 是否会降低 input throughput？
3. 最佳 quantum 是否依赖模型、backend 和硬件，以至于只能作为实验性配置？
4. Nsight 中，尾延迟下降来自 batch shape/kernel 组合变化的哪一部分？

## 30 秒面试表达

我先发现 vLLM v0.26 已有全局 `long_prefill_token_threshold`，所以没有重复写
一个 quantum。我用 MRV2 将 A Decode 加 B=8K Prefill 的 8192-token mixed
batch，与 threshold=2048 的四个 2049-token mixed batches 做了三次重复。
阈值让 A 在 Prefill 阶段的 ITL P95 中位数下降 52.5%，B TTFT 只增加 0.7%，
吞吐变化约 -0.1%；四次纯 8K Prefill 对照也没看到明确退化。因此当前工程
判断是复用现有配置并扩大边界测试，而不是为完成 patch 制造重复 Scheduler
逻辑。
