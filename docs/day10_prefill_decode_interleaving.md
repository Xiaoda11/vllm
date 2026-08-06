# Day 10：Prefill/Decode 交错实验

## 工程问题

当请求 A 已经进入 Decode、请求 B 携带 8K 或 16K prompt 到达时，v0.26
统一 Scheduler 如何分配每个 step 的全局 token budget？A 和 B 会表现出什么
延迟取舍？

## 版本与执行路径

- 源码分支：基于 v0.26.0 的 `exp/mrv2-scheduler-trace`。
- 运行环境：`/home/xiaoda/vllm-lab/.venv-v026`。
- 实测执行路径：Path A；设置 `VLLM_WSL2_ENABLE_PIN_MEMORY=1` 和
  `VLLM_USE_V2_MODEL_RUNNER=1`，使用 MRV2。
- 模型：`/home/xiaoda/vllm-lab/models/Qwen2.5-0.5B-Instruct`，FP16。
- 实测 commit：`7c3664f0ce`（`bench: add decode prefill timing matrix`）。
- 实测 runner/backend：MRV2 和 `TRITON_ATTN`；trace 显示一个包含 24 层的
  `FullAttentionSpec` group，使用 NHD layout。

## 实验前假设

1. A 进入 Decode 后，它的下一个输出 token 会先消耗全局
   `max_num_batched_tokens` budget 中的 1 token，剩余 budget 再分给 B 的
   partial prefill。
2. 因此，一个混合 step 会给 A 调度 1 token，并在 Scheduler 其他限制和 KV
   分配允许的情况下，最多给 B 调度 `budget - 1` tokens。
3. 更大的 token budget 应减少 partial-prefill step 数，并可能降低 B 的
   TTFT；但更大的混合 batch 也可能提高 A 的 ITL。这只是预测，单次冷启动
   实验不足以支持稳定性能结论。
4. 在相同 budget 下，B=16K 应比 B=8K 需要更多 partial-prefill steps，因而
   延长 A Decode 与 B Prefill 的交错区间。

## 受控 Workload

- A：1024 prompt tokens、512 output tokens，在 0 秒到达。
- B：8192 或 16384 prompt tokens、32 output tokens，计划在 1.0 秒到达。
- 全局 token budget：2048、4096 或 8192。
- 并发数：2；开启 chunked prefill；关闭 prefix caching；使用 eager mode。
- 矩阵规模：两种 B prompt 长度 × 三种 budget，共六组实测。

最初的 0.25 秒延迟未通过校准：B 在 0.250434 秒提交，早于 A 在 0.338839
秒产生首 token。修正后的 1.0 秒仍然只是 workload 输入，不能单独证明发生了
交错。只有当实测产物同时证明 A 在 B 提交前已经产生 token，并且至少一个
Scheduler step 同时包含 A Decode 和 B Prefill，该组运行才可纳入矩阵。

## 观测量

- Scheduler JSONL/CSV：每个请求的 scheduled tokens、初始/剩余 budget、
  running/waiting 顺序、KV 分配和 preemption。
- MRV2 JSONL：执行顺序、persistent row、`idx_mapping`、
  `query_start_loc` 和输入 tensor shape。
- Request timing：A 的 TPOT/E2E，以及 B 的 TTFT/E2E。
- Token timing：CPU 观察到的流式输出事件时间；只有相邻 chunk 都恰好包含
  1 token 时，才记录为精确的 single-token ITL。
- Batch 类型：仅 A Decode、A-Decode/B-Prefill 混合、仅 B Prefill，以及 B
  Prefill 完成后的 Decode batch。

这些 trace 可以证明 batch 的组成和 shape，但不能证明 GPU kernel 耗时或硬件
因果关系；后者需要后续 Nsight 实验。

## 判定标准

一组 workload 必须同时满足以下条件才算通过：

1. 两个请求都生成了各自要求的准确 output-token 数；
2. A 的首 token 时间早于 B 的提交时间；
3. 至少一个 step 给 A 调度 1 token，同时给 B 调度 prompt tokens；
4. scheduled tokens 永远不超过记录的全局 token budget；
5. Scheduler scheduled-token 总数与 MRV2 input shape 一致；
6. 多 token 的流式 chunk 不会被误记成精确 single-token ITL。

## 实验产物与验证结果

六组正式 run ID 为：

- `day10-s5-b8k-budget{2048,4096,8192}-20260801`
- `day10-s5-b16k-budget{2048,4096,8192}-20260801`

每个目录均位于 `/home/xiaoda/vllm-lab/outputs`，并包含
`run_metadata.json`、`startup.log`、`request_timing.csv`、
`token_timing.csv`、两条 JSONL trace 和展开后的 Scheduler CSV。综合分析目录为
`/home/xiaoda/vllm-lab/outputs/day10-matrix-analysis-20260801`。

六组运行全部通过预先声明的检查：

- 两个请求都生成了准确的目标 output-token 数；
- A 在 B 提交前已经产生首 token；
- 每组都存在 A-Decode/B-Prefill 混合 step；
- 没有任何 step 超过全局 token budget；
- 每个实际 forward step 中，Scheduler scheduled tokens 均与 MRV2 的
  `input_ids.shape[0]`、batch token 数和 `query_start_loc[-1]` 一致；
- preemption 和 allocation failure 均为 0。

## Scheduler 实测行为

在每个完整的混合 Prefill step 中，A 消耗 1 token，B 消耗剩余的
`budget - 1` tokens。最后一个 prompt 尾块 step 只调度 B 达到准确 prompt
长度所需的 tokens。

| B prompt | Budget | 完整混合 shape `[A,B]` | 尾块 shape | 混合 Prefill step 数 |
|---:|---:|---:|---:|---:|
| 8K | 2048 | `[1,2047]` × 4 | `[1,4]` | 5 |
| 8K | 4096 | `[1,4095]` × 2 | `[1,2]` | 3 |
| 8K | 8192 | `[1,8191]` × 1 | `[1,1]` | 2 |
| 16K | 2048 | `[1,2047]` × 8 | `[1,8]` | 9 |
| 16K | 4096 | `[1,4095]` × 4 | `[1,4]` | 5 |
| 16K | 8192 | `[1,8191]` × 2 | `[1,2]` | 3 |

以一个完整的 4096-budget step 为例，对应的 MRV2 证据为
`num_scheduled_tokens=[1,4095]`、`query_start_loc=[0,1,4096]` 和
`input_ids.shape=[4096]`。这在不读取 GPU tensor 的前提下，闭合了从
Scheduler 全局 budget 到 MRV2 输入 batch 的 step-level 证据链。

## 延迟与吞吐实测

矩阵中每个点只运行了一次冷启动，并使用 eager mode。下表描述的是实际运行，
不是稳定性能 benchmark。

| B prompt | Budget | A TPOT ms | B TTFT ms | B Prefill 期间 A ITL mean / P95 ms | Output token/s |
|---:|---:|---:|---:|---:|---:|
| 8K | 2048 | 34.281 | 5336.749 | 764.127 / 2067.815 | 30.719 |
| 8K | 4096 | 34.808 | 5327.839 | 892.135 / 3245.029 | 30.108 |
| 8K | 8192 | 35.204 | 5341.096 | 1339.300 / 4488.222 | 29.814 |
| 16K | 2048 | 64.431 | 20718.605 | 1884.249 / 4472.850 | 16.380 |
| 16K | 4096 | 64.643 | 20670.714 | 2586.731 / 8024.546 | 16.331 |
| 16K | 8192 | 64.884 | 20689.070 | 4142.045 / 13304.196 | 16.267 |

在所有运行中，B Prefill 阶段之外的 A 平均 ITL 都接近 24–25 ms，包括 B
到达前、B 的 32-token Decode 期间，以及 B 完成后。B Prefill 期间，A 的
mean 和 P95 ITL 明显升高。增大 budget 会减少 partial-prefill 混合 step 的
数量，但会让单次停顿更长。

在该矩阵中，更大的 budget 没有实质改善 B 的 TTFT：三组 8K 运行的 TTFT
极差为 13.257 ms，三组 16K 运行的极差为 47.891 ms。因此，本次观测到的是
停顿粒度变化，而不是已经得到证明的 TTFT 收益。A 的平均 TPOT 同样会掩盖
Prefill 阶段长达数秒的尾部停顿，所以对于该 workload，分阶段 ITL 是信息量
更高的指标。

表中的吞吐定义为：完成的 output tokens 除以 workload makespan。它不是 input
吞吐，因此不能把 B=8K 与 B=16K 视为相同工作量直接比较。

## Batch 与 kernel 的证据边界

Trace 直接证明了四种 batch 阶段：仅 A Decode、A-Decode/B-Prefill 混合、
双请求 Decode，以及 B 完成后的仅 A Decode。所有 attention metadata 都是
`TritonAttentionMetadata`；启动日志记录了 `kernel_unified_attention` 和
`reduce_segments` 的 JIT。

这些证据不包含每个 kernel 的持续时间。因此，它们可以支持上述 batch shape
和 backend 结论，但不能支持“某个特定 kernel 导致 ITL spike”的说法。Kernel
耗时和硬件因果关系仍然属于 Day 16/17 的 Nsight 问题。

## 源码事实、实测结果与推断

- 源码/trace 事实：在每个混合 step 中，v0.26 统一 Scheduler 将 A 的 Decode
  token 和 B 的 Prefill chunk 计入同一个全局 budget。
- 实测结果：更大的 budget 产生更少但更大的 mixed-prefill batches；在相同
  prompt 长度内，B 的 TTFT 近似不变，而 A 的 Prefill 阶段 ITL 尾部增大。
- 待验证推断：在当前模型/backend 上，B 的 TTFT 主要由 Prefill 总工作量决定，
  而不是由 partial-prefill step 数决定。在做出因果性能结论前，还需要 Nsight
  或重复 warm runs。

## 尚未解释的问题

如果进行重复 warm runs，B TTFT 近似不变、ITL 尾部随 budget 增大的趋势是否
仍然成立？还是部分单次运行差异来自冷 Triton JIT、温度或正常运行波动？

## 30 秒技术摘要

我构造了一个 vLLM v0.26 MRV2 workload：A 是一个 1K prompt、生成 512 tokens
的请求，在它进入 Decode 后，再加入一个 8K 或 16K Prefill。Scheduler trace
显示，每个混合 step 会先给 A 分配 1 个 Decode token，再把剩余全局 budget
分给 Prefill。例如 budget 为 4096 时，MRV2 batch 是 `[1,4095]`，对应
`query_start_loc=[0,1,4096]`。增大 budget 会减少 Prefill chunk 数，但会给
Decode 请求造成次数更少、持续时间更长的 ITL 停顿，而 B 的单次运行 TTFT
近似不变。这说明平均 TPOT 会掩盖尾部延迟，也为下一步从 ITL 证据选择调度
策略提供了依据。
