# vLLM v0.26 Scheduler Trace 与 Waiting HOL 策略实验报告

## 摘要

本项目在固定的 vLLM v0.26.0、RTX 2060 Laptop GPU 和 WSL2 环境中，构建了
一个默认关闭的逐 step Scheduler/MRV2 trace，并用可控 workload 贯通
Scheduler、KV Cache Manager、Model Runner V2 和 `TRITON_ATTN`。Trace 先复现
双长 Prefill、Decode/Prefill interleaving、Prefix Cache 和 preemption，再用于
定位 KV 压力下的 waiting head-of-line（HOL）blocking。

项目实现并评估了两版 opt-in waiting bypass。无界版本能显著提前后到短请求，
但会推迟队首长请求；one-admission bounded 版本限制了绕过次数，却没有限制被
准入请求持有 KV 的生命周期。长生命周期反例新增 3 次 preemption，并使
fairness、makespan 和吞吐量退化。因此最终结论不是“应该合并一个更聪明的
Scheduler”，而是：HOL 问题真实、局部收益真实，但 admission-count bound 不是
安全的 starvation 或完成时间上界，当前 patch 不适合 upstream PR。

## 1. 环境与证据边界

| 项目 | 固定值或实测结果 |
|---|---|
| 基线 | vLLM v0.26.0，`f2654939e69b4069b13977e9aef3e31d4dcaf051` |
| 实验分支 | `exp/mrv2-scheduler-trace` |
| Python / PyTorch | 3.12.13 / 2.11.0+cu129 |
| GPU | RTX 2060 Laptop，6 GiB，SM 7.5 |
| Model Runner | MRV2，启动日志实测 |
| Attention backend | `TRITON_ATTN`，启动日志实测 |
| 模型 | Qwen2.5-0.5B-Instruct，FP16 |
| WSL 前置条件 | `VLLM_WSL2_ENABLE_PIN_MEMORY=1`，否则 UVA 不可用 |

MRV2 Path A 通过了三次独立启动、离线生成与 OpenAI server smoke。所有 GPU
实验显式设置 `VLLM_USE_V2_MODEL_RUNNER=1`，避免把 V1 runner 数据误写成
MRV2 实测。原始输出保存在仓库外的 `/home/xiaoda/vllm-lab/outputs/`；Git 只
保存脚本、测试和紧凑报告。

本文严格区分三类结论：源码事实、相同 run 的 trace/profile 观测，以及基于
观测的推断。Scheduler 的 `num_computed_tokens` 是逻辑进度，不是 CUDA kernel
完成时间戳。

## 2. 可控 workload 与 Scheduler Trace

Workload generator 接受精确 prompt/output token 长度、arrival time、并发度、
shared prefix 和 request ID。它通过 `AsyncLLM` 提交预 tokenized 输入，并为每个
run 保存 commit、完整命令、resolved scenario、request timing 和 trace 路径。

Scheduler trace 默认关闭，开启后输出两条 JSONL：

- CPU Scheduler：队列顺序、请求状态、scheduled token、budget、KV usage、
  block-table 变化、Prefix Cache hit、allocation failure 和 preemption；
- MRV2：实际 batch request 顺序、persistent row、`idx_mapping` 以及已经存在的
  CPU input-shape metadata。

实现不读取 GPU tensor、不调用 `.item()`、不增加 CUDA synchronize，JSON
序列化和落盘由后台 writer 完成。转换器按 `step_id` 将两条流展开为 CSV。

Canonical S3 的 8K/16K 双 Prefill 证明了 Scheduler 与 MRV2 的实际交错。例如
在一个 4096-token budget run 中，step 3 同时给 A 1 token、给 B 4095 tokens，
并在 MRV2 batch 中观察到 `[A, B]`。这说明 B 的进入由逐 step trace 证明，不能
从客户端提交时间推断。

## 3. Scheduler–KV–MRV2 数据流

每个 step 的受控证据链是：

```text
Request arrival
→ Scheduler waiting/running queues
→ token-budget decision
→ KV Cache allocation / failure / preemption
→ SchedulerOutput.num_scheduled_tokens
→ MRV2 update_requests and persistent rows
→ idx_mapping / gathered input metadata
→ flattened GPU input
→ TRITON_ATTN and model kernels
```

Day 6/7 用相同 trace 区分了三件经常被混淆的事：request block table 增长、
Prefix Cache 复用和物理新 block 分配；并记录了 running allocation failure、
preemption victim 及 recomputation。Day 8/9 又验证了 Scheduler scheduled token
总数、MRV2 `input_ids.shape[0]`、query start 与 attention metadata 的一致性。

本机实测 KV layout 为一个 FullAttention group、24 层、NHD、FP16，shape 为
`[17243, 2, 16, 128]`。这是当前模型/backend 的观测，不推广到所有 vLLM 模型
或 backend。

## 4. 从已有机制到真实问题

最初候选是限制 mixed Prefill quantum，但源码审计发现 v0.26 已有
`long_prefill_token_threshold`。在 B=16K mixed workload 中，threshold=2048
使 Decode Prefill-stage ITL P95 中位数下降 66.1%，而纯 16K Prefill 的
TTFT/E2E 仅变化约 +0.3%/+0.4%。预先定义的 fail-closed Gate 因此拒绝新增
重复策略。

随后从 Day 7 的 full-ISL reservation trace 中定位到真实 HOL 问题：B 位于 FCFS
队首，需要约 1024 blocks；当时只有 933 free。B allocation failure 后，现有
循环直接停止。排在后面的 C 只需约 64 blocks，本可以放入，却一同等待。

该 witness 同时满足：

- B 比 C 更早提交且位于队首；
- B 的完整输入在该 step 放不下；
- 同一 step 的 free blocks 足以容纳 C；
- C 没有被调度；
- 没有把 allocation failure 误写成 OOM 或 preemption。

## 5. 策略设计与实现

第一版 opt-in bypass 在 waiting request 分配失败时，把它临时放入
`step_skipped_waiting` 并继续扫描后续请求；下一 step 优先重试被跳过请求。
默认关闭时保持原行为，full-ISL reservation、running preemption 和 MRV2 状态
均不改变。

无界版本在三请求 repeated workload 中把 C TTFT 中位数从 33.250 s 降到
0.182 s，B first scheduled step 保持 517，三次 baseline/modified 均完成且无
preemption 或 input-shape mismatch。然而 8 个短请求 burst 中，所有 C 都能先
进入，B first scheduled step 从 517 推迟到 587，B TTFT 增加 10.0%。逐 step
优先重试无法撤销已经发生的 KV admission，因此它没有 starvation bound。

第二版 one-admission bound 为每个 blocked head 最多允许一个后续请求准入。
新增状态记录该 head 是否已经耗尽 bypass 机会，并在请求成功调度或移除时清理。
它限制的是 admission count，不是 admitted request 的运行时间或 KV 生命周期。

## 6. Baseline 与 bounded 结果

所有下表运行使用 MRV2、`TRITON_ATTN`、full-ISL reservation、1450-block KV
override、eager execution，并关闭 Prefix Cache。

### 三请求功能 Gate

| 指标 | Strict | Bounded |
|---|---:|---:|
| B TTFT | 33.240 s | 31.757 s |
| C TTFT | 33.407 s | 0.179 s |
| B first scheduled step | 517 | 517 |
| Preemption | 0 | 0 |

这证明一个短 C 可以安全地被提前，但不足以覆盖请求生命周期差异。

### 长生命周期 burst：8 个 1K/512 请求

| 指标 | Strict | Bounded | 变化 |
|---|---:|---:|---:|
| B TTFT | 31.229 s | 31.863 s | +2.03% |
| C TTFT median | 31.931 s | 32.142 s | +0.66% |
| Makespan | 52.603 s | 53.241 s | +1.21% |
| Output throughput | 88.208 tok/s | 87.150 tok/s | -1.20% |
| TTFT Jain index | 0.930977 | 0.831934 | -10.64% |
| Preemption | 0 | 3 | 退化 |

Bounded 只让 C1 的 TTFT 降至约 0.174 s，其余 C 仍约为 31.8–33.1 s。C1 长时间
持有 KV 后导致 C6、C7、C8 preempt。这一反例证明“B first step 不变”不是充分
的安全性条件。

### 短生命周期 burst：8 个 1K/32 请求

| 指标 | Strict | Bounded | 变化 |
|---|---:|---:|---:|
| B TTFT | 31.243 s | 31.440 s | +0.63% |
| C1 TTFT | 31.472 s | 0.173 s | -99.45% |
| C TTFT median | 31.938 s | 32.018 s | +0.25% |
| Makespan | 40.693 s | 40.771 s | +0.19% |
| Preemption | 0 | 0 | 不变 |

当 C1 很快结束时，它在这个单次 descriptive run 中是安全的；但仍只帮助一个
由队列位置决定的请求，没有改善 median 或 tail。长/短 case 合在一起说明，
请求长度与 KV 生命周期才是 admission safety 的关键，而 one-admission count
没有表达它们。

## 7. Scheduler 到 GPU 工作组成

Nsight Systems 2025.6.3 在当前 WSL 路径只记录 CUDA API，没有 kernel activity；
最小 PyTorch matmul 和显式 software trace 同样失败，所以 `.nsys-rep` 只作为
失败 Gate 证据。

受控 fallback 使用 vLLM 支持的 PyTorch Profiler，分别采集 strict/bounded 的
Scheduler step 60–79。分析器验证 40 个 execution annotation 与 Scheduler
scheduled-token total 全部一致，两侧均有真实 CUDA kernel event。

| 20-step profile window | Strict | Bounded |
|---|---:|---:|
| Single Decode | 20 | 17 |
| Prefill/Decode mixed | 0 | 1 |
| Dual Decode | 0 | 2 |
| Attention calls | 480 | 480 |
| GEMM calls | 0 | 291 |
| Observed kernel duration | 162.173 ms | 299.080 ms |
| Longest annotated GPU range | 30.292 ms | 143.275 ms |

Bounded 的 mixed step 是 A Decode 1 token 加 C Prefill 1024 tokens。相同的 24
层仍各调用一次 attention，但更大的 query shape 改变了 attention duration，并
引入 46.068 ms 的 GEMM 工作。策略 patch 没有修改 attention/GEMM kernel；它
改变的是现有 kernel 何时以什么 shape 运行。

这些 profile 数值来自每种模式一次且 profiler overhead 较高，只用于说明工作
组成，不能证明 bounded 全局慢 84%。稳定策略判断来自无 profiler benchmark。

Nsight Compute 最初因 `ERR_NVGPUCTRPERM` 停止。用户明确授权并通过 NVIDIA
Control Panel 开启 performance-counter 访问后，最小 matmul Gate 与真实 vLLM
Prefill GEMM 的 basic/full/stall collection 均成功。目标 kernel 为
`turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_tn`：

| 单 launch NCU 指标 | 实测值 |
|---|---:|
| Duration | 2.03 ms |
| SM throughput | 41.68% |
| DRAM throughput | 25.84% |
| L2 hit rate | 84.01% |
| Achieved occupancy | 24.67% |
| Registers / thread | 254 |
| Waves / SM | 20.27 |
| `math_pipe_throttle` stall | 61.85% |
| `long_scoreboard` stall | 1.41% |

该 launch 的 SM 和 DRAM throughput 都未饱和，主要 stall 是 math-pipe
throttle，而 long-scoreboard 很低，因此不呈现简单的 DRAM-latency-bound
特征。低 occupancy 同时受寄存器和 shared memory 限制，但不能由此推出“提高
occupancy 必然加速”或“整个 Prefill compute-bound”。

NCU launch grid 为 `(76,16,1)`，尚未与 Day 16 C mixed-step 的 `(76,9,1)`
唯一对齐；basic/full/stall 还使用不同 replay/section。因此这些结果是 targeted
single-kernel 微架构证据，不是 strict/bounded 性能归因或端到端 benchmark。

## 8. 正确性与测试

项目验证覆盖：

- trace 默认关闭且关闭时不创建 writer；
- Scheduler/MRV2 JSONL 写入与 CSV join；
- workload config、token budget、Prefix Cache、full-ISL 和 bypass override；
- 默认 strict 行为不变；
- 单请求和多个 allocation failure 的相对顺序；
- Priority policy；
- bypass 状态在成功调度、finish 和 abort 时清理；
- benchmark analyzer 对请求完成、HOL witness、preemption、shape mismatch 和
  profiler/Scheduler token alignment 的 fail-closed 校验。

已归档的相关运行均精确生成请求 token；被作为成功证据的主要 Gate 没有 MRV2
input-shape mismatch。完整 Scheduler 文件测试曾在无关 tokenizer SSL 等待中被
人工停止，因此不报告为全文件通过。

## 9. 工程判断与 PR Gate

当前 patch 不提交 upstream，理由是：

1. Day 15 审计时 upstream 已有相近的无界 skip draft，重复实现价值有限；
2. one-admission bound 只限制一次准入，不限制 C 的完成时间或 KV 生命周期；
3. 长生命周期反例出现 3 次 preemption，并使 fairness、makespan 和吞吐量退化；
4. 再增加 output-token threshold 会依赖 workload/hardware，仍不是原则性安全
   保证；
5. 真正 conservative backfill 需要预测完成时间、预留未来 KV 或强制回收已
   bypass 工作，设计范围显著扩大。

这是一个完整的负结果：问题、机制、patch、测试、正反 workload 和 GPU 执行
证据均成立，但数据否决了发布决策。若未来公开结果，更合适的动作是向已有讨论
提供 counterexample，而不是打开重复 PR；本项目当前未发布任何外部评论。

## 10. 局限与后续工作

- 单机 RTX 2060/WSL2/Qwen 0.5B，不能外推到多卡、大模型或其他 backend；
- eager 模式，没有验证 CUDA Graph 下的相同行为；
- arrival 由 wall clock 控制，跨 profiler run 应使用同 run trace 对齐，而不是
  假定固定 step；
- NCU 已完成一个真实 Prefill GEMM 的 counter 分析，但没有完成 Scheduler step
  唯一归属或 strict/bounded 同-shape 对照；
- 如果重新打开策略设计，应先定义可证明的 delay/KV bound，而不是继续叠加
  heuristic。

## 11. 复现与详细证据

- 环境：`docs/environment_v026.md`
- Workload 与 trace：`benchmarks/scheduler_trace/README.md`
- Scheduler trace：`docs/day4_scheduler_trace.md`
- Scheduler–MRV2：`docs/scheduler_to_model_runner.md`
- KV allocation/preemption：`docs/day6_kv_cache_allocation.md`、
  `docs/day7_preemption.md`
- HOL 策略：`docs/day13_waiting_hol_policy.md`
- 重复与公平性：`docs/day14_waiting_hol_benchmark.md`
- Bounded PR audit：`docs/day15_waiting_bypass_pr_audit.md`
- Profiling：`docs/day16_nsys_systems_gate.md`
- NCU Gate：`docs/day17_ncu_gate.md`

## 30 秒项目表达

我在 vLLM v0.26/MRV2 上实现了默认关闭的逐 step Scheduler Trace，把 token
budget、KV allocation、preemption、persistent row 和 GPU input shape 对齐。
Trace 发现 KV 压力下，一个放不下的长 Prefill 会阻塞后面本可容纳的短请求。
我实现了 opt-in bounded bypass：三请求中短请求 TTFT 从约 33 秒降到 0.18 秒，
但长生命周期 burst 新增 3 次 preemption，公平性、makespan 和吞吐量都退化。
Profiler 又证明策略改变的是 batch shape 和 kernel mix，而不是 kernel 代码。
因此我保留 patch 和反例作为工程证据，但否决了 upstream PR。
