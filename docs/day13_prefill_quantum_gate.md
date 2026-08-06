# Day 13：Prefill Quantum Gate 与策略换题

## 结论

Day 13 关闭 Prefill quantum 新代码路线，不执行 Gate C/D，也不实现
mixed-only threshold。

原因不是该配置无效，而是它已经有效且没有暴露需要新代码解决的代价：

- B=16K mixed workload 中，`long_prefill_token_threshold=2048` 将 A 在 B
  Prefill 阶段的 ITL P95 中位数降低 66.1%；
- 纯 16K Prefill 的 TTFT/E2E 中位数只增加 0.3%/0.4%，input throughput
  中位数下降 0.4%；
- 12 次 MRV2 GPU run 均完成准确输出，无 preemption、allocation failure 或
  input-shape mismatch；
- 非混合退化远低于预注册的 5% Gate，因此代码 Gate 未触发。

继续扩展 threshold 矩阵只会变成现有 Chunked Prefill 配置调参，不能完成
“一个小而真实的 Scheduler 策略修改”的里程碑。下一步返回 Day 11，从已经
测得的 KV 压力控制流选择新问题：full-ISL admission failure 导致的 waiting
head-of-line blocking。

## 环境与产物

- commit：`67e0a4542b6c80c90229cc964495de84735681c8`；
- 环境：`/home/xiaoda/vllm-lab/.venv-v026`；
- Path A：`VLLM_USE_V2_MODEL_RUNNER=1`、
  `VLLM_WSL2_ENABLE_PIN_MEMORY=1`；
- backend：`TRITON_ATTN`；
- 模型：Qwen2.5-0.5B-Instruct FP16；
- global token budget：8192；
- Baseline/Tuned threshold：0/2048；
- 原始目录：`/home/xiaoda/vllm-lab/outputs/day13-gate-a-*-20260804` 和
  `/home/xiaoda/vllm-lab/outputs/day13-gate-b-*-20260804`；
- 汇总目录：
  `/home/xiaoda/vllm-lab/outputs/day13-gate-final-analysis-20260804`。

## Gate A：B=16K mixed workload

S5 固定 A=1K prompt/512 output、B=16K prompt/32 output、B 在 1 秒到达；
threshold 0/2048 各独立启动 3 次。

| 中位数指标 | threshold=0 | threshold=2048 | 变化 |
|---|---:|---:|---:|
| A Prefill-stage ITL mean | 4300.284 ms | 2157.039 ms | -49.8% |
| A Prefill-stage ITL P95 | 13815.885 ms | 4679.600 ms | -66.1% |
| B TTFT | 21495.259 ms | 21565.559 ms | +0.3% |
| B E2E | 22314.780 ms | 22376.244 ms | +0.3% |
| Output throughput | 15.653 token/s | 15.571 token/s | -0.5% |

Trace pattern 在三次重复中完全一致：

- threshold=0：B Prefill 为 `8191×2 + 2×1`，最大 mixed batch=8192；
- threshold=2048：B Prefill 为 `2048×8`，最大 mixed batch=2049。

这与 Day 11 的 B=8K 结果方向一致，满足 mixed ITL P95 至少改善 30% 的
第一项 Gate。

## Gate B：纯 16K Prefill

S2 固定 16K prompt/32 output，按 `0 → 2048` 交错顺序各运行 3 次。

| 中位数指标变化 | threshold=2048 相对 0 |
|---|---:|
| TTFT | +0.323% |
| E2E | +0.384% |
| Input throughput | -0.383% |

Trace pattern 从 `8192×2` 变为 `2048×8`，但三项请求级指标均远低于 5%
退化门槛；所有 run 的 preemption、allocation failure 和 MRV2 shape mismatch
均为 0。因此第二项 Gate 明确不成立，完整代码 Gate 为 false。

## 为什么停止而不是继续 Gate C/D

Gate C/D 的目的原本是寻找全局 threshold 的明显副作用。Gate A/B 已经证明：

1. 现有配置在 8K/16K mixed workload 都稳定改善目标指标；
2. 纯 8K/16K workload 都没有明确退化；
3. 当前没有 mixed-only 分支要修复的已测问题。

继续增加并发点可以扩充 benchmark，但不会改变“这是已有配置”的事实。它可以
保留为后续 Nsight 的 Baseline/Tuned 案例，不能再占用策略 patch 主线。

## 新问题：waiting head-of-line blocking

Day 7 已证明默认 `scheduler_reserve_full_isl=true` 会在 waiting admission 时
检查整个输入是否能放入 KV cache。这个 guard 能防止 over-admission 和反复
preemption，本身不应删除。

新的源码问题发生在分配失败之后：waiting loop 对队首请求调用
`allocate_slots(..., full_sequence_must_fit=True)`；返回 `None` 时直接 `break`。
因此，只要 FCFS 队首长请求暂时放不下，排在它后面、实际能放下的短请求也不会
在本 step 被检查。这是可由 trace 验证的 head-of-line blocking，不是已有
threshold 的重新包装。

下一最小 workload：

- A：8K prompt + 512 decode，先占用约 513 个 KV blocks；
- B：16K prompt，随后到达并位于 waiting 队首，需要约 1024 blocks；
- C：1K prompt，在 B 后 10 ms 到达，只需要约 64 blocks；10 ms 间隔用于
  固定异步提交顺序，不作为性能变量；
- KV pool：1450 blocks；full-ISL reservation 保持开启。
- max model length：16416，与 B 的 16K prompt + 32 output 边界一致，且不超过
  1450 blocks 提供的 23200-token KV capacity。

预期 baseline：B 因剩余 blocks 不足而 admission failure，当前 `break` 使 C
一同等待。候选默认关闭策略只在 waiting allocation failure 时，把本轮失败的
请求临时放入已有 `step_skipped_waiting`，继续检查后续请求；下一 step 仍先重试
B。它不关闭 full-ISL guard，不改变 running preemption，也不引入 priority、
deadline 或动态 token budget。

是否实现该策略，先由 baseline GPU trace 和 Scheduler 单元测试证明：B 失败时
C 的完整 input 确实能由当时 free blocks 容纳，但 C 因队首 `break` 未被调度。

## 30 秒技术摘要

我没有把已有 long-prefill threshold 包装成新功能。12 次 MRV2 重复实验显示，
2048 quantum 在 16K mixed workload 将 Decode ITL P95 中位数降低 66.1%，
纯 16K Prefill 的 TTFT/E2E 只退化约 0.3%–0.4%，所以预注册的新代码 Gate
没有触发。我终止了这条配置调参路线，转而从 KV 压力 trace 定位到真正的
Scheduler 控制流问题：full-ISL 检查让队首长请求失败后直接 break，可能阻塞
后面能放下的短请求。下一步用三请求和 1450-block KV pool 验证这个 HOL 问题。
