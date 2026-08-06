# Day 16：个人验收与技术问答

## 验收规则

学习者能够不看报告回答问题 1-8，并完成 30 秒表达，即视为通过。不要求记忆准确的 kernel symbol 和每个小数。必须掌握的锚点是：

- strict：20 个 single Decode step；
- bounded：17 个 single Decode + 1 个 mixed Prefill/Decode + 2 个 dual Decode；
- mixed step：A Decode 1 + C Prefill 1024 = 1025 scheduled tokens；
- attention 调用次数相同，但 shape 和 duration 改变；
- Decode 以 GEMV 为主，Prefill 引入 GEMM；
- profiler 证据能够解释工作位置，但不是稳定性能 benchmark。

## 问题

1. 为什么前两份 Nsight Systems 报告即使包含约 111k 次 `cudaLaunchKernel` API 调用，仍然不能作为 GPU 证据？
2. 最小 matmul smoke test 如何进一步缩小 Nsight 失败原因的范围？
3. 为什么可以接受 PyTorch Profiler 作为 fallback？它不能证明什么？
4. 我们怎么知道 profile 精确覆盖 Scheduler step 60-79，而不只是大约相同的 wall-clock 区间？
5. 如何从 execution annotation 推导 bounded mixed step 的 1025 个 scheduled tokens？
6. Strict 与 bounded 的执行组成有什么区别？
7. 为什么两份 profile 中都有 480 次 unified-attention 调用？
8. 为什么 bounded 会引入 GEMM，而 strict 以 GEMV 为主？
9. 299 ms 对 162 ms 是否证明 bounded scheduling 慢了 84%？为什么？
10. 为什么“B 的首次调度 step 保持为 517”不足以构成完整的公平性/安全性结论？
11. 为什么 C 在这里首次调度于 step 77，而更早的运行中是 step 68？
12. 使用 Nsight Compute 指标之前还必须通过什么 Gate？

## 参考答案

1. API launch 记录只能证明 CPU 发出了 kernel launch，不能证明 Nsight 记录到了 kernel 的开始/结束活动。`cuda_gpu_kern_sum` 没有 kernel 表，因此 GPU gap、overlap、名称和 duration 都没有被观测到。
2. 单个 CUDA matmul 也以相同方式失败，排除了 vLLM multiprocessing 和复杂 workload 是唯一原因。显式 software trace 同样失败，把问题范围缩小到了本机 Nsight/WSL/host 路径。
3. 它是 vLLM 支持的 profiler，而且 smoke test 显示了真实 kernel 名称和 CUDA time。它有中高程度的 overhead，因此可以证明观察到的执行 shape 和 kernel composition，但不能证明稳定的端到端性能。
4. Profiler 配置了 delay 60 和 max 20。分析器把 20 个 execution annotation 与 Scheduler step 60-79 配对，并在 annotation token 总数与 Scheduler scheduled tokens 不一致时直接判定失败。
5. `context_1(sq1024...)` 表示一个具有 1024 个 query token 的 context/Prefill 请求。`generation_1(sq1...)` 表示一个具有 1 个 query token 的 Decode 请求。因此该 step 执行 `1024 + 1 = 1025` 个 token。
6. Strict 是 20 个 single Decode step。Bounded 是 17 个 single Decode、1 个 mixed Prefill/Decode 和 2 个 multi-Decode step。
7. Qwen 有 24 个 attention layer，两个窗口都包含 20 个 forward step，因此 `24 × 20 = 480`。Batch shape 改变的是 duration，不是 layer invocation 次数。
8. 单 token Decode 执行类似 matrix-vector 的线性运算。处理 1024-token Prefill 会形成更大的矩阵，并调用 Turing FP16/CUTLASS GEMM。
9. 不能。Bounded 执行了 strict 延后的 C 工作，而且 profiler 本身会增加 overhead。这些数值解释额外 mixed 工作的代价和位置；策略性能应由未使用 profiler 的重复运行决定。
10. Day 15 的长生命周期 burst 中，B 的首次 step 虽然不变，但后来新增了 3 次 preemption，公平性和 makespan 也发生退化。准入变化可以在 B 启动后继续改变 KV 压力。
11. 请求到达由 wall clock 控制。Profiling overhead 改变了 C 在 7.01 秒到达之前执行的 Scheduler step 数量。同一次运行内的 trace 对齐才是权威证据。
12. 必须先获得明确授权，并让最小单 kernel NCU smoke 成功。该 Gate 已于 2026-08-06 在 NVIDIA Control Panel 开启 performance-counter 访问后通过；现在可以解释有明确目标的 vLLM NCU 指标，但仍需区分单 launch 微架构证据、Scheduler step 归属和端到端性能。

## 30 秒回答

我把一个 20-step PyTorch GPU profile 与 Scheduler trace 对齐。Strict admission 执行了 20 个单 token Decode step。Bounded admission 插入了一个包含 A 的 1 个 Decode token 和 C 的 1024-token Prefill 的 step，之后又有两个 dual-Decode step。两个 profile 仍然都调用 attention 480 次，因为模型有 24 层、窗口有 20 个 step；但 mixed shape 引入了 Tensor Core GEMM，并使该 step 明显变长。因此，该策略改变的是现有 kernel 的执行时机和输入 shape，而不是 kernel 代码；profiler timing 也不能作为稳定吞吐量 benchmark。

## 2 分钟回答提纲

1. 从 D13 HOL 问题以及 strict/full-ISL 行为讲起。
2. 解释为什么只有 API 的 Nsight trace 没有通过 kernel-evidence Gate。
3. 说明 bounded PyTorch profile 窗口以及 Scheduler-token 校验。
4. 推导 step 77 的 `1 Decode + 1024 Prefill = 1025`。
5. 解释 GEMV-heavy Decode 与 GEMM-heavy Prefill。
6. 区分 kernel-composition 证据和重复策略 benchmark。
7. 最后结合 Day 15 公平性反例和不提交 PR 的决定收尾。
