# Day 16：Profiling Gate 与 Scheduler-to-Kernel 证据

## 决策

Day 16 通过一条有完整记录的 fallback 路径完成。

- Nsight Systems 2025.6.3 在 vLLM 和最小 PyTorch matmul 上都只记录到了 CUDA API 调用，没有记录 GPU kernel activity。因此，这些报告不能作为 GPU timeline 证据。
- 显式使用 `--trace=cuda-sw` 得到了相同结果。
- vLLM 支持的 PyTorch Profiler 能够记录 CUDA kernel。我们分别为 strict 和 bounded waiting admission 采集了 Scheduler step 60-79 的有界窗口。
- 对全部 40 个已 profile 的 step，Scheduler trace 与 profiler annotation 中的 scheduled token 数完全一致。

结果回答了 Day 16 的受控问题：bounded admission 把 C 的 Prefill 插入 A 的 Decode 窗口，因此改变了执行形态和 kernel mix。它没有修改任何单个 kernel 的实现。

上游 PR 路线继续暂停。本次没有发布外部评论，也没有更新 PR。

## 源码事实与本地 Gate

NVIDIA 的 CUDA on WSL 指南说明，Volta 及更新架构支持 Nsight Systems CLI 和 CUPTI trace，但 profiler 还需要额外满足 Windows/driver 条件。当前 GPU 为 Turing，Windows driver 为 R596，但无法从本次 WSL 会话中恢复准确的 Windows build。

NVIDIA Nsight Systems release notes 说明，虚拟化环境可能不支持 CUDA hardware trace，并可能退回 legacy software trace。本地报告诊断确认发生了这种 fallback。但在本机上，无论 hardware mode 还是显式 software mode，最终都没有记录 GPU kernel activity。

官方参考资料：

- <https://docs.nvidia.com/cuda/wsl-user-guide/index.html>
- <https://docs.nvidia.com/nsight-systems/ReleaseNotes/index.html#cuda-trace-issues>
- 本仓库中的 `docs/contributing/profiling.md`

最小 Nsight Compute 运行最初能够到达 GPU，但失败并报告 `ERR_NVGPUCTRPERM`。
用户随后于 2026-08-06 明确授权修改 host performance-counter 权限；在 NVIDIA
Control Panel 开启访问后，最小 matmul 和真实 vLLM 单 GEMM counter Gate 均已
通过，详见 `docs/day17_ncu_gate.md`。这不改变本页的 Nsight Systems 结论：Systems
GPU kernel activity 仍未恢复。

## 受控问题

> Bounded waiting admission 是否只改变 Scheduler/请求顺序，还是也会改变 C 准入附近的 GPU 工作组成？

假设：

- Strict admission 使 profile 窗口保持为只有 A 的 Decode。
- Bounded admission 插入 C 的 1024-token Prefill，随后让 A/C 一起 Decode。
- mixed step 会引入面向 Prefill 的 GEMM 工作；这是 batch/shape 的变化，不是新的 kernel 实现。

判定标准：

1. 两次运行使用相同的 commit、模型、backend、KV 容量、token budget、eager mode 和 profiler 窗口。
2. 每个 profiler execution annotation 都与对应 Scheduler step 的 scheduled-token 总数一致。
3. 两份 trace 都包含真实 CUDA kernel 事件。
4. 结论必须区分观察到的 shape/kernel composition 和稳定性能结论。

## 实现

Commit `1e2f11bf54` 增加了仅用于实验室的有界 PyTorch profiler 模式：

```text
--torch-profile
--torch-profile-delay-iterations 60
--torch-profile-max-iterations 20
```

实际 profiler 配置会记录 shape，关闭 stack/memory/FLOPs，忽略 AsyncLLM frontend，并使用详细的 worker annotation。Profile 在 60 次 worker iteration 后开始，记录 20 个 step，从而避开模型加载和初始 Triton JIT。

自动分析器为 `scripts/lab_day16_profile_analyze.py`。它会验证 D13 HOL witness、请求完成情况、无 preemption、MRV2 shape 不变量、准确的 profile 窗口、Scheduler/profile token 一致性，以及非空 kernel activity，并输出 step CSV 和 JSON summary。

## Workload 与命令

两次运行都使用 MRV2、`TRITON_ATTN`、完整 ISL 预留、关闭 Prefix Cache、eager execution、2048-token budget，以及 1450-block KV override。唯一的策略变量是 waiting bypass 的关闭/开启。

```bash
VLLM_WSL2_ENABLE_PIN_MEMORY=1 \
VLLM_USE_V2_MODEL_RUNNER=1 \
/home/xiaoda/vllm-lab/.venv-v026/bin/python \
  scripts/lab_v026_workload.py \
  --config benchmarks/scheduler_trace/configs/d13_waiting_hol_blocking.json \
  --model /home/xiaoda/vllm-lab/models/Qwen2.5-0.5B-Instruct \
  --num-gpu-blocks-override 1450 \
  --waiting-bypass off \
  --scheduler-trace --token-timing \
  --torch-profile \
  --torch-profile-delay-iterations 60 \
  --torch-profile-max-iterations 20 \
  --run-id day16-torch-d13-strict-20260805
```

Bounded 命令只改变：

```bash
--waiting-bypass on --run-id day16-torch-d13-bounded-20260805
```

## Scheduler 与执行的对齐

| Profile 窗口 | Strict | Bounded |
|---|---:|---:|
| Scheduler steps | 60-79 | 60-79 |
| 单 Decode steps | 20 | 17 |
| Prefill/Decode mixed steps | 0 | 1 |
| 多 Decode steps | 0 | 2 |
| Profile/Scheduler token 不匹配数 | 0 | 0 |

Bounded 的 mixed step 是 Scheduler step 77：

```text
A Decode: 1 token
C Prefill: 1024 tokens
total scheduled: 1025 tokens
annotation: execute_1025_context_1(...sq1024...)_generation_1(...sq1...)
```

Step 78 和 79 各调度两个 generation token，A 和 C 各一个。Strict 的 step 60-79 每一步都只调度 A 的一个 Decode token。

在这次 bounded profiled run 中，C 首次出现在 step 77，而不是更早 Nsight 运行中的 step 68。请求到达由 wall clock 控制，profiler overhead 会改变 7.01 秒之前完成的 Scheduler iteration 数量。因此，本报告使用同一次运行中的 Scheduler/profile 对齐得出结论，不假设不同工具运行中的 step 编号固定不变。

## 观察到的 Kernel 组成

以下数值分别描述每种模式中的一个 20-step profiled window。PyTorch Profiler 会带来显著开销，因此这些结果不是吞吐量 benchmark。

| 指标 | Strict | Bounded |
|---|---:|---:|
| CUDA kernel 调用数 | 5,661 | 5,643 |
| Kernel duration 总和 | 162.173 ms | 299.080 ms |
| Unified-attention 调用数 | 480 | 480 |
| Unified-attention duration | 71.569 ms | 169.749 ms |
| GEMV duration | 83.282 ms | 70.649 ms |
| GEMM duration | 0 ms | 46.068 ms |
| 最长 annotated GPU range | 30.292 ms | 143.275 ms |

解释：

- 两个窗口每个 step 都执行 24 次 attention layer 调用：`24 × 20 = 480`。调用次数保持不变，但 bounded mixed step 改变了 query/context shape，因此 attention duration 发生变化。
- Strict 是纯单 token Decode，执行以 GEMV 为主。
- Bounded 用一个 1025-token mixed step 和两个 dual-Decode step 替换了三个 single-Decode step。Prefill 引入了 Turing FP16/CUTLASS GEMM。
- Mixed step 的 GPU range 为 143.275 ms；同一窗口中 strict 最长 step 为 30.292 ms。

这不能证明该策略整体更慢。Bounded 执行了 strict 有意推迟的有效 C 工作。Profile 解释的是工作被移动到哪里，以及 A 为什么可能出现 ITL 扰动；稳定的延迟/吞吐量判断仍应来自 Day 14/15 中没有 profiler overhead 的重复 benchmark。

## 请求与正确性检查

两次 profiled run 都满足：

- A/B/C 完成，并精确生成请求的 token 数；
- 包含同一 step 中 B-blocked/C-fits 的 HOL witness；
- preemption 为 0；
- MRV2 input-shape mismatch 为 0；
- B 的首次调度 step 保持为 517。

Profiled request timing 被有意排除在性能结论之外。例如，C 的 TPOT 会被 profiler flush/overhead 严重扭曲。

## 实验产物

```text
/home/xiaoda/vllm-lab/outputs/day16-torch-d13-strict-20260805
/home/xiaoda/vllm-lab/outputs/day16-torch-d13-bounded-20260805
/home/xiaoda/vllm-lab/outputs/day16-torch-analysis-20260805
/home/xiaoda/vllm-lab/outputs/day16-nsys-20260805
```

两个 `.pt.trace.json.gz` 文件可以直接在 Perfetto 中打开。`.nsys-rep` 文件仅作为失败 Gate 的证据保留。

## 结论

Bounded waiting admission 改变的不只是 CPU 队列决策。在观察窗口中，它把 Model Runner batch 从 20 个单 token Decode step，改成了包含一个“1024-token Prefill + Decode”step 和两个 dual-Decode step 的窗口。这种 shape 变化用 GEMM 工作替换了一部分 GEMV-heavy 执行，并延长了 mixed step。策略代码没有修改 attention/GEMM kernel；它改变的是现有 kernel 在什么时间、以什么 shape 运行。

## 未解决问题

能否进一步恢复 Nsight Systems 的 GPU kernel timeline？Nsight Compute 的
performance-counter 权限已解决，但允许所有用户访问 counter 会扩大 host 侧的
信息暴露面；不需要继续采集时可在 NVIDIA Control Panel 中恢复限制。

## 30 秒面试表达

我把一个有界的 20-step GPU profile 与 Scheduler JSONL 对齐，而不是 profile 整个 server。Strict admission 产生了 20 个单 token Decode step。Bounded admission 产生了 17 个单 Decode step、一个包含 A Decode 和 C 的 1024-token Prefill 的 mixed step，以及两个 dual-Decode step。这个 mixed step 引入了 Tensor Core GEMM，并把 annotated GPU range 从约 30 ms 的 Decode 最大值延长到 143 ms。这说明 Scheduler 策略改变了 batch shape 和 kernel composition，而不是 kernel 代码本身。
