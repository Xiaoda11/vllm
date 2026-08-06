# Day 17：Nsight Compute 单 Kernel Gate

## 决策

Day 17 的硬件计数器 Gate 已于 2026-08-06 通过。

最初，最小 PyTorch CUDA matmul 能够被 Nsight Compute attach，但采集因
`ERR_NVGPUCTRPERM` 停止；仅从提权的 WSL 进程启动仍然失败。在用户明确授权并
通过 NVIDIA Control Panel 允许 GPU performance counter 访问后，同一最小
matmul 成功生成了包含硬件计数器的报告。随后又对真实 vLLM workload 中的一个
Prefill GEMM 各采集了一份基础、完整和 warp-stall 报告。

这次采集回答了单 kernel 的微架构问题，但不是端到端 benchmark，也没有完成
Scheduler step 与 NCU kernel launch 的唯一归属。报告中的 launch shape 更接近
该 workload 的另一个 Prefill，因此不能把它直接写成 Day 16 的 C mixed step。

## Gate 结果

Gate 依次满足：

1. 用户明确授权修改 host performance-counter 权限；
2. 最小 CUDA matmul 生成非空 NCU 报告，9 个 replay pass 均完成；
3. 报告能够读取 SM、memory、cache、occupancy 和 warp-stall counter；
4. 收集范围缩小到真实 vLLM workload 的一个目标 GEMM kernel。

最小 matmul 的基础结果为：

- kernel：`volta_sgemm_128x64_nn`；
- duration：529.09 us；
- Compute (SM) throughput：89.83%；
- Memory throughput：43.57%；
- achieved occupancy：45.98%。

它的用途只是证明计数器链路已经工作，不参与 vLLM 策略结论。

## vLLM 单 Kernel 结果

目标 kernel 为：

```text
turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_tn
```

完整报告中该 launch 为 grid `(76,16,1)`、block `(128,1,1)`，主要指标如下：

| 指标 | 实测值 |
|---|---:|
| Duration | 2.03 ms |
| Compute (SM) throughput | 41.68% |
| Memory throughput | 33.21% |
| DRAM throughput | 25.84% |
| L2 throughput | 33.21% |
| L1/TEX throughput | 48.21% |
| Device memory throughput | 68.14 GB/s |
| L1 hit rate | 0.09% |
| L2 hit rate | 84.01% |
| Registers / thread | 254 |
| Static shared memory / block | 32.77 KB |
| Theoretical occupancy | 25.00% |
| Achieved occupancy | 24.67% |
| Active warps / SM | 7.90 |
| Waves / SM | 20.27 |

Occupancy 同时受寄存器和 shared memory 限制：两者都只允许每个 SM 驻留两个
thread block。高达 254 registers/thread 是需要记住的直接证据，而不是把
“occupancy 低”本身当作性能根因。

## Warp Stall

显式 stall 报告中，各采样 stall 比例如下：

| Stall reason | 比例 |
|---|---:|
| `math_pipe_throttle` | 61.85% |
| `wait` | 21.41% |
| `selected` | 5.10% |
| `mio_throttle` | 3.06% |
| `barrier` | 2.95% |
| `not_selected` | 2.69% |
| `long_scoreboard` | 1.41% |
| `short_scoreboard` | 0.46% |

`math_pipe_throttle` 是第一主因，`wait` 是第二主因，而代表长延迟内存依赖的
`long_scoreboard` 只有 1.41%。同时，SM throughput 41.68% 和 DRAM throughput
25.84% 都没有接近饱和。因此这个 launch 不呈现简单的 DRAM-bound 特征；更准确
的表述是：低 occupancy 下可参与调度的 warp 较少，观测到的 stall composition
主要是数学执行管线压力与固定延迟依赖。

这仍不能推出“提高 occupancy 一定加速”，也不能仅凭一个 launch 宣称整个
Prefill compute-bound。

## 与 Scheduler 策略的关系

Day 16 已经证明 bounded admission 改变了 Scheduler step 的 batch shape 和
kernel mix：它把 Prefill 插入 Decode 窗口，但没有修改 GEMM 或 attention kernel
代码。

Day 17 补充的是一个真实 vLLM Prefill GEMM 的硬件行为。它没有进行 strict 与
bounded 的同 step、同 shape NCU 对照，所以不能把上述指标解释成策略导致的
微架构退化。当前最强结论仍然是：策略先改变准入顺序和 Model Runner 输入 shape，
继而改变执行的 kernel 组合；稳定的策略性能结论来自 Day 14/15 的无 profiler
重复实验。

## 证据与限制

```text
/home/xiaoda/vllm-lab/outputs/day17-ncu-authorized-20260806/
├── matmul_basic.ncu-rep
├── vllm_gemm_basic.ncu-rep
├── vllm_gemm_full.ncu-rep
└── vllm_gemm_stalls.ncu-rep
```

- `basic`、`full` 和 `stalls` 使用不同 replay/section，报告中的 duration 有 profiler
  扰动，不能互相作为性能对照。
- NCU 目标 workload 为有意中止的 targeted-kernel run，其 request `status` 可能仍为
  `initializing`；它不是有效的 request latency 或 throughput 样本。
- 目标 launch 的 grid `(76,16,1)` 与 Day 16 C mixed-step 中看到的 `(76,9,1)`
  不同。没有 NVTX/launch-config 过滤前，精确 step 归属保持未决。
- Nsight Systems 在本机 WSL 路径仍未记录 GPU kernel activity；NCU counter 权限
  已解决不等于 Systems timeline 问题也已解决。

## 30 秒技术摘要

我先用单 CUDA matmul 做 Nsight Compute fail-closed Gate。初次 attach 因
`ERR_NVGPUCTRPERM` 失败；获得明确授权并启用 NVIDIA performance counter 后，
同一 Gate 成功。真实 vLLM Prefill GEMM 的 achieved occupancy 约 24.7%，每线程
使用 254 个寄存器，主要 stall 是约 61.9% 的 math-pipe throttle，而 long
scoreboard 只有约 1.4%，所以它不是简单的 DRAM-latency-bound 现象。不过这只是
一个 targeted launch，不能替代端到端 benchmark，也不能在没有 step 对齐时归因
给具体 Scheduler 策略。
