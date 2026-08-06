# Scheduler Trace Lab：项目展示与简历材料

## 一句话定位

这是一个以真实 workload 驱动的 vLLM v0.26/MRV2 Scheduler 二次开发项目：
从逐 step 可观测性出发，定位 waiting HOL blocking，实现并验证 bounded bypass，
最后用反例否决不安全的发布方案。

## 展示页摘要

### 为什么做

请求级 TTFT/TPOT 只能告诉我们“慢了”，不能回答哪个 Scheduler step、哪次 KV
allocation 或哪个 MRV2 batch 造成了等待。本项目建立可复现的 trace，让调度
决策、KV 状态、MRV2 input shape 和 GPU 工作组成可以在同一个证据链中解释。

### 改了哪里

- 默认关闭的 Scheduler/MRV2 JSONL trace；
- 可控的精确 token workload generator；
- trace 到 CSV、benchmark aggregate 和 profiler alignment 分析器；
- 默认关闭的 one-admission waiting bypass；
- Scheduler、workload 和 analyzer 的针对性测试。

### 最重要的结果

- 复现真实 HOL witness：B full-ISL allocation failure 时 free blocks 放得下 C，
  但 strict FCFS 直接停止扫描；
- repeated 三请求实验中，bypass 将 C TTFT median 从 33.250 s 降至 0.182 s；
- 无界 burst 会把 B first step 从 517 推迟到 587，说明逐 step 重试不构成
  starvation bound；
- bounded 长生命周期 burst 新增 3 次 preemption，makespan +1.21%、throughput
  -1.20%、TTFT Jain -10.64%；
- 20-step profile 显示策略插入 1024-token Prefill，改变 batch shape 和 kernel
  mix，而不是 kernel 实现；
- NCU 对真实 Prefill GEMM 测得 24.67% achieved occupancy、84.01% L2 hit，
  主要 stall 为 61.85% `math_pipe_throttle`，不呈现简单 DRAM-latency-bound
  特征；该 targeted launch 不作具体 Scheduler step 或策略性能归因。

### 最终判断

不提交 upstream PR。一个优秀的系统项目不要求 patch 必须上线；能够用受控反例
证明局部优化缺少安全边界，并停止继续堆叠 heuristic，同样是工程能力证据。

## 快速复现入口

先验证配置，不启动 GPU：

```bash
/home/xiaoda/vllm-lab/.venv-v026/bin/python \
  scripts/lab_v026_workload.py \
  --config benchmarks/scheduler_trace/configs/d13_waiting_hol_blocking.json \
  --model /home/xiaoda/vllm-lab/models/Qwen2.5-0.5B-Instruct \
  --validate-only
```

运行 strict baseline：

```bash
VLLM_WSL2_ENABLE_PIN_MEMORY=1 VLLM_USE_V2_MODEL_RUNNER=1 \
  /home/xiaoda/vllm-lab/.venv-v026/bin/python \
  scripts/lab_v026_workload.py \
  --config benchmarks/scheduler_trace/configs/d13_waiting_hol_blocking.json \
  --model /home/xiaoda/vllm-lab/models/Qwen2.5-0.5B-Instruct \
  --num-gpu-blocks-override 1450 \
  --scheduler-trace --token-timing
```

Modified 只增加：

```text
--waiting-bypass on
```

每个 run 会在仓库外生成唯一目录，保存 commit、完整命令、resolved config、
request timing 和 Scheduler/MRV2 trace。详细复现矩阵见
`benchmarks/scheduler_trace/README.md`。

## 简历版本

### 两条精简版

- 基于 vLLM v0.26/MRV2 实现默认关闭的逐 step Scheduler Trace，对齐 token
  budget、KV allocation/preemption、persistent row 与 GPU input shape，并构建
  8K/16K Prefill、Decode/Prefill interleaving 等可复现 workload。
- 定位 KV 压力下 waiting HOL blocking 并实现 bounded bypass；重复实验将目标
  短请求 TTFT median 从 33.250 s 降至 0.182 s，同时以长生命周期 burst 的
  3 次 preemption 和公平性/吞吐退化否决 upstream PR，形成完整的收益—风险
  工程结论。

### 一条超短版

基于 vLLM v0.26/MRV2 构建 Scheduler–KV–Model Runner trace，定位 waiting HOL
blocking 并实现 bounded bypass，通过 GPU benchmark 与 profiler 证明局部 TTFT
收益及 preemption/fairness 代价，最终基于反例作出 no-PR 决策。

## 面试证据索引

| 追问 | 最直接的证据 |
|---|---|
| MRV2 在 WSL 怎么跑通 | `docs/environment_v026.md` |
| Trace 如何避免同步 | `docs/day4_scheduler_trace.md` |
| Scheduler 如何映射到 MRV2 | `docs/scheduler_to_model_runner.md` |
| KV allocation 与 preemption | `docs/day6_kv_cache_allocation.md`、`docs/day7_preemption.md` |
| 为什么没有重复实现 Prefill quantum | `docs/day11_policy_problem_selection.md` |
| HOL witness 与 patch | `docs/day13_waiting_hol_policy.md` |
| 收益和 starvation 反例 | `docs/day14_waiting_hol_benchmark.md` |
| bounded 为什么仍不安全 | `docs/day15_waiting_bypass_pr_audit.md` |
| 调度如何影响 GPU 工作 | `docs/day16_nsys_systems_gate.md` |
| NCU 单 kernel 说明什么 | `docs/day17_ncu_gate.md` |
| 完整项目叙事 | `docs/scheduler_trace_lab_final_report.md` |

## 展示时必须避免的表述

- 不说“优化了 vLLM 99%”；只说目标 C 在特定三请求 Gate 中的 TTFT 变化。
- 不把一次 burst 或 profiler window 描述成稳定性能结论。
- 不说运行了 FlashAttention；实测 backend 是 `TRITON_ATTN`。
- 不把 Scheduler logical step 当成 GPU kernel 时间戳。
- 不说 bounded 策略解决了 starvation；证据恰好说明它没有可靠的生命周期上界。
- 不把单个 NCU launch 外推成整个 Prefill 的 memory/compute-bound 结论，也不在
  grid 尚未对齐时归因给具体 Scheduler step 或策略。
