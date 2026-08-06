# Day 13：Waiting Head-of-Line Bypass 策略

## 决策摘要

Day 13 结束已有 Prefill threshold 调参路线后，从 Day 7 的 KV 压力控制流中
定位并实现了一个新的、默认关闭的 Scheduler 策略：

> 当 waiting 队首请求因当前 KV 容量不足而 admission failure 时，本 step
> 临时跳过它，继续检查后续 waiting 请求；下一 step 仍先重试被跳过的请求。

新配置为 `scheduler_allow_waiting_bypass`，默认 `false`。它不关闭
`scheduler_reserve_full_isl`，不改变 running-request preemption，也不引入
priority、deadline、aging 或动态 token budget。

一次干净的 Baseline/Modified MRV2 GPU trace 已证明策略按设计生效：短请求 C
的 TTFT 从 33.832 秒降至 0.182 秒，而队首长请求 B 仍正常完成。由于每组目前
只有一次独立启动，延迟百分比是功能性、描述性证据，不是稳定 benchmark 结论。

## 问题与源码根因

默认 `scheduler_reserve_full_isl=true` 会在接纳 waiting request 前检查整个
当前输入是否能放入 KV cache。这个 guard 能避免 Chunked Prefill 过度接纳和
反复 preemption，应当保留。

问题发生在 `Scheduler.schedule()` 的 waiting loop：

```text
request = waiting.peek_request()
new_blocks = allocate_slots(..., full_sequence_must_fit=True)
if new_blocks is None:
    break
```

FCFS 队首长请求暂时放不下时，`break` 终止本 step 的 waiting traversal，后面
实际能放下的短请求不会被检查。这是 admission head-of-line blocking。

## 最小复现 workload

- A：8192 prompt / 512 output，先进入 Decode 并持有 KV blocks；
- B：16384 prompt / 32 output，7.00 秒到达，位于 waiting 队首；
- C：1024 prompt / 32 output，7.01 秒到达，严格排在 B 后；
- token budget：2048；
- KV pool：1450 blocks，即 23200-token capacity；
- block size：16；
- max model length：16416；
- full-ISL reservation：开启；
- Prefix Cache：关闭；
- Path A MRV2 / `TRITON_ATTN` / eager。

10 ms 到达差只用于固定异步提交顺序。一次探索 run 中 B/C 同时到达时，C 实际
早提交 0.432 ms，因而没有复现 HOL；该探索目录保留为
`day13-hol-baseline-kv1450-v2-20260804`，不作为 Baseline。

正式产物：

```text
/home/xiaoda/vllm-lab/outputs/day13-hol-baseline-kv1450-v3-20260804
/home/xiaoda/vllm-lab/outputs/day13-hol-bypass-kv1450-20260804
```

更早的 `day13-hol-baseline-kv1450-20260804` 在 Engine 初始化时失败，因为探索
配置的 max model length 32768 超过 1450 blocks 的 KV capacity；它没有进入
推理，不计入结果。

## Baseline trace 证据

step 53：

- running：A；waiting：[B, C]；
- A Decode 调度 1 token，剩余 token budget=2047；
- free blocks=933；
- B 完整 16K input 约需 1024 blocks，因此 waiting allocation failure；
- C 完整 1K input 只需 64 blocks，933 blocks 足以容纳；
- 当前 `break` 使 scheduled requests 只有 A，C 留在 waiting。

该状态持续到 A 完成：

- B 第一次调度：step 517；
- C 第一次调度：step 525；
- waiting allocation failures：465；
- preemptions：0；
- C TTFT/E2E：33.832/34.582 秒。

因此 C 的等待不是自身容量不足，也不是 preemption，而是 B 的队首失败终止了
waiting traversal。

## 策略实现

配置链：

- `SchedulerConfig.scheduler_allow_waiting_bypass=false`；
- `EngineArgs` 与 CLI `--scheduler-allow-waiting-bypass`；
- workload override `--waiting-bypass on|off|config`。

热路径只在 waiting allocation failure 分支增加：

```text
if allow_waiting_bypass:
    pop failed request from current queue
    prepend it to this step's local skipped queue
    continue traversing waiting requests
else:
    break
```

本 step 结束后，失败请求合并到已有 `skipped_waiting`。FCFS 下一 step 先选择
`skipped_waiting`，所以 B 会在新请求之前重试；只有它再次放不下时，后续可容纳
请求才继续 bypass。没有新增 per-request 状态或 GPU metadata。

复杂度：

- 默认关闭：与 baseline 相同；
- 开启：一个 step 最坏扫描当前 waiting queue 一次，时间 O(W)；
- 使用已有 request queue，额外空间 O(W)（本 step 暂存被跳过的引用）；
- 不增加 GPU-to-CPU 同步。

## Modified trace 证据

step 71：

- B 位于 `skipped_waiting` 并再次 allocation failure；
- Scheduler 将 B 暂存到本 step local skipped queue，继续检查 waiting 中的 C；
- A 调度 1 token，C 调度 1024 tokens；
- C 分配 64 blocks 后，B 仍排在 `skipped_waiting` 等待下一 step 重试；
- B 第一次调度仍是 step 517，与 Baseline 相同；
- C 第一次调度从 step 525 提前到 step 71；
- 所有请求准确完成，无 preemption。

请求级单次结果：

| Request / metric | Baseline | Modified | 描述性变化 |
|---|---:|---:|---:|
| A TTFT | 5.723 s | 5.470 s | -4.4% |
| A E2E | 19.320 s | 17.880 s | -7.5% |
| B TTFT | 33.669 s | 31.488 s | -6.5% |
| B E2E | 34.591 s | 32.278 s | -6.7% |
| C TTFT | 33.832 s | 0.182 s | -99.46% |
| C E2E | 34.582 s | 0.949 s | -97.26% |

C 是本策略的目标收益。A/B 的数值变化不能从单次冷启动解释为策略收益；两组
都包含 Triton JIT 和运行顺序噪声。

## 测试与默认行为

Scheduler 单元测试构造 10-block pool：A 已占用 blocks，B 位于 waiting 队首且
无法完整放入，C 位于其后且可以放入。参数化结果：

- flag=false：A 调度 1 token，B/C 保持 waiting；
- flag=true：A 调度 1 token、C 调度完整 prompt，B 进入 skipped waiting；
- 目标 Scheduler tests：2 passed；
- lab workload/analyzer tests：29 passed；
- Ruff 与 `git diff --check`：通过。

本地 Scheduler test 使用 `VLLM_TEST_MODEL` 指向本地 Qwen 配置，只为避免测试
夹具联网解析 `facebook/opt-125m`；测试本身不执行模型 forward。

## 风险与后续 Gate

- FCFS 语义：开启后允许能放下的后到请求临时越过放不下的早到请求，因此必须
  保持 opt-in；
- starvation：B 每 step 都先重试，但持续短请求可能不断 bypass；需在 burst
  benchmark 记录最长等待与 Jain fairness；
- 多个失败请求：已有本 step skipped queue 应保持其相对顺序，需补单元测试；
- Priority policy、LoRA constraint、encoder input cleanup、KV connector 和
  preempted request 需要边界测试；
- 正式性能结论至少需要 Baseline/Modified 各 3 次交错运行。
