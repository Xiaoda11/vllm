# Day 14：Waiting HOL 重复实验与公平性 Gate

## 决策摘要

Day 14 对默认关闭的 `scheduler_allow_waiting_bypass` 完成了两层验证：

1. 标准 D13 三请求 workload 的 Baseline/Modified 交错重复各 3 次；
2. 一个由 8 个短请求组成的 burst 压力对照。

标准重复证明收益稳定：C TTFT 中位数从 33.250 秒降到 0.182 秒，B 第一次
调度在六次运行中均为 step 517，没有观察到 B 退化。所有运行均准确完成，具有
同 step HOL witness，无 preemption 和 MRV2 input-shape mismatch。

burst 对照同时证明当前策略存在明确公平性代价：8 个短请求的 TTFT 中位数从
32.038 秒降到 0.706 秒，makespan 降低 20.0%，但 B 第一次调度从 step 517
推迟到 587，B TTFT 增加 10.0%。因此 Day 14 的上游 PR Gate 结论是：

> 问题、收益和最小实现成立，但当前 unbounded bypass 不能直接作为默认上游
> 行为提交；需要先设计有界 bypass/aging，或者与维护者确认 opt-in 策略语义。

## 假设、观测量与判定标准

### 标准重复

假设：单次 Day 13 的 33.8 秒到 0.18 秒变化不是启动顺序或 GPU 噪声造成的，
并且一个短请求 bypass 不会延迟 B。

观测量：

- A/B/C TTFT、E2E；
- makespan、output tokens/s；
- B/C 第一次调度 step；
- waiting allocation failures；
- preemption 和 MRV2 input-shape mismatch；
- B failure 时 C 是否仍在其后且 free blocks 足以容纳 C。

判定标准：Baseline/Modified 各至少 3 次；每次 B 严格先于 C 提交；所有请求
完成；每次存在 HOL witness；Modified 每次都让 C 先于 B 调度且不推迟 B。

### burst 公平性 Gate

假设：如果持续短请求能够占用 B 最终需要的 KV，那么逐 step 优先重试 B 仍可能
无法避免 B 被推迟。

burst workload：

- A：8192 prompt / 512 output，`t=0`；
- B：16384 prompt / 32 output，`t=7.00s`；
- C1–C8：各 1024 prompt / 512 output，`t=7.01–7.08s`；
- 1450 blocks、block size 16、token budget 2048、max sequences 10；
- full-ISL reservation 开启、Prefix Cache 关闭、MRV2、`TRITON_ATTN`、eager。

该 pair 只用于发现公平性边界，目前每种策略一次，性能百分比是描述性证据。

## 标准三请求重复结果

运行顺序：B1 → M1 → B2 → M2 → B3 → M3。六次运行均记录干净 commit
`e7d7ec67a748c809b471b8cbbddd18eb8be68a53`。

| 中位数指标 | Baseline | Modified | 变化 |
|---|---:|---:|---:|
| A TTFT | 5725.702 ms | 5675.024 ms | -0.885% |
| B TTFT | 33080.985 ms | 32943.819 ms | -0.415% |
| C TTFT | 33249.554 ms | 182.052 ms | -99.452% |
| A E2E | 18397.439 ms | 18297.785 ms | -0.542% |
| B E2E | 34017.797 ms | 33668.591 ms | -1.027% |
| C E2E | 34008.117 ms | 954.356 ms | -97.194% |
| makespan | 41018.745 ms | 40669.096 ms | -0.852% |
| output tokens/s | 14.042 | 14.163 | +0.862% |
| B first scheduled step | 517 | 517 | 0 |
| C first scheduled step | 525 | 61 | -464 steps |

逐次结果也稳定：

- Baseline C TTFT：33.250、33.261、33.072 秒；
- Modified C TTFT：0.182、0.174、0.183 秒；
- Baseline C first step：三次均为 525；
- Modified C first step：62、61、59；
- B first step：六次均为 517。

六次自动 Gate 全部满足：

- B 在 C 前提交；
- B failure 时 C 完整输入能放入当前 free blocks；
- Baseline 每次 B 先于 C 调度；
- Modified 每次 C 先于 B 调度；
- 所有请求准确完成；
- preemption=0；
- MRV2 input-shape mismatch=0。

## burst 公平性结果

burst config 与原始 trace 记录 commit
`3a7132eebe95bf456fd816e650d0bf03e0d7b10e`。

| 指标 | Baseline | Modified | 描述性变化 |
|---|---:|---:|---:|
| B TTFT | 31.332 s | 34.465 s | +9.997% |
| B E2E | 32.810 s | 35.198 s | +7.278% |
| C1–C8 TTFT 中位数 | 32.038 s | 0.706 s | -97.796% |
| C1–C8 TTFT 最大值 | 32.998 s | 0.944 s | -97.138% |
| makespan | 52.747 s | 42.199 s | -19.997% |
| output tokens/s | 87.967 | 109.955 | +24.996% |
| B first scheduled step | 517 | 587 | +70 steps |
| C first step 范围 | 525–557 | 72–75 | 提前 |
| allocation failures | 476 | 516 | +8.403% |

两组均准确完成、无 preemption、无 MRV2 shape mismatch，并具有 B 队首失败而
后续 C 能放入的 HOL witness。

Modified 中 Scheduler 每 step 的确先重试 B，但 C1–C8 已经被接纳并持有 KV，
所以 A 完成后可用 blocks 仍不足以接纳 B。B 必须继续等待部分 C 完成。这说明
“重试优先级”只约束检查顺序，不能撤销之前的 admission 决策，也不能独立提供
starvation bound。

报告中的 TTFT Jain index 仅描述请求延迟相等程度，不能单独解释为策略好坏：
Baseline 中所有请求一起等待会得到很高的相等度，Modified 大幅帮助短请求后反而
降低相等度。PR 判断以 B 的绝对退化、首次调度推迟、短请求收益和 makespan
共同评估。

## Scheduler 边界测试

Day 14 将目标 Scheduler coverage 从 2 个参数化 case 扩展为 4 个通过 case：

- flag=false 保持原 `break` 行为；
- flag=true 允许一个可容纳请求 bypass；
- 两个连续失败请求保持原 FCFS 相对顺序，并排在新 waiting work 之前；
- Priority policy 下高优先级失败请求仍留在优先队列，低优先级可容纳请求可运行。

Lab tests 为 42 passed。LoRA、encoder cleanup、KV connector 和 preempted-request
组合尚未动态覆盖；由于 burst Gate 已证明策略语义需要先收紧，这些组合测试推迟到
新策略形态确定后，避免为可能被替换的 unbounded 语义扩展测试。

目标 waiting-bypass Scheduler tests 为 4 passed。尝试运行整个
`tests/v1/core/test_scheduler.py` 时，12 个测试通过后，一个无关的 encoder stats
测试进入外部 tokenizer 的 SSL 访问等待；154 秒后人工终止，因此不将完整文件
报告为通过，也不把该环境阻塞解释为本 patch 的测试失败。

## PR Gate 与下一步设计

源码事实：最新 `upstream/main` 在 waiting allocation failure 后仍直接
`break`，没有等价 transient bypass；上游对永久不可能容纳请求的拒绝修复不能
解决本实验中“空闲时能放下、当前暂时放不下”的 B。

实测事实：当前策略稳定消除了后续短请求 HOL，但 burst 中会让 B 的接纳推迟。

工程推断：下一版至少需要一种 starvation bound，例如：

1. 对同一 blocked head 限制可 bypass/admit 的后续请求数；
2. 记录 bypass age/count，达到阈值后停止接纳更年轻请求；
3. 在不引入复杂 admission reservation 的前提下，先选择一个可单元测试的有界规则。

在该规则确定前，不创建官方 vLLM PR。当前 patch 仍是完整、可复现的 fork 策略
实验，也是设计下一版 bounded policy 的 baseline。

## 原始产物

标准重复：

```text
/home/xiaoda/vllm-lab/outputs/day14-hol-{b1,m1,b2,m2,b3,m3}-20260804
/home/xiaoda/vllm-lab/outputs/day14-hol-analysis-20260804
```

burst：

```text
/home/xiaoda/vllm-lab/outputs/day14-hol-burst-baseline-20260804
/home/xiaoda/vllm-lab/outputs/day14-hol-burst-modified-20260804
/home/xiaoda/vllm-lab/outputs/day14-hol-burst-analysis-20260804
```

## 30 秒技术摘要

我先用三次交错重复确认 waiting bypass 将后续 1K 请求 TTFT 从 33.25 秒稳定降到
182 毫秒，长请求首次调度 step 不变。随后我主动构造 8 个长 Decode 短请求的
公平性反例：短请求 TTFT 中位数降到 706 毫秒、makespan 降低 20%，但它们持有
KV 后把队首 16K 请求推迟 70 steps，TTFT 增加 10%。因此我没有急着提 PR，而是
把当前实现保留为 baseline，下一版增加 bypass count/age bound。这体现了调度策略
不能只看短请求收益，还必须验证 admission 决策对长请求公平性的代价。
