# Day 7：显存压力、分配失败与 Preemption

## 问题、假设与判定标准

本日回答三个问题：

1. KV Cache 容量不足时，`allocate_slots()` 在哪里失败？
2. FCFS Scheduler 会抢占谁，请求怎样恢复？
3. 抢占丢弃了多少已计算 token，对 TTFT、TPOT 和 E2E 有什么描述性影响？

实验前假设：

- 固定请求和 2048-token 全局 budget，只缩小 KV block 数，可以稳定复现
  running-request allocation failure。
- 被抢占请求会释放全部 blocks，`num_computed_tokens` 清零并回到 waiting。
- Prefix Cache 关闭时，恢复请求必须从 token 0 重算。

判定标准：

- trace 必须同时出现 running allocation failure 和
  `preempted_request_ids`；
- 必须记录失败候选、free blocks、抢占 victim 及其清零前的 computed tokens；
- 请求最终完成，且 baseline 不出现 allocation failure。

## 环境与正式产物

- vLLM：v0.26.0；
- 实验提交：`aee2d7facc22d20515fcabb8479ead80eabe56a9`；
- 两组 `run_metadata.json` 的 `git_status_short` 均为空；
- GPU：RTX 2060 Laptop 6 GiB / WSL2；
- `VLLM_USE_V2_MODEL_RUNNER=1`；
- `VLLM_WSL2_ENABLE_PIN_MEMORY=1`；
- attention backend：`TRITON_ATTN`；
- block size：16 tokens；
- token budget：2048；
- Prefix Cache：关闭；
- `scheduler_reserve_full_isl=false`；
- MRV2 async scheduling：开启。

正式产物：

```text
/home/xiaoda/vllm-lab/outputs/day7-baseline-default-kv-20260730
/home/xiaoda/vllm-lab/outputs/day7-pressure-kv1450-20260730
```

每个目录包含 `run_metadata.json`、`request_timing.csv`、
`scheduler_trace.jsonl`、`scheduler_trace.mrv2.jsonl` 和
`scheduler_trace.csv`。

## 受控 workload

两组使用完全相同的请求：

| Request | Prompt | Output | Arrival |
|---|---:|---:|---:|
| A | 8192 | 16 | 0 s |
| B | 16384 | 32 | 0 s |

A 完成 prefill 后进入短 decode，B 同时做 long partial prefill。两组唯一容量
差异是：

| 组 | KV blocks | KV token capacity |
|---|---:|---:|
| Baseline | 17243（profiled） | 275888 |
| Pressure | 1450（override） | 23200 |

`num_gpu_blocks_override` 是 v0.26 的原生测试配置。它覆盖 profile 得到的
block 数，因此没有通过改变模型、dtype 或 prompt 来间接制造压力。

## v0.26 的 full-ISL admission guard

v0.26 默认 `scheduler_reserve_full_isl=true`。waiting request 调用
`allocate_slots(..., full_sequence_must_fit=True)` 时，会先检查整个当前输入
序列是否能放入 KV Cache，而不只检查本轮 chunk。

探索运行
`/home/xiaoda/vllm-lab/outputs/day7-pressure-exploratory-20260730`
保留了这个默认值。在 A 占用 514 blocks、只剩 936 blocks 时，B 的完整
16384-token input 需要 1024 blocks，因此 B 连续 16 个 step 在 waiting
admission 失败，没有进入 running，也没有 preemption。

这说明默认 guard 的目标就是防止 chunked-prefill 过度接纳和 KV thrashing。
为了观察 Day 7 指定的 running preemption 路径，正式 baseline 和 pressure
两组都显式关闭该 guard。默认 guard 下的 waiting failure 不能报告成
preemption。

## 源码证据链

### 1. allocation failure

对 running request，`Scheduler.schedule()` 先计算 `num_new_tokens`，再调用：

```text
KVCacheManager.allocate_slots(request, num_new_tokens)
```

`allocate_slots()` 计算本轮结束后需要的 blocks。如果
`required_blocks > available_blocks`，返回 `None`，不进行部分分配。

### 2. victim 选择

FCFS policy 下，Scheduler 从 running queue 尾部 `pop()` 最低顺序请求。
本实验失败候选 B 本身位于尾部，所以两次 victim 都是 B，而不是正在 decode
的 A。

### 3. preemption

`Scheduler._preempt_request()`：

```text
释放 B 的 KV blocks
→ status = PREEMPTED
→ num_computed_tokens = 0
→ num_preemptions += 1
→ prepend 回 waiting queue
```

Prefix Cache 关闭，因此后续 admission 没有本地 cached tokens 可以恢复，
B 从 token 0 重做 prefill。

## GPU trace：失败、释放与恢复

两次失败发生在 step 12 和 step 20：

| Step | B computed before | B blocks | New tokens requested | Free blocks | Result |
|---:|---:|---:|---:|---:|---|
| 12 | 14329 | 896 | 2047 | 40 | B preempted |
| 20 | 14329 | 896 | 2048 | 40 | B preempted |

失败点可以直接由 block 数解释：

- B 已有 896 blocks。
- 本轮后 token 边界需要
  `ceil((14329 + 2047) / 16) = 1024` blocks。
- B 还需要 128 blocks，但全局只剩 40。
- B 释放 896 blocks 后，free blocks 从 40 增加到 936。

第一次恢复：

```text
step 12: B RUNNING(14329) → PREEMPTED(0)，896 blocks → 0
step 13: B PREEMPTED(0) → RUNNING(2047)，重新取得 128 blocks
```

第二次恢复：

```text
step 20: B RUNNING(14329) → PREEMPTED(0)，896 blocks → 0
step 21: A 已完成并释放 blocks；B 从 0 调度 2048 tokens
```

B 在整次压力运行中被调度 45073 tokens。无抢占时完成请求需要
`16384 + 31 = 16415` scheduled tokens，两次被丢弃的进度为：

```text
14329 + 14329 = 28658 recomputed tokens
16415 + 28658 = 45073 total scheduled tokens
```

因此本实验中的额外 B prefill work 与 trace 完全闭合。

## Baseline 与压力组

| 指标 | Baseline | 1450 blocks |
|---|---:|---:|
| Scheduler steps | 46 | 61 |
| Allocation failures | 0 | 2 |
| Preemptions | 0 | 2 |
| Recomputed tokens | 0 | 28658 |
| Peak KV usage | 8.92% | 97.24% |
| Minimum free blocks | 15704 | 40 |

请求级 timing：

| Request / metric | Baseline | 1450 blocks | 描述性变化 |
|---|---:|---:|---:|
| A TTFT | 5.563 s | 5.719 s | +2.8% |
| A TPOT | 1.438 s | 2.237 s | +55.6% |
| A E2E | 27.130 s | 39.269 s | +44.7% |
| B TTFT | 26.980 s | 61.271 s | +127.1% |
| B TPOT | 0.02372 s | 0.02360 s | -0.5% |
| B E2E | 27.716 s | 62.002 s | +123.7% |

B 的主要代价发生在 first token 之前：两次 14K prefill 被丢弃，所以 TTFT
和 E2E 大幅增加；B 进入稳定 decode 后，TPOT 基本不变。A 没有被抢占，但
B 的重复大 prefill 与 A decode 交错，使 A 的平均 TPOT 和 E2E 也恶化。

两组都是独立 Engine 冷启动，均包含 Triton JIT，且每组只有一次正式运行。
这些百分比只描述本次受控 trace，不能作为稳定性能收益或回归结论。抢占次数、
失败点、释放 blocks 和重计算 token 数则由逐 step trace 直接证明。

## 结论

显存压力不是“GPU OOM 后再恢复”。本实验中模型和 KV tensor 初始化均成功，
运行期是 block pool 的逻辑容量不足：`allocate_slots()` 原子地返回 `None`，
Scheduler 选择 victim、释放其 KV blocks、把 computed progress 清零，再从
waiting 恢复。

v0.26 的默认 full-ISL admission guard 会提前把不够完整 input 容量的新请求
留在 waiting，从而避免本实验中的 thrashing。关闭 guard 后，1450-block
配置稳定复现两次 B 自抢占，代价是 28658 个重复 prefill tokens，并显著推高
B TTFT 和 A 的 decode 间隔。

未解决问题：在更大并发下，默认 full-ISL guard、watermark 和允许
over-admission 三种策略怎样权衡 KV 利用率、waiting latency 与 preemption
代价，需要后续 benchmark matrix 才能回答。

## 30 秒面试表达

我用 vLLM v0.26 原生 `num_gpu_blocks_override` 把 KV pool 从 17243 blocks
缩到 1450，固定 8K decode 请求和 16K prefill 请求。B 在 14329 tokens 时
扩容需要 128 blocks，但只剩 40，于是 `allocate_slots` 返回失败，FCFS 将
队尾 B 自身抢占，释放 896 blocks、computed tokens 清零并从 waiting 重跑。
同一点发生两次，累计重算 28658 tokens。B 的 TTFT 从 27 秒升到 61 秒；而
v0.26 默认 full-ISL guard 会提前阻止过度接纳，所以默认配置下表现为 waiting
而不是 preemption。
