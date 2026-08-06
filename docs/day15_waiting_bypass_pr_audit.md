# Day 15：有界 waiting bypass 的 PR 审计

## 决策

当前的单次准入上限还不适合提交上游 PR。它限制了多少个后到请求可以绕过因 KV block 不足而阻塞的队首请求，但没有限制已准入请求占用 KV 的时长，也没有限制它后续造成的压力。长生命周期 burst 实验产生了 3 次 preemption，并使公平性、makespan 和吞吐量退化。短生命周期 burst 只安全地改善了 C1；中位数和尾部延迟基本没有变化。

这是一个有价值的调度负结果，但不能证明该功能应该发布。当前 patch 继续保持为默认关闭的 opt-in 功能，不应作为默认策略提出。

## 对比策略

Strict admission 保留 waiting policy 的队列顺序：当 B 无法为完整输入预留空间时，Scheduler 停止检查更年轻的请求。这不是 work-conserving 策略，但它可以保护 B，避免更年轻的任务继续分配 KV，并保持 FCFS 行为可预测。

Bounded bypass 为每个被阻塞的 waiting 请求提供一次让后续请求准入的机会。它阻止了 Day 14 中的无界 burst，但这个上限只约束准入次数。C1 一旦被准入，就成为 running 请求，并可能在长 Decode 过程中一直持有 KV。

## 当前 commit 的 GPU 证据

所有已完成的实验对都使用 MRV2、`TRITON_ATTN`、完整 ISL 预留、1450-block KV override、eager execution，并关闭 Prefix Cache。

### 三请求 workload

| 指标 | Strict | Bounded |
|---|---:|---:|
| B TTFT | 33.240 s | 31.757 s |
| C TTFT | 33.407 s | 0.179 s |
| B 首次调度 step | 517 | 517 |
| Preemption 次数 | 0 | 0 |

这验证了基本机制，但单个小 C 无法检验已准入请求的后续生命周期影响。

### 长生命周期 burst：8 个 1K prompt / 512 output 请求

| 指标 | Strict | Bounded | 变化 |
|---|---:|---:|---:|
| B TTFT | 31.229 s | 31.863 s | +2.03% |
| C TTFT 中位数 | 31.931 s | 32.142 s | +0.66% |
| Makespan | 52.603 s | 53.241 s | +1.21% |
| 输出吞吐量 | 88.208 tok/s | 87.150 tok/s | -1.20% |
| TTFT Jain 指数 | 0.930977 | 0.831934 | -10.64% |
| B 首次调度 step | 517 | 517 | 不变 |
| C 首次调度 step 范围 | 525-557 | 72-554 | 只有 C1 非常早 |
| Preemption 次数 | 0 | 3 | 退化 |

Bounded 下 C1 的 TTFT 为 0.174 秒，而另外 7 个 C 请求仍约为 31.8-33.1 秒。C1 持续占用的 KV 加剧了内存压力，导致 C6、C7、C8 被 preempt。因此，“B 的首次调度 step 不变”不足以构成安全性结论：准入变化会改变系统之后的行为。

### 短生命周期 burst：8 个 1K prompt / 32 output 请求

| 指标 | Strict | Bounded | 变化 |
|---|---:|---:|---:|
| B TTFT | 31.243 s | 31.440 s | +0.63% |
| C1 TTFT | 31.472 s | 0.173 s | -99.45% |
| C TTFT 中位数 | 31.938 s | 32.018 s | +0.25% |
| Makespan | 40.693 s | 40.771 s | +0.19% |
| Preemption 次数 | 0 | 0 | 不变 |
| B 首次调度 step | 517 | 517 | 不变 |

当 C1 在 B 能够进入之前就结束时，本次运行中的 bypass 是安全的。但它仍然只让一个由队列位置决定的请求受益，并没有改善队列的中位数或尾部延迟。这些较小的总体差异只是单次运行的描述性数值，不能作为稳定的性能结论。

### 现有替代方案：关闭完整 ISL 预留

D13 的替代运行在 Scheduler step 166 时被手动终止，此前 B 已发生 12 次 preemption。B 反复执行到大约 14,329 个 computed tokens，随后被 preempt 并重新开始。该运行是被截断的数据，不能报告为一次成功的对比实验。但它确实表明，在这种内存压力下，取消完整输入预留并不是安全的替代方案。

## 风险与语义

- FCFS 从严格顺序变为条件顺序。恰好一个更年轻的请求会因为队列位置获得很大优势。
- 该上限既不是 starvation bound，也不是 wall-clock bound。一个被准入的请求可能长时间 Decode、等待远端 KV，或持有大量 KV。
- 后续的 preemption 和 recomputation 可能抵消表面上的利用率收益。
- Priority scheduling 本来就允许后到的高优先级请求绕过 B，因此“允许一个后续请求绕过”的简单描述只在 FCFS 或相同/更低优先级请求下成立。
- token 数阈值会依赖硬件和 workload。再增加一个启发式阈值也无法提供有原则的“不延迟”保证。

真正保守的 backfill 策略，只有在能够证明 C 不会延迟 B 时才准入 C。但 vLLM 不知道 C 的完成时间，而预留未来 KV 或强制 preempt 被绕过任务都会变成规模大得多的调度设计。

## 上游相关性

上游 Draft PR `vllm-project/vllm#33499` 已经提出无界 skip 方向。再开一个等价 PR 会造成重复。对上游更有价值的信息是这个反例：每一步都优先重试被阻塞的队首请求，并不能阻止更年轻的已准入请求持续占用 KV，也不能防止它们延迟后续工作或破坏系统稳定性。

本项目没有发布任何外部评论。只有在决定公开本地 benchmark 结果后，才可以考虑把这些证据提供给该 PR。

## 复现实验产物

```text
/home/xiaoda/vllm-lab/outputs/prgate-d13-strict-r1-20260804
/home/xiaoda/vllm-lab/outputs/prgate-d13-bounded-r1-20260804
/home/xiaoda/vllm-lab/outputs/prgate-d13-partial-r1-20260804
/home/xiaoda/vllm-lab/outputs/prgate-d14b-strict-r1-20260804
/home/xiaoda/vllm-lab/outputs/prgate-d14b-bounded-r1-20260804
/home/xiaoda/vllm-lab/outputs/prgate-d14b-bounded-analysis-20260804
/home/xiaoda/vllm-lab/outputs/prgate-d15-short-strict-r1-20260804
/home/xiaoda/vllm-lab/outputs/prgate-d15-short-bounded-r1-20260804
/home/xiaoda/vllm-lab/outputs/prgate-d15-short-analysis-20260804
```
