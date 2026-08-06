# Day 20：模拟面试与 PR Gate

## 最终 PR 决策

当前 waiting bypass patch 只保留在实验 fork，不向 upstream 提交，不发布外部
评论。这个决定不是因为问题不存在，而是因为已有数据证明当前 bound 不足以支撑
安全的通用策略。

| Gate | 结果 | 依据 |
|---|---|---|
| 问题真实且可稳定观察 | 通过 | 同 step B-blocked/C-fits HOL witness |
| 默认行为不变 | 通过 | feature 默认关闭，strict tests 覆盖 |
| 修改范围可解释 | 通过 | waiting admission 与少量生命周期状态 |
| 有明确局部收益 | 通过 | C TTFT median 33.250 s → 0.182 s |
| 有 starvation/delay bound | 不通过 | 无界 burst 推迟 B；count bound 不限制 C 生命周期 |
| 无新增 preemption/fairness 退化 | 不通过 | 长生命周期 bounded case 新增 3 次 preemption |
| 不重复 upstream 工作 | 不通过 | Day 15 审计时已有相近无界 skip draft 方向 |
| NCU 单 kernel 微架构证据 | 通过 | 真实 Prefill GEMM 的 full/stall counter 完整 |
| NCU 策略级归因 | 未通过但非必要 | launch 未与 mixed step 唯一对齐，无同-shape 对照 |

所以更准确的项目成果是一个有代码、有正反实验、有执行证据的调度负结果，而不是
待发布功能。

## 5 分钟表达

### 1. 背景，约 40 秒

我想把对 Continuous Batching 和 KV Cache 的理解转成真实 vLLM 工程能力，所以
固定在 v0.26.0，先解决 WSL 下 MRV2 的 UVA 前置条件，再构建逐 step trace。目标
不是通读仓库，而是能够从一次请求延迟一直定位到 Scheduler、KV、Model Runner
和 GPU 执行形态。

### 2. Trace，约 60 秒

我实现了默认关闭的两条 JSONL。Scheduler 侧记录 waiting/running 顺序、token
budget、scheduled tokens、KV block、allocation failure 和 preemption；MRV2 侧
复用已有 CPU metadata，记录 persistent row、index mapping 和 input shape。
实现不读 GPU tensor、不调用 `.item()`、不增加同步。随后用精确 token 和 arrival
time 的 workload 验证 8K/16K Prefill、Prefix Cache、preemption 以及 Decode/
Prefill mixed step。

### 3. 问题与修改，约 80 秒

Trace 显示在 1450-block KV pool 中，B 是 waiting 队首且完整输入需要约 1024
blocks，但当时只有 933 free；C 在 B 后面只需要约 64 blocks。默认 Scheduler 在
B allocation failure 后直接 break，所以 C 也等待约 33 秒。这是 waiting HOL
blocking。

我先实现 opt-in skip，再收紧成 one-admission bound：一个 blocked head 最多让
一个后续请求进入，之后必须优先处理或移除 head。三请求 repeated run 中，C
TTFT median 从 33.250 秒降到 0.182 秒，B first step 保持 517。

### 4. 反例与 profiling，约 90 秒

我没有停在这个漂亮数字。无界 burst 会让 B first step 从 517 推迟到 587。
Bounded 在 8 个 1K/512 长生命周期请求中只让 C1 很早进入，却新增 3 次
preemption；makespan 增加 1.21%，吞吐下降 1.20%，TTFT Jain 指数下降 10.64%。
原因是 admission count 限制不了 C1 持有 KV 的时间。

Profiling 方面，本机 Nsight Systems 没有 kernel activity，所以我改用与 Scheduler
step 60–79 精确对齐的 PyTorch Profiler。Strict 是 20 个 single Decode；bounded
插入一个 1024 Prefill 加 1 Decode 的 mixed step 和两个 dual Decode。它新增 GEMM
工作，说明策略改变 batch shape 和 kernel mix，而不是 kernel 代码。

授权开启 performance counter 后，NCU 对一个真实 Prefill GEMM 测得约 24.7%
achieved occupancy、84.0% L2 hit，主要 stall 是 61.9% math-pipe throttle，而
long scoreboard 约 1.4%，所以该 launch 不是简单的 DRAM-latency-bound。不过
它的 grid 没有与 mixed step 唯一对齐，因此我只把它作为单 kernel 微架构证据，
不归因成 bounded 策略的性能变化。

### 5. 结论，约 30 秒

最终我没有提交 PR。HOL 问题和局部收益都是真的，但 one-admission bound 不是
starvation 或完成时间保证，长生命周期反例已经出现 preemption 和公平性退化。
这个项目证明的是我能定位、修改、测量，并在数据不支持时否决自己的方案。

## 15 分钟展开顺序

1. 环境 Gate：为什么 WSL 默认 UVA 失败，为什么显式固定 MRV2。
2. `Request → SchedulerOutput → update_requests → persistent row →
   idx_mapping → GPU input`。
3. `num_computed_tokens`、in-flight 和 processed 的区别。
4. KV block table 增长、physical allocation、Prefix Cache reuse 的区别。
5. running allocation failure 如何触发 preemption/recomputation。
6. mixed A Decode/B Prefill 如何分 token budget。
7. 为什么现有 `long_prefill_token_threshold` 使新 Prefill quantum patch 重复。
8. D13 HOL witness 和 strict full-ISL admission。
9. unbounded 与 bounded 策略状态和生命周期。
10. 三请求收益、无界 starvation 反例、bounded lifetime 反例。
11. Scheduler-aligned profiler 如何证明 batch shape 变化。
12. NCU 的 occupancy、cache 和 stall 数据能说明什么，不能说明什么。
13. PR Gate 与更原则化方案需要什么。

## 20 分钟追问题

### Scheduler 与状态

1. 为什么 `num_computed_tokens=8192` 不能证明 GPU 已完成 8192 tokens？
2. 为什么 persistent row 的具体编号不能跨 run 比较？
3. Scheduler request order 与 MRV2 batch order 为什么可能不同？
4. waiting、skipped_waiting、running 和 preempted request 如何迁移？
5. Priority policy 下 bypass 的语义为什么比 FCFS 更复杂？

### KV 与策略

6. full-ISL reservation 解决什么问题，又制造了什么 HOL 代价？
7. 为什么直接关闭 full-ISL reservation 不是安全替代？
8. allocation failure、OOM 和 preemption 有什么区别？
9. one-admission bound 为什么不是 wall-clock bound？
10. 为什么只看 B first scheduled step 会漏掉后续风险？
11. Jain fairness 在这个实验里为何需要结合业务语义解释？
12. 如果 C 的 output 长度未知，怎样做 conservative backfill？

### GPU 与证据

13. Decode 为什么通常 GEMV-heavy，Prefill 为什么更容易出现 GEMM？
14. 两个 profile 为什么都有 480 次 attention call？
15. 为什么 mixed step 的 kernel duration 更长不能直接说明策略整体更慢？
16. Nsight Systems 中有 `cudaLaunchKernel` API 为什么仍不能证明 kernel timeline？
17. NCU 已打通后，为什么仍不能说整个 Prefill compute-bound？

### 工程判断

18. 为什么没有为完成项目而重复实现 Prefill quantum？
19. 哪些证据会让你重新打开 waiting bypass 设计？
20. 如果 upstream 已有相近 PR，最合适的贡献方式是什么？

## 关键答案锚点

- Scheduler logical progress 不等于 CUDA completion timestamp。
- 同一次 run 内 row 稳定和 mapping 正确才有意义，row 数字本身没有跨 run 语义。
- Full-ISL 防止长请求反复 preempt/recompute，但可能让可容纳的小请求 HOL 等待。
- B first step 不变不代表后续 KV 压力、preemption 或 completion time 不变。
- Jain 指数奖励“大家一样慢”，所以必须与 TTFT、head delay、makespan 和
  preemption 一起解释。
- Profiler 证明 observed work placement；无 profiler 重复实验决定性能。
- NCU 证明 targeted launch 的 counter 行为；没有 step 对齐和同-shape 对照，就
  不能归因给 bounded 策略，也不能外推整个 Prefill。

## 个人验收规则

不看报告完成以下三项即通过：

1. 在 5 分钟内讲清问题、trace、patch、正结果、反例和 no-PR 决策；
2. 从 20 个追问题中随机回答 12 个，其中必须包含第 1、6、9、15、17 题；
3. 能主动指出一个数字的实验边界，而不是等面试官追问。
