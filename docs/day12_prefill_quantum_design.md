# Day 12：Prefill Quantum 配置策略与代码 Gate

## 决策摘要

Day 12 不实现新的 Scheduler 分支。

Day 11 已证明，在当前 RTX 2060、Qwen2.5-0.5B、MRV2、`TRITON_ATTN` 和
S5 B=8K workload 下，v0.26 现有 `long_prefill_token_threshold=2048` 能把
Decode 请求在 Prefill 阶段的 ITL P95 中位数降低 52.5%，同时 B TTFT/E2E
中位数各增加约 0.7%，output token/s 中位数变化为 -0.1%。单请求 8K Prefill
四次对照没有观察到明确退化。

因此当前策略是：

> 把 v0.26 现有 `long_prefill_token_threshold` 作为可开关的 Prefill quantum
> baseline，先完成边界实验。只有全局阈值在非混合 workload 中产生稳定、
> 明确且无法用现有配置规避的退化，才设计 mixed-only 新逻辑。

这保留了原计划“默认行为不变、一次只修改一个策略变量”的目标，也避免为了
完成 patch 而复制 upstream 已有功能。

## 版本与执行路径

- 固定源码：v0.26.0 tag，commit
  `f2654939e69b4069b13977e9aef3e31d4dcaf051`。
- 当前实验分支：`exp/mrv2-scheduler-trace`。
- Day 11 阈值入口：`3e301bd939`。
- Day 11 报告：`e02bb751dc`。
- 环境：`/home/xiaoda/vllm-lab/.venv-v026`。
- 执行路径：Path A，显式设置 `VLLM_USE_V2_MODEL_RUNNER=1` 和
  `VLLM_WSL2_ENABLE_PIN_MEMORY=1`。
- 模型/backend：Qwen2.5-0.5B-Instruct FP16 / `TRITON_ATTN`。
- Day 11 原始证据：
  `/home/xiaoda/vllm-lab/outputs/day11-threshold-gate-*-20260803`。

## 当前行为

### 配置语义

`SchedulerConfig.long_prefill_token_threshold` 默认是 0，表示不启用 Prefill
cap。它已经通过 `AsyncEngineArgs` 和命令行暴露。

当 `max_num_partial_prefills > 1` 且阈值仍为 0 时，配置初始化会自动把阈值设为
`max_model_len × 0.04`。本项目没有开启该自动路径，而是显式设置 0 或 2048，
保证实验只改变一个变量。

### Scheduler 控制流

每个 step 先用 `max_num_scheduled_tokens` 初始化全局 `token_budget`，再按顺序：

1. 调度 `running` 请求；
2. 从同一个 budget 中扣除这些请求的 scheduled tokens；
3. 用剩余 budget 接纳和调度 `waiting` 请求。

对 running 和 waiting 两条路径，代码都会执行等价的截断：

```text
if 0 < threshold < num_new_tokens:
    num_new_tokens = threshold
num_new_tokens = min(num_new_tokens, token_budget)
```

因此阈值是全局 Prefill cap，不是 mixed-step 专用 cap。普通 Decode 的
`num_new_tokens=1`，不会被 2048 阈值截断。

### Day 11 实测行为

在 S5 B=8K、budget=8192 中：

- threshold 0：每次 run 都出现 `A=1, B=8191`，随后一个尾块 step；
- threshold 2048：每次 run 都出现四个 `A=1, B=2048` mixed steps；
- 最大 mixed batch 从 8192 tokens 降到 2049 tokens；
- 三次重复均无 preemption、allocation failure 或 MRV2 shape mismatch。

在 S1 单请求 8K Prefill 中：

- threshold 0：`8192×1`；
- threshold 2048：`2048×4`；
- forward step 数从 32 增至 35，但四次重复没有观察到明确的请求延迟退化。

## 工程问题与根因

工程问题不是“vLLM 没有 Prefill quantum”。这个能力已经存在。

尚未回答的问题是：

> 全局 Prefill cap 在混合 workload 中降低 Decode ITL 尾部时，是否会在更长
> 的纯 Prefill、多长 Prefill 并发、burst 或 KV 压力 workload 中造成稳定且
> 有工程意义的退化？

源码根因边界：阈值只根据单请求本轮 `num_new_tokens` 截断，不区分当前 step
是否包含 Decode，也不根据队列年龄、SLO、GPU kernel 时间或 KV 压力动态调整。
这是一个静态配置的能力边界，不等价于已经证明的缺陷。

## 选择的策略

### 当前策略：显式静态 quantum

保留两个配置档：

- Baseline：`long_prefill_token_threshold=0`；
- Tuned：`long_prefill_token_threshold=2048`。

Tuned 只用于本项目的 latency-sensitive mixed workload 实验。目前不把 2048
写成通用推荐默认值，也不修改 vLLM 默认配置。

### 暂不采用：mixed-only quantum

候选 mixed-only 语义是：只有本 step 已经调度至少一个 Decode 请求时，才对
后续长 Prefill 使用 quantum；纯 Prefill step 保持原始 global budget。

该候选暂不实现，原因是：

- 现有阈值已经达到最小目标；
- 单请求 8K Prefill 尚未暴露全局阈值的明确代价；
- 新分支会引入额外语义、配置和测试面；
- 当前没有证据证明这些复杂度能换来可测收益。

## 新增状态与修改函数

当前配置策略：

- 新增 Scheduler 状态：无；
- 修改 Scheduler 函数：无；
- 默认行为：不变；
- 热路径额外分支：无；
- 新增持久化或 per-request metadata：无。

Day 11 只在实验工具中增加了
`--long-prefill-token-threshold`，把现有 `AsyncEngineArgs` 配置写入有效 scenario
和 run metadata。

如果后续代码 Gate 被触发，Day 12 不预先承诺字段名或修改位置。新的设计必须
重新说明 mixed step 的判定边界，尤其是 running partial prefill、多个 Decode、
waiting admission、preemption/resume、async scheduling 和 speculative decode。

## 时间与空间复杂度

当前配置策略复用既有比较与 `min`：

- Scheduler 渐进时间复杂度：不变；
- Scheduler 空间复杂度：不变；
- per-request 状态：不变；
- GPU metadata 大小：不变；
- 可能变化的是 forward step 数和每 step batch shape，而不是算法阶数。

在已测 S5 中，mixed Prefill step 从 2 增至 4；在 S1 中，总 forward step 从
32 增至 35。这种常数项变化必须通过 TTFT、ITL、吞吐和后续 Nsight 衡量。

## 预期收益

已测收益，不是预测：

- S5 B=8K 中 A 的 Prefill 阶段 ITL mean 中位数下降 32.7%；
- A 的 Prefill 阶段 ITL P95 中位数下降 52.5%；
- B TTFT/E2E 中位数各增加约 0.7%；
- output token/s 中位数变化为 -0.1%。

待验证收益：

- B=16K 时是否仍能降低 Decode ITL P95；
- 多个 Decode 请求存在时是否仍保持相同方向；
- burst workload 中是否改善 request-level ITL P95/P99。

## 可能退化的场景

必须主动寻找而不是假定不存在：

- 单个 16K Prefill：更多 step/kernel launch 可能提高 TTFT；
- 多个长 Prefill 同时到达：更小 quantum 可能改变 FCFS、batch shape 和输入
  throughput；
- 高并发 burst：更多 active partial prefills 可能增加 KV 占用或排队开销；
- KV 压力：更早接纳多个请求可能改变 allocation failure/preemption 行为；
- Prefix Cache：命中后的 remaining tokens 可能落在阈值两侧，收益不同；
- 其他模型/backend/GPU：2048 不是可直接迁移的最佳值；
- CUDA Graph 开启时：更多 shape 可能改变 graph 命中，本轮 eager 结果不能
  外推。

## 最小边界实验计划

为控制 GPU 时间，按 Gate 顺序执行，不一次展开 Day 15 全矩阵。

### Gate A：更长 mixed workload

- S5：A=1K/512，B=16K/32，arrival=1.0s；
- budget=8192；threshold=0/2048；
- 每组至少 3 次独立启动；
- 目标：确认 Decode ITL P95 收益能否从 B=8K 延伸到 B=16K。

### Gate B：更长纯 Prefill

- S2：16K prompt / 32 output；
- budget=8192；threshold=0/2048；
- 每组至少 3 次，并使用交错运行顺序；
- 目标：检查更多 Prefill step 是否造成稳定 TTFT/E2E/input throughput 退化。

### Gate C：双长 Prefill

- S3：8K 和 16K 同时到达；
- budget=8192；threshold=0/2048；
- 每组至少 3 次；
- 目标：观察两请求首次调度、TTFT 差距、step-level token share、Jain fairness、
  KV usage 和总 input throughput。

### Gate D：burst/KV 压力

只在 A--C 暴露有意义边界后执行：

- concurrency=4/8；
- threshold=0/2048；
- 记录 allocation failure、preemption、recompute tokens 和 peak KV usage；
- 不同时加入 aging、priority、deadline 或动态 budget。

## 观测量

- request：TTFT、TPOT、E2E；
- token：Decode 分阶段 ITL mean/P50/P95/max；
- workload：makespan、output token/s、input token/s；
- fairness：首次调度 step、等待时间、Jain fairness；
- Scheduler：scheduled tokens、partial-prefill step 数、队列顺序；
- KV：usage、allocation failure、preemption、recompute tokens；
- MRV2：`input_ids.shape`、`query_start_loc[-1]` 与 scheduled tokens 一致；
- 环境：commit、完整命令、runner/backend、温度和必要启动日志。

## 代码 Gate

以下是本机实验的预注册判定标准，不是通用 vLLM SLO：

1. mixed workload 中，threshold=2048 相对 baseline 的 Decode ITL P95 中位数
   至少降低 30%，并在 B=8K/16K 上方向一致；
2. 非混合 workload 中，TTFT、E2E 或 input throughput 出现超过 5% 的稳定
   退化，或新增 preemption/allocation failure；
3. 退化至少在 3 次重复中方向一致，且不能由运行顺序、首次 JIT、温度或正常
   波动解释；
4. 调整现有 `long_prefill_token_threshold`、`max_num_partial_prefills` 和
   `max_long_partial_prefills` 不能同时保留 mixed 收益并规避退化。

只有四项同时满足，才重新设计 mixed-only feature flag。否则：

- 不修改 Scheduler；
- 把 Baseline vs Tuned 作为配置策略实验；
- 进入 Benchmark/Nsight，解释 batch shape 与尾延迟取舍；
- 如需完成“新策略 patch”里程碑，必须返回 Day 11，从新数据选择另一个真实
  问题，而不是把既有配置包装成新功能。

## 测试计划

配置策略阶段：

- workload config/override 单测；
- 默认 threshold=0 的 metadata 与 trace；
- threshold=2048 的 metadata 与 trace；
- 准确输出、arrival 控制、无 shape mismatch；
- analyzer 对重复 run 的中位数和分阶段 ITL 检查。

如果代码 Gate 被触发，再设计 Scheduler 单元测试，至少覆盖：

- feature flag 默认关闭时逐 step 行为不变；
- 单请求纯 Prefill 不受 mixed-only cap 影响；
- Decode + Prefill 才触发 cap；
- 多 Decode + 多 Prefill；
- resumed/preempted request；
- budget、block-size 和 max-model-len 边界；
- KV allocation failure；
- async scheduling 下不增加 GPU-to-CPU 同步。

## Day 13 输入

Day 13 不直接实现 feature flag。先执行 Gate A 和 Gate B，并为重复实验增加紧凑
聚合分析。Gate A/B 结果决定：

- 继续 Gate C/D；
- 进入 mixed-only 代码设计；
- 或确认现有配置足够，转向 Benchmark/Nsight。

## 未解决问题

1. 2048 quantum 的效果是否可迁移到 B=16K？
2. 纯 16K Prefill 是否会暴露 8K 中看不到的 step 开销？
3. 多长 Prefill 并发时，阈值改善公平性还是损害总 input throughput？
4. eager mode 下的结论在 CUDA Graph 开启后是否仍成立？
5. Nsight 中，ITL 尾部下降来自哪些 kernel/batch-shape 变化？

## 30 秒技术摘要

我没有直接新增一个 Prefill quantum，因为 vLLM v0.26 已经有
`long_prefill_token_threshold`。三次 MRV2 对照显示，2048 阈值把 8192-token
mixed batch 拆成四个约 2K 的 batch，Decode 的 Prefill 阶段 ITL P95 中位数
下降 52.5%，而 Prefill TTFT 和吞吐代价都不到 1%；四次纯 8K Prefill 也没有
明确退化。因此我的 Day 12 设计是一个 fail-closed code Gate：先测 16K、双长
Prefill 和 burst，只有全局阈值稳定伤害非混合 workload 且现有配置无法规避，
才增加 mixed-only feature flag。
