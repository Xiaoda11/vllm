# Day 5：两个长 Prefill 如何调度

## 问题、假设与判定标准

固定 A=8192 prompt tokens、B=16384 prompt tokens，交叉：

- 全局每 step token budget：2048、4096、8192；
- 到达顺序：同时到达、A 先到 0.5s、B 先到 0.5s。

实验前假设：

1. `max_num_batched_tokens` 是 Scheduler 每个 step 的全局 budget，不是每个
   请求各自的 chunk 大小。
2. 默认配置按 FCFS 让队首长 Prefill 消耗 budget，不会公平轮转两个长
   partial prefill。
3. 请求在首次成功分配 KV blocks 并获得 scheduled tokens 的 step 从
   `WAITING` 变为 `RUNNING`。

判定只使用 Scheduler JSONL、MRV2 JSONL 和请求 timing。Scheduler 的
`num_computed_tokens` 是逻辑进度，不能当作 CUDA kernel 完成时间戳。

## 环境与可追溯性

- vLLM：v0.26.0；
- 实验提交：`9b837fac815c553d27bbdf65a22960ddcb09b21b`；
- 九组 metadata 的 `git_status_short` 均为空；
- GPU：RTX 2060 Laptop 6 GiB / WSL2；
- `VLLM_USE_V2_MODEL_RUNNER=1`；
- `VLLM_WSL2_ENABLE_PIN_MEMORY=1`；
- attention backend：`TRITON_ATTN`；
- chunked prefill 开启，prefix caching 关闭；
- `max_num_seqs=2`，`gpu_memory_utilization=0.70`，eager mode。

正式产物位于：

```text
/home/xiaoda/vllm-lab/outputs/day5-{simultaneous,a-first,b-first}-b{2048,4096,8192}-20260729
```

每个目录包含 `run_metadata.json`、`request_timing.csv`、
`scheduler_trace.jsonl`、`scheduler_trace.mrv2.jsonl` 和转换后的
`scheduler_trace.csv`。

## GPU 实测结果

“首次 step/tokens”表示请求第一次得到 scheduled tokens；括号中的 blocks
是该 step 新分配的 KV block 数。`records` 包含末尾空调度和 finished 清理
记录。

| 到达顺序 | Budget | A 首次 step/tokens | B 首次 step/tokens | Records | 双 partial-prefill step | Preemption |
|---|---:|---:|---:|---:|---:|---:|
| 同时 | 2048 | 1 / 2048 (128) | 5 / 2047 (128) | 46 | 0 | 0 |
| 同时 | 4096 | 1 / 4096 (256) | 3 / 4095 (256) | 40 | 0 | 0 |
| 同时 | 8192 | 1 / 8192 (512) | 2 / 8191 (512) | 37 | 0 | 0 |
| A 先到 | 2048 | 1 / 2048 (128) | 5 / 2047 (128) | 46 | 0 | 0 |
| A 先到 | 4096 | 1 / 4096 (256) | 3 / 4095 (256) | 40 | 0 | 0 |
| A 先到 | 8192 | 1 / 8192 (512) | 2 / 8191 (512) | 37 | 0 | 0 |
| B 先到 | 2048 | 9 / 2047 (128) | 1 / 2048 (128) | 46 | 0 | 0 |
| B 先到 | 4096 | 5 / 4095 (256) | 1 / 4096 (256) | 40 | 0 | 0 |
| B 先到 | 8192 | 3 / 8191 (512) | 1 / 8192 (512) | 37 | 0 | 0 |

同时到达和 A 先到时，A 先消耗完整 Prefill。B 先到时，B 先消耗完整
Prefill。第二个请求首次被调度时，先到请求的 Prefill 已结束；本 step 给
先到请求 1 个 Decode token，再把剩余的 `budget - 1` 给第二个请求。

所以 trace 中确实存在同时调度两个请求的 mixed step，但不存在同时推进两个
partial prefill 的 step。两者不能混为一谈。

三组 budget 下的最小剩余 KV block 数分别为 15702、15331、14573，最高
KV usage 分别约为 8.93%、9.13%、9.56%。九组均无 preemption，因此本矩阵
观测到的是 token-budget 限制，不是 KV 容量限制。

## 六个问题的答案

### 1. 4096 是单请求 chunk，还是全局每 step budget？

是全局每 step budget。

4096 run 的关键 mixed step 总量为 4096：先到请求获得 1 个 Decode token，
第二个请求获得 4095 个 Prefill tokens。不是每个请求各有 4096。

源码中 `schedule()` 以 `self.max_num_scheduled_tokens` 初始化一个
`token_budget`，每调度一个请求就从同一个变量中扣除。

### 2. 一轮能否调度多个 partial prefill？

本次默认配置下不能。九组 trace 的双 partial-prefill step 数都为 0。

可以在同一 step 调度多个请求，但本矩阵里的 mixed step 是
Decode + Prefill。`SchedulerConfig.max_num_partial_prefills` 默认是 1，
`long_prefill_token_threshold` 默认是 0；本次没有开启 concurrent partial
prefill 配置。

### 3. 请求何时进入 RUNNING？

请求首次被 Scheduler 选中、KV allocation 成功并获得非零
`num_scheduled_tokens` 时，从 `WAITING` 变成 `RUNNING`。

例如 simultaneous + 2048 中，B 在 step 2–4 仍是 WAITING；step 5 得到
2047 tokens，同时变为 RUNNING。客户端已提交请求不等价于 Scheduler 已将其
置为 RUNNING。

### 4. KV block 何时分配？

Scheduler 先计算本 step 的 `num_new_tokens`，随后调用
`kv_cache_manager.allocate_slots()`。分配成功后才把 waiting request 移入
running、记录 scheduled tokens 并扣减 budget。

Trace 在同一个首次调度 step 中同时观察到：

- `WAITING → RUNNING`；
- 非零 scheduled tokens；
- 非空 `allocated_block_ids`。

2048、4096、8192 的首次分配分别是 128、256、512 个 block。

### 5. Budget 不足与 KV 空间不足分别导致什么？

Budget 不足是本次 GPU 实测：

- `num_new_tokens` 被截断到当前剩余 `token_budget`；
- 队首请求用完 budget 后，后续 waiting request 留在队列；
- 没有因此产生 preemption。

KV 空间不足没有在本矩阵中实测。源码行为是：

- waiting request 的 `allocate_slots()` 返回 `None` 时，本轮停止继续接纳，
  请求保持 waiting；
- running request 扩容失败时，Scheduler 会从低优先级端抢占 running
  request，直到分配成功或当前请求本身也无法继续。

因此不能把“本 step budget 已用完”报告成“KV OOM”，也不能用本次无压力
数据声称已经复现了 KV preemption。

### 6. MRV2 如何消费 SchedulerOutput？

`SchedulerOutput` 携带 new/resumed requests、每请求
`num_scheduled_tokens` 和总 scheduled tokens。

`GPUModelRunner.update_requests()`：

1. 为新请求取得一个 persistent request row；
2. 把 token IDs、长度和 computed-token 等增量写入
   `RequestState` 的 `StagedWriteTensor`；
3. 为本 step 的执行顺序生成 CPU `idx_mapping_np`；
4. 用 mapping 从 persistent rows gather 本 step 的 GPU inputs。

九组 MRV2 trace 中，每个请求在自己的生命周期内 row 都保持稳定；但 A/B
具体得到 row 0 还是 row 1 会跨 run 变化。稳定性和 mapping 是语义，具体
数字不是请求身份。

## 时延边界

各 run 都独立启动 Engine，首轮推理日志显示 Triton kernel JIT。当前
request-level TTFT 受首次 JIT 和异步执行影响明显，因此 Day 5 不用这九个
单次样本做性能优劣结论。需要比较 TTFT/吞吐时，应增加 shape warmup 和重复
次数；本报告只把 timing 当作请求成功及到达控制证据。

## 结论与未解决问题

结论：默认 v0.26 Scheduler 在该 workload 上体现 FCFS 头部效应。先到的长
Prefill 按全局 budget 分块推进，后到的长 Prefill 不会与它轮转；只有先到
请求转入 Decode 后，第二个 Prefill 才使用同一个 step 的剩余 budget。

未解决问题：显式设置 `max_num_partial_prefills > 1` 和
`long_prefill_token_threshold` 后，两个长 Prefill 是否会公平推进，以及
收益是否足以抵消更复杂 batch 的执行代价。这是第 3 周 per-request quantum
策略候选的直接 baseline。

## 30 秒面试表达

在 vLLM v0.26 的 8K/16K 双长请求实验中，我用 Scheduler 和 MRV2 JSONL
证明 `max_num_batched_tokens` 是每 step 的全局 budget。默认 FCFS 下，先到
请求会独占 partial prefill；它完成 Prefill 后，同一 step 才出现 1-token
Decode 加 `budget-1` 的第二请求 Prefill。九组到达顺序和 budget 实验都没有
出现两个 partial prefill 同步推进，也没有 KV 压力或 preemption。MRV2 中
请求 persistent row 在生命周期内稳定，由每 step 的 index mapping 组成实际
执行 batch。
