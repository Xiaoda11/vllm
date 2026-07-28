# Day 4：Scheduler Trace v1

## 工程问题

Day 3 的 request timing 只能说明请求何时提交、何时收到首 token、何时结束。
Day 4 要直接回答每个 Scheduler step 的内部决策：

- waiting/running 队列如何变化；
- 4096-token budget 分给了谁；
- 请求有多少 token 已被调度、仍在 flight、已被 Scheduler 消费结果；
- KV blocks 如何分配或释放；
- Scheduler 顺序如何映射到 MRV2 persistent rows。

## 观测边界

`num_computed_tokens` 是 vLLM Scheduler 的原生逻辑进度。它在
`_update_after_schedule()` 中于调度完成后立即增加，不等价于“CUDA 已完成
这些 token”。

因此 trace 同时记录：

- `num_computed_tokens`：Scheduler 已计入的 token；
- `num_in_flight_tokens`：已经调度、尚未被 Scheduler output update 消费的
  token；
- `num_processed_tokens = num_computed_tokens - num_in_flight_tokens`：
  Scheduler 已消费执行结果的 token。

`num_processed_tokens` 仍然不是 GPU kernel 完成时间戳。Scheduler output 已
消费可以证明执行链路已经返回该 step，不能用于恢复 CUDA kernel 的精确
起止时间。

## 实现

### 两条 JSONL 流

`scheduler_trace.jsonl` 每个 Scheduler step 记录：

- step ID、时间戳、初始/已用/剩余 token budget；
- running、waiting、skipped-waiting 的前后快照；
- 每个请求的状态、prompt/output/computed/in-flight/processed tokens；
- prefix-cache hit tokens；
- KV usage、free blocks、block IDs 及 allocated/freed 差集；
- scheduled、preempted、finished request IDs。

`scheduler_trace.mrv2.jsonl` 在 MRV2 已经构造好的 CPU
`idx_mapping_np` 处记录：

- Scheduler step ID；
- 实际 batch request 顺序；
- 每个请求的 persistent row；
- 每个请求的 scheduled tokens。

MRV2 warmup 记录使用 `step_id=0`。转换器只把正数 Scheduler step 与
MRV2 行连接。

### 不破坏异步路径

- 未设置 `LAB_V026_SCHEDULER_TRACE_PATH` 时 writer 不创建，默认路径没有
  trace 工作。
- Scheduler 只复制 Python/CPU 状态。
- MRV2 直接复用 `idx_mapping_np`，不读取 GPU tensor。
- JSON 序列化与落盘由后台线程完成。
- 没有新增 `.item()`、device-to-host copy 或 CUDA synchronize。

### CSV

`scripts/lab_scheduler_trace_to_csv.py` 把两条 JSONL 流按 `step_id`
连接，输出“一行一个 step/request”的 CSV，方便排序和筛选。

## GPU 实测

最终 schema 集成 Run：`day4-s3-trace-v1-20260728`

- vLLM v0.26.0；
- MRV2：`VLLM_USE_V2_MODEL_RUNNER=1`；
- WSL pinned memory opt-in；
- S3 canonical：A=8192，B=16384，均在 `t≈0` 到达；
- `max_num_batched_tokens=4096`；
- 40 个 Scheduler records，其中 step 1–38 有 MRV2 batch，step 39–40
  是空调度/finished 清理；
- MRV2 文件另有 2 条 `step_id=0` warmup records；
- 无 preemption。

前 7 步的关键字段如下。`processed before` 由
`computed before - in-flight before` 得到。

| Step | Req | Processed before | In-flight before | Scheduled | MRV2 row | MRV2 order |
|---:|---|---:|---:|---:|---:|---|
| 1 | A | 0 | 0 | 4096 | 0 | A |
| 2 | A | 0 | 4096 | 4096 | 0 | A |
| 2 | B | 0 | 0 | 0 | — | A |
| 3 | A | 4096 | 4096 | 1 | 0 | A, B |
| 3 | B | 0 | 0 | 4095 | 1 | A, B |
| 4 | A | 8192 | 1 | 1 | 0 | A, B |
| 4 | B | 0 | 4095 | 4095 | 1 | A, B |
| 5 | A | 8193 | 1 | 1 | 0 | A, B |
| 5 | B | 4095 | 4095 | 4095 | 1 | A, B |
| 6 | A | 8194 | 1 | 1 | 0 | A, B |
| 6 | B | 8190 | 4095 | 4095 | 1 | A, B |
| 7 | A | 8195 | 1 | 1 | 0 | A, B |
| 7 | B | 12285 | 4095 | 4 | 1 | A, B |

## 结果解释

Step 1–2 的 4096-token budget 全部分给 A。Step 2 开始时 A 的第一块
4096 tokens 仍是 in-flight，但 Scheduler 已经继续给 A 调度第二块 4096。

Step 3 是关键证据：

- A 在 Scheduler 侧只消费了前 4096 tokens 的执行结果，另有 4096
  tokens in-flight；
- 本步仍给 A 1 token，同时给 B 4095 tokens；
- MRV2 的实际 batch 顺序是 `[A, B]`，persistent rows 稳定为 A=0、B=1。

所以正确结论不是“客户端提交了 B，因此 vLLM 已经调度 B”，也不是“必须等
A 完全结束才准备 B”。证据表明：A 尚有一块 4096-token 工作处于 in-flight
状态时，Scheduler 已让 B 进入同一个 mixed batch，MRV2 也已为 B 使用
persistent row 1 准备执行。

这仍不能证明 step 3 发生瞬间某个 CUDA kernel 的物理完成百分比；若需要
kernel 时间线，后续应使用 CUDA event 或 Nsight，而不是把 Scheduler
逻辑计数当成 GPU 时间戳。

## 验证

- `pytest --confcutdir=tests/lab tests/lab/test_v026_workload.py
  tests/lab/test_scheduler_trace.py -q`：13 passed；
- Ruff lint：通过；
- Ruff format check：通过；
- `git diff --check`：通过；
- S1 2K trace smoke：通过；
- S3 canonical 8K/16K trace：通过。

## 个人验收

不看本文回答：

1. 为什么 `num_computed_tokens=8192` 不能直接说 GPU 已完成 8192 tokens？
2. Step 3 中哪三个字段共同证明 B 已经进入实际 MRV2 batch？
3. A 的 persistent row 0、B 的 row 1 证明什么，不证明什么？
4. 为什么客户端已经接收 B 不能证明 vLLM Scheduler 已调度 B？
5. 若要证明 CUDA kernel 的物理重叠，还缺少哪类证据？
