# Day 8：Scheduler Output → Model Runner

## 问题、假设与判定标准

本日回答三个问题：

1. `SchedulerOutput` 怎样更新 MRV2 的 request-lifetime state？
2. persistent row 与本轮执行顺序为什么是两套索引？
3. scheduled tokens 怎样变成 input IDs、positions、block tables、
   slot mappings 和 attention metadata？

实验前假设：

- request 在一次 active lifetime 内持有固定 persistent row；
- 每一步按本轮工作量重新排序，再用 `idx_mapping` gather persistent rows；
- `sum(num_scheduled_tokens)`、`query_start_loc[-1]` 和实际 input token 数相等；
- 只记录 CPU 已有值与 tensor metadata，能够验证 shape 链而不引入
  GPU-to-CPU 同步。

判定标准：

- Scheduler trace 与 MRV2 trace 能按 `step_id` 对齐；
- A/B 的 persistent row 在各自 active lifetime 内不变；
- 每个实际 forward step 都满足 token-count 和 shape 不变量；
- trace 实现不调用 `.item()`、`.cpu()` 或显式 synchronize；
- 源码事实、GPU 实测和推断分开陈述。

## 环境与正式产物

- vLLM：v0.26.0；
- trace commit：`1893048f1a782465bb159c35b349822f983891f6`；
- 正式运行的 `git_status_short` 为空；
- GPU：RTX 2060 Laptop 6 GiB / WSL2；
- `VLLM_USE_V2_MODEL_RUNNER=1`；
- `VLLM_WSL2_ENABLE_PIN_MEMORY=1`；
- attention backend：`TRITON_ATTN`；
- CUDA Graph：关闭，`cudagraph_mode=NONE`；
- token budget：2048；
- block size：16 tokens；
- Prefix Cache：关闭。

正式产物：

```text
/home/xiaoda/vllm-lab/outputs/day8-s3-runner-flow-20260731
```

目录包含 `run_metadata.json`、`request_timing.csv`、
`scheduler_trace.jsonl`、`scheduler_trace.mrv2.jsonl` 和
`scheduler_trace.csv`。

受控 workload 是 canonical S3：

| Request | Prompt | Output | Arrival |
|---|---:|---:|---:|
| A | 8192 | 32 | 0 s |
| B | 16384 | 32 | 0 s |

正式命令：

```bash
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export VLLM_USE_V2_MODEL_RUNNER=1

/home/xiaoda/vllm-lab/.venv-v026/bin/python \
  scripts/lab_v026_workload.py \
  --config benchmarks/scheduler_trace/configs/s3_dual_simultaneous.json \
  --model /home/xiaoda/vllm-lab/models/Qwen2.5-0.5B-Instruct \
  --token-budget 2048 \
  --run-id day8-s3-runner-flow-20260731 \
  --scheduler-trace
```

## Opt-in input metadata trace

原有 `model_runner_batch` 事件记录 per-step request order、persistent rows 和
scheduled tokens。本日增加独立的 `model_runner_inputs` 事件，记录：

- `R`、`R_padded`、`T`、`T_padded` 和 CUDA Graph mode；
- CPU 已有的 `query_start_loc`、computed/prefill counts 和 phase；
- persistent tensors 的 shape、dtype 和 device；
- gathered input tensors、block tables 和 slot mappings 的 metadata；
- 每层 attention metadata 的 Python 类型；
- 最终 model input keys。

`tensor_metadata()` 只读取 `shape`、`dtype` 和 `device` 属性。事件中的
`.tolist()` 只作用于 `idx_mapping_np`、`num_scheduled_tokens` 和其他已有
NumPy 数组，没有读取 GPU tensor 内容。因此 trace 没有新增 `.item()`、
`.cpu()`、GPU-to-CPU copy 或显式同步。

Tracing 未开启时 writer 为 `None`，`_record_input_metadata()` 立即返回。
默认执行路径不构造 JSON event。

## 源码数据链

### 1. SchedulerOutput 的字段边界

`Scheduler.schedule()` 产生：

- `scheduled_new_reqs`：首次调度或 preemption 后重新加入的完整 request
  data；
- `scheduled_cached_reqs`：已缓存 request 的 computed-token 和 block-table
  增量；
- `num_scheduled_tokens`：本 step 每个 request 要执行的 token 数；
- `finished_req_ids` 和 `preempted_req_ids`：需要移除的 active state。

`num_scheduled_tokens` 表示本轮工作量，不等于进入本轮前已经完成的
`num_computed_tokens`。

### 2. execute_model 更新 active state

非 dummy 路径按以下顺序执行：

```text
update_pp_decode_requests
→ finish_requests
→ free_states
→ add_requests
→ update_requests
→ block_tables.apply_staged_writes
→ prepare_inputs
→ prepare_attn
→ model_state.prepare_attn
→ model forward
```

`finish_requests()` 将 finished 和 preempted request 都从 MRV2 active state
移除。Preempted request 恢复时由 Scheduler 重新放入
`scheduled_new_reqs`，因此是 fresh add，而不是复用旧 state。

`add_requests()`：

1. 从 `RequestState.free_indices` 取得 row；
2. 建立 request ID ↔ row 映射；
3. stage token IDs、lengths 和 computed-token state；
4. 使用同一 row 初始化 model、block-table、LoRA 和 sampler state；
5. apply staged writes。

`update_requests()` 处理 cached request。它更新
`num_computed_tokens_np` 这个 optimistic CPU mirror，并 stage 新 block IDs。
GPU 上的 sampled/computed state还会由 MRV2 postprocess 路径维护，不能把
`update_requests()` 简化为“同步写完全部 GPU state”。

### 3. fixed row 与 per-step order

`RequestState` 为 active request 保留固定 row。`prepare_inputs()` 不直接按
row 顺序执行，而是：

```text
num_scheduled_tokens
→ sort_batch_req_ids(decode → short extend → prefill)
→ req_id_to_index lookup
→ idx_mapping
→ GPU input preparation / block-table gather
```

因此：

- persistent row：request-lifetime state 的地址；
- request execution order：本 step 的紧凑 batch 顺序；
- `idx_mapping`：`batch row → persistent row` 的间接映射。

### 4. input 与 attention metadata

`prepare_inputs()` 根据 `idx_mapping` 和 `query_start_loc`：

- 从 persistent token state 准备 `input_ids`；
- 从 computed-token state 准备 `positions` 和 `seq_lens`；
- 生成 per-step `InputBatch`。

`prepare_attn()` 使用同一个 `idx_mapping`：

- gather fixed-row block tables 到紧凑 per-step block tables；
- 根据 positions、block tables 和 block size 计算 slot mappings。

随后 `model_state.prepare_attn()` 生成 backend-specific metadata，并由
`set_forward_context()` 传给 attention 层。正式运行的 24 个 attention
layer entries 均为 `TritonAttentionMetadata`。

## GPU trace 结果

Scheduler 共产生 46 个 step。Step 1–44 有实际 model input；step 45 是零
token async step，step 46 只通知 B finished，因此这两个 step 不执行
forward。MRV2 JSONL 另含两个 `step_id=0` warmup records，正式分析已排除。

44 个实际 forward step 全部满足：

```text
sum(num_scheduled_tokens)
= batch.num_tokens
= query_start_loc[-1]

input_ids.shape
= positions.shape
= [num_tokens_after_padding]

seq_lens.shape
= [num_reqs_after_padding]
```

本实验使用 eager mode，所以所有正式 step 都有：

```text
R_padded = R
T_padded = T
```

这不是对 CUDA Graph 模式无 padding 的推广结论。

### Persistent row 实测

| Request | Active steps | Persistent row |
|---|---|---:|
| A | 1–35 | 1 |
| B | 5–44 | 0 |

A/B 在各自 active lifetime 内 row 不变。row 顺序 `[1, 0]` 与 request
execution order `[A, B]` 明确不同。

### 代表 step 5

Step 5 是最清楚的 Scheduler → MRV2 混合 batch：

| 字段 | A | B |
|---|---:|---:|
| 进入本 step 前 computed tokens | 8192 | 0 |
| 本 step scheduled tokens | 1 | 2047 |
| Persistent row | 1 | 0 |
| Phase | Decode | Partial Prefill |

逐步闭合：

```text
request order = [A, B]
idx_mapping = [1, 0]
num_scheduled_tokens = [1, 2047]
query_start_loc = [0, 1, 2048]
T = 1 + 2047 = 2048
```

本 step 的实际 tensor metadata：

| Tensor | Shape | Dtype | Device |
|---|---|---|---|
| `idx_mapping` | `[2]` | `torch.int32` | `cuda:0` |
| `query_start_loc` | `[3]` | `torch.int32` | `cuda:0` |
| `input_ids` | `[2048]` | `torch.int32` | `cuda:0` |
| `positions` | `[2048]` | `torch.int64` | `cuda:0` |
| `seq_lens` | `[2]` | `torch.int32` | `cuda:0` |
| gathered block table group 0 | `[2, 2048]` | `torch.int32` | `cuda:0` |
| slot mappings | `[1, 2048]` | `torch.int64` | `cuda:0` |

Scheduler trace 在 step 5 调度完成后的 optimistic state 是 A=8193、
B=2047；MRV2 input metadata记录的进入本轮前 computed state 是 A=8192、
B=0。两组数字描述不同时间边界，不是 trace 冲突。

### Persistent tensor shape

| State | Shape | 实现/含义 |
|---|---|---|
| `all_token_ids` | `[2, 32768]` | UVA-backed `StagedWriteTensor` |
| `prompt_len` | `[2]` | `UvaBackedTensor` |
| `prefill_len` | `[2]` | `UvaBackedTensor` |
| `total_len` | `[2]` | GPU `StagedWriteTensor` |
| `num_computed_tokens` | `[2]` | GPU staged state + NumPy mirror |
| persistent block table group 0 | `[2, 2048]` | fixed-row block-table state |

Trace 中 UVA-mapped tensors 的 device 字符串也是 `cuda:0`。这只表示 GPU
访问视图，不能据此声称其物理 storage 位于普通 GPU 显存；UVA backing 是
源码事实。

### Batch shape 分布

| `(R, T, T_padded)` | Steps |
|---|---:|
| `(2, 2, 2)` | 22 |
| `(1, 1, 1)` | 9 |
| `(2, 2048, 2048)` | 8 |
| `(1, 2048, 2048)` | 4 |
| `(2, 9, 9)` | 1 |

这张表描述本次 S3 trace，不是性能 benchmark。

## V1 / MRV2 字段映射

下表的 MRV2 列来自本次 GPU 实测与 v0.26 源码；V1 列只来自当前 v0.26
MRV1 源码阅读，没有做本日 V1 GPU 实测。

| 关注点 | V1 / MRV1 | MRV2 |
|---|---|---|
| Python request state | `dict[req_id, CachedRequestState]` | request ID ↔ fixed row maps |
| Persistent batch | `gpu_input_batch.InputBatch` 同时保存 state 和 input layout | `RequestState`/`BlockTables` 与 per-step `InputBatch` 分离 |
| Row 维护 | remove、add、condense、必要时 reorder | active lifetime 内固定 |
| Token IDs | persistent CPU tensors/lists，再准备 GPU input | `all_token_ids` UVA-backed state |
| Computed tokens | `num_computed_tokens_cpu` 等 InputBatch fields | GPU staged state + optimistic NumPy mirror |
| KV block table | `InputBatch.block_table` | fixed-row `BlockTables` + per-step gather |
| 本 step 顺序 | persistent InputBatch rows 需要整理成执行布局 | `sort_batch_req_ids` + `idx_mapping` |
| Compact input IDs | 从当前 InputBatch rows 准备 | GPU kernel 按 `idx_mapping` 从 state 准备 |
| Preemption resume | cached request state 支持重新插入 | remove old state，resume 时 fresh add |
| 异步安全 | input-prep barrier 和 previous-row bookkeeping | staged writes、UVA pools、stream ordering |

V1/MRV2 的字段不是逐个同名替换。核心差异是 MRV2 将 request-lifetime state
与 per-step execution layout 解耦。

## 静态验证

```text
python -m pytest tests/lab/test_scheduler_trace.py \
  tests/lab/test_v026_workload.py -q
→ 21 passed

python -m ruff check \
  vllm/v1/core/sched/trace.py \
  vllm/v1/worker/gpu/model_runner.py \
  tests/lab/test_scheduler_trace.py
→ All checks passed

git diff --check
→ passed
```

测试覆盖 trace 默认关闭、JSONL writer、Scheduler/MRV2 join，以及
`tensor_metadata()` 只依赖 shape/dtype/device 的接口。

## 结论

MRV2 没有把 Scheduler 的 request list 直接当成 persistent tensor rows。
SchedulerOutput 先增量更新 request-lifetime state；`prepare_inputs()` 再按
本 step 工作量建立 execution order，并用 `idx_mapping` gather fixed rows。
同一映射继续驱动 token/position preparation 和 block-table gather，最终
构造 backend-specific attention metadata。

S3 step 5 实测把 A decode 的 1 token 与 B partial prefill 的 2047 tokens
组成 `[A, B]` batch，同时 persistent rows 是 `[1, 0]`。输入 tensors 的
首维由 `R=2` 或 `T=2048` 决定，所有 token-count 不变量闭合，24 层均消费
`TritonAttentionMetadata`。

未解决问题：CUDA Graph 开启后 `R_padded/T_padded` 的 shape bucket 如何
改变 input buffers 和 metadata，以及不同 KV-cache group 的 backend/layout
如何改变 block-table tuple，将在 Day 9 处理。

## 30 秒技术摘要

我在 vLLM v0.26 MRV2 增加了 opt-in input-shape trace，只读取 CPU 已有数组
和 tensor metadata，不做 GPU 回读。8K/16K S3 的 step 5 中，Scheduler 给
A decode 1 token、给 B prefill 2047 tokens；执行顺序是 A/B，但它们的
persistent rows 是 1/0，所以 `idx_mapping=[1,0]`。随后得到
`query_start_loc=[0,1,2048]`、2048 个 input IDs/positions、两行 gathered
block table 和 `[1,2048]` slot mapping，24 层 attention metadata 都是
Triton 类型。这证明 MRV2 把 request-lifetime state 与 per-step batch layout
解耦。
