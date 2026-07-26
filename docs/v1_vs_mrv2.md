# vLLM v0.15 V1 → v0.26 MRV2 差异地图

日期：2026-07-26

状态：Day 2 源码地图；尚未添加 Scheduler instrumentation 或修改调度策略

## 范围与证据规则

本文对比：

- 历史 V1 源码：
  `f176443446f659dbab5315e056e605d8984fd976`；
- 当前 MRV2 源码：
  `f2654939e69b4069b13977e9aef3e31d4dcaf051`；
- Day 1 实测的 v0.26 执行路径：
  MRV2 + `TRITON_ATTN`。

本文只关注 Scheduler Trace 所需的链路：

```text
SchedulerOutput
→ EngineCore
→ Executor / GPUWorker
→ GPUModelRunner request-state update
→ per-step index mapping
→ GPU input metadata
```

这不是一份完整的 V1/MRV2 功能对比。

### 假设

MRV2 将 request 生命周期内的 persistent state 与每一步的 batch 顺序分离，
从而减少异步调度所需的状态维护。每个 request 在存活期间持有固定 row，
每一步通过一层间接映射，按本次执行顺序 gather 对应的 rows。

### 观测项

- request state 的所有权与生命周期；
- new/cached/finished/preempted request 的处理方式；
- row 的分配与复用；
- CPU 到 GPU 的状态更新机制；
- 每一步的 request 顺序与 `idx_mapping`；
- CPU 准备 step N+1 时，哪些 buffer 可能仍被 GPU step N 使用。

### 判定标准

当这张地图能够回答以下六个 Day 2 问题，并且不把未实测的运行状态写成
实测结论时，即视为完成：

1. `SchedulerOutput` 如何到达 `GPUModelRunner.update_requests`？
2. 一个 request 如何获得 persistent row？
3. `RequestState` 保存哪些长期状态？
4. `StagedWriteTensor` 为什么需要 UVA？
5. `idx_mapping` 如何组织执行顺序？
6. CPU 如何在不破坏 GPU step N 的情况下准备 step N+1？

## 保持稳定的外层控制流

两个版本的 Engine Core 高层循环仍然相似：

```text
Scheduler.schedule()
→ SchedulerOutput
→ model_executor.execute_model(..., non_block=True)
→ GPUWorker.execute_model()
→ GPUModelRunner.execute_model()
→ Scheduler.update_from_output()
```

v0.26 的直接单步路径为：

```text
EngineCore.step
  scheduler.schedule
  model_executor.execute_model(non_block=True)
    Executor RPC
      GPUWorker.execute_model
        GPUModelRunner.execute_model
  future.result / sample_tokens
  scheduler.update_from_output
```

`EngineCore.step_with_batch_queue` 在此基础上维护 in-flight batches 队列。
它会尝试先调度下一批，再阻塞等待最早提交的结果。因此，V1 → MRV2 的主要
变化发生在 worker/model runner 的状态准备内部，而不是替换整套 Scheduler
API。

## SchedulerOutput → MRV2 request 更新

`Scheduler.schedule()` 构造的 `SchedulerOutput` 包含：

| 字段 | 对 model runner 的含义 |
|---|---|
| `scheduled_new_reqs` | 首次加入或重新加入的 request 完整数据 |
| `scheduled_cached_reqs` | 已缓存 request 的增量数据 |
| `num_scheduled_tokens` | 本 step 每个 request 的 token 数 |
| `total_num_scheduled_tokens` | 本 step 全局调度的 token 总数 |
| `finished_req_ids` | 可以释放的 request state |
| `preempted_req_ids` | 在 MRV2 中按 finished 处理的 state |
| cached `new_block_ids` | KV block table 的增量扩展 |

v0.26 的 `GPUModelRunner.execute_model` 按以下顺序调用：

```text
update_pp_decode_requests()
finish_requests()
free_states()
add_requests()
update_requests()
block_tables.apply_staged_writes()
prepare_inputs()
prepare_attn()
model forward
```

需要区分：

- `add_requests()` 处理 new/re-added requests 的完整状态，并应用 request、
  model 和 sampler 的 staged writes；
- `update_requests()` 处理 cached-request diffs，重点包括
  `num_computed_tokens`、新增 KV block IDs、fresh block 清零和 KV
  copy-on-write 操作；
- Scheduler 不会直接调用 `update_requests()`。`SchedulerOutput` 会先经过
  Engine Core、Executor 和 GPUWorker，最后到达 model runner。

## Persistent state：V1 与 MRV2 对比

| 关注点 | v0.15 V1 | v0.26 MRV2 |
|---|---|---|
| Python request 备份 | `dict[req_id, CachedRequestState]` | 没有等价的通用备份对象 |
| Persistent batch | `InputBatch` 同时混合状态和直接模型输入 | Persistent state 与每 step 的 `InputBatch` 分离 |
| Row 生命周期 | Row 可能被删除、填洞和压紧 | active lifetime 内固定 row |
| 未被调度的 request | 从 V1 persistent batch 移除，但保留在 cached request dict | 除非 finished/preempted，否则 state 留在原 slot |
| Preemption | cached state 支持后续恢复并重新插入 | 按 finish 处理；恢复时重新 add |
| 每 step 顺序 | 重排或压紧 persistent `InputBatch` | 用 `idx_mapping` 按 step 顺序 gather 固定 rows |
| Token state | 以 CPU tensor/list 为主，之后 copy/prepare | GPU/UVA persistent tensors，必要处保留 CPU mirror |
| KV block table | 属于 V1 `InputBatch` 布局 | 独立的 persistent `BlockTables`，通过 `idx_mapping` gather |
| 异步保护 | preprocessing 周围使用 `synchronize_input_prep` barrier | async-first buffers、staged writes 和 round-robin UVA pools |

V1 的 `_update_states()` 会移除未调度的 rows、重新加入 requests、调用
`InputBatch.condense()`、根据 backend 要求调整 batch 顺序并刷新 metadata。
由于 active state 可能暂时不在 persistent batch 中，所以需要
`CachedRequestState`。

MRV2 将 request-lifetime state 与当前 step 的紧凑 batch 分离。执行顺序
变化时，不再需要移动每个 request 的 persistent row。

## Request 如何获得 persistent row

MRV2 的 `RequestState` 初始化：

```text
req_id_to_index
index_to_req_id
free_indices = [0, ..., max_num_reqs - 1]
```

当 `add_requests()` 收到新 request 时：

1. `RequestState.add_request()` 从 free indices 中取出一个 index。
2. 写入 request ID 的双向映射。
3. 在该 index 初始化 request-lifetime fields。
4. Model-specific、block-table、LoRA 和 sampler state 使用相同的 request
   index。
5. Forward pass 之前应用 staged writes。

这个 row 会一直保留到调用 `remove_request()`。Finish 和 preemption 都会
释放 row；被 preempt 的 request 恢复时会重新加入，并可能获得不同的 row。

因此，“persistent row”表示在一次 active lifetime 内保持稳定，而不是跨越
preemption/resume 永久固定。

## RequestState 保存的状态

v0.26 的 `vllm/v1/worker/gpu/states.py::RequestState` 持有：

| 状态 | 表示形式 | 用途 |
|---|---|---|
| request ID ↔ row | Python dicts | 稳定的 row 查询 |
| free rows | Python list | slot 分配与复用 |
| `all_token_ids` | UVA-backed `StagedWriteTensor` | 保存完整 token history，避免巨大的 GPU 副本 |
| `prompt_len` | `UvaBackedTensor` | 原始用户 prompt 长度 |
| `prefill_len` | `UvaBackedTensor` | 必须经过 prefill 的 token 数 |
| `total_len` | GPU `StagedWriteTensor` | prompt 与 generated tokens 的总长度 |
| computed prefill count | NumPy array | CPU 侧的 prefill phase 判断 |
| `num_computed_tokens` | GPU staged tensor + optimistic NumPy mirror | GPU 可见进度与 CPU 输入准备 |
| last sampled tokens | GPU tensor | 下一次 decode/input 准备 |
| maximum sequence length | NumPy array | request stopping/PP metadata |
| draft tokens | GPU tensor | speculative path |
| next prefill tokens | GPU tensor | prefill 输入准备 |

KV block tables、sampler configuration、LoRA state、multimodal state 和
model-specific recurrent state 属于独立组件，只是使用同一个 row 作为 key。
它们不应被描述为 `RequestState` 自身的字段。

## StagedWriteTensor 为什么使用 UVA

只有少数 rows 或 slices 发生变化时，`StagedWriteTensor` 可以避免复制整个
persistent tensor：

```text
在 CPU 上暂存 row/start/content diffs
→ 打包 write metadata
→ 通过可用的 UVA buffer 暴露或复制 metadata
→ 异步复制打包后的 contents
→ 启动 Triton kernel 修改 persistent tensor
```

UVA 在这里有两类用途：

1. Write indices、starts 和 cumulative lengths 通过 `UvaBufferPool`
   提供，使 GPU kernel 能直接读取 pinned CPU memory，不需要阻塞式
   metadata copy。
2. `all_token_ids` 这类体积很大但访问不频繁的状态可以直接由 UVA-backed
   memory 承载，从而避免数 GiB 的 GPU 副本。

这也解释了 Day 1 的 WSL Gate：MRV2 在初始化阶段就会构造这些 buffers，
因此 pinned memory 被禁用时，UVA 会在模型开始执行前变得不可用。

## idx_mapping 如何定义每 step 顺序

Persistent row 顺序与实际执行顺序是有意分离的。

`prepare_inputs()` 执行以下步骤：

1. `sort_batch_req_ids()` 根据本 step 的要求选择 request 顺序，其中包含
   decode query length 的排序规则。
2. 通过 `req_states.req_id_to_index` 查询每个 request ID。
3. 将得到的 NumPy vector 异步复制到 GPU，作为 `idx_mapping`。

例如：

| Persistent row | Active request |
|---:|---|
| 2 | B |
| 7 | A |
| 11 | C |

如果本 step 的执行顺序为 `[A, C, B]`，则：

```text
idx_mapping = [7, 11, 2]
```

GPU preparation kernels 使用该映射读取 persistent token/progress state，
生成 `input_ids`、positions、sequence lengths 和 sampling metadata。
`BlockTables.gather_block_tables()` 使用同一映射，为 forward pass 生成紧凑
的 block-table view。

使用 speculative decoding 时，`expanded_idx_mapping` 会重复或展开原始
mapping，使多个 logits rows 仍然能够定位到正确的 persistent request
state。

## 为什么 step N+1 不会破坏 step N

MRV2 的 async-first 设计依赖以下机制协同工作：

1. Engine Core 以 non-blocking 方式提交模型执行，并且可以在消费最早结果
   之前继续将下一批加入队列。
2. Persistent CPU source state 通常不直接充当 in-flight GPU copy 正在读取
   的 pinned buffer。临时 pinned copies 将 CPU mutation 与 GPU reads
   隔离。
3. UVA metadata 使用 round-robin `UvaBufferPool`。其深度至少等于允许同时
   in-flight 的最大 batch 数，因此 step N+1 不会立即覆盖 step N 的
   metadata buffer。
4. 大型 persistent GPU state 通过 staged-write kernels 更新。更新操作和
   后续消费者按顺序进入 CUDA stream，依赖设备执行顺序，不需要 CPU
   synchronization barrier。
5. 每 step 的 `idx_mapping` 与紧凑 input buffers 只描述当前 step，固定的
   request rows 不需要在 GPU 执行过程中重新排列。

这不代表系统从结构上消除了所有 race。任何新增 MRV2 功能如果复用 pinned
CPU buffer、调用 `.item()`、执行 unpinned blocking transfer，或者在上述
lifetime 规则之外写入 shared surface，都可能重新引入 barrier 或 race。

## Scheduler Trace 源码地图

| 问题 | v0.26 主要源码 |
|---|---|
| Scheduler 决策与输出 | `vllm/v1/core/sched/scheduler.py`、`output.py` |
| Engine Core 异步提交 | `vllm/v1/engine/core.py` |
| Worker dispatch | `vllm/v1/worker/gpu_worker.py` |
| Request add/update/execute | `vllm/v1/worker/gpu/model_runner.py` |
| Persistent request state | `vllm/v1/worker/gpu/states.py` |
| Staged writes/UVA pools | `vllm/v1/worker/gpu/buffer_utils.py` |
| Persistent/gathered block tables | `vllm/v1/worker/gpu/block_table.py` |
| 架构设计目标 | `docs/design/model_runner_v2.md` |

后续 trace patch 应当在进入这条 worker 路径前捕获 Scheduler state。MRV2
row/index 信息属于可选的 worker-side 扩展，不能为了丰富一行日志而强制
GPU synchronization。

## 性能对比警告

历史 v0.15 GPU 结果与 v0.26 路径不仅使用了不同 runner，实测 attention
backend 也不同：

```text
v0.15 历史分支：V1 runner + FLASHINFER
v0.26 实测分支：MRV2 + TRITON_ATTN
```

因此，不能把 v0.15/v0.26 的延迟差异单独归因于 MRV2。要得到具有因果意义
的 runner 对比，需要在 v0.26 内控制 MRV1/MRV2 实验，并保持模型、backend、
workload、内存配置和机器状态一致。

## 5 分钟口述

vLLM 的外层循环仍然是 Scheduler 调度、non-blocking 模型执行，再使用输出
更新 Scheduler。MRV2 的关键变化在于 model runner 如何存储和准备 request
state。

V1 包含两个耦合层：Python `CachedRequestState` 备份，以及 rows 同时作为
直接模型输入的 `InputBatch`。当 request 暂时不出现在某一步时，V1 会移除
rows，之后再重新加入、压紧空洞并刷新 metadata。异步调度下，这种设计需要
额外保护 row 移动和 CPU/GPU buffer 生命周期。

MRV2 为每个 active request 分配固定 row。Token history、lengths 和
computed-token progress 保存在 persistent GPU 或 UVA-backed state 中。
Request 可以按任意顺序执行，因为每一步都会创建 `idx_mapping`，把紧凑
batch 顺序映射回 persistent rows。GPU kernels 再根据该映射准备 input IDs、
positions、sequence lengths 和 gather 后的 block tables。

增量变化通过 `StagedWriteTensor` 处理：CPU 只记录发生变化的 rows，GPU
kernel 再应用这些 diffs。UVA 既用于提供少量 write metadata，也可以承载
很大的 token-history state。为了保持 async-safe，pinned/UVA buffers 的
pool 深度至少覆盖 in-flight batch 数，因此 CPU 准备 step N+1 时不会覆盖
GPU step N 正在读取的 buffer。

实现 Scheduler Trace 时，我会在 CPU 侧记录 Scheduler decisions，不调用
`.item()`，也不引入同步。Persistent row 和 index mapping 证据将作为可选
的 MRV2-side 扩展，不会与 Scheduler policy 的结论混为一谈。

## 留给后续阶段的问题

- 哪些 row/index 字段可以在不增加 worker RPC 或 synchronization 的情况下
  暴露给 tracing？
- 实测 `TRITON_ATTN` backend 的顺序约束，会怎样影响 mixed
  prefill/decode batch 的 `sort_batch_req_ids()`？
- 在 6 GiB GPU 限制下，8K/16K workload 能否直接运行？如果不能，是否应
  使用 2K/4K 比例完成 GPU 实测，同时通过 Scheduler tests 保留 8K/16K
  逻辑验证？
- 长时间启用 WSL pinned memory 时，其稳定性是否能超过 Day 1 smoke test
  的持续时间？
