# vLLM v0.15 V1 → v0.26 MRV2 差异地图

Date: 2026-07-26

Status: Day 2 source map; no Scheduler instrumentation or policy change

## Scope and evidence rule

This document compares:

- historical V1 source:
  `f176443446f659dbab5315e056e605d8984fd976`;
- current MRV2 source:
  `f2654939e69b4069b13977e9aef3e31d4dcaf051`;
- measured v0.26 execution path from Day 1:
  MRV2 + `TRITON_ATTN`.

The goal is limited to the path needed by Scheduler Trace:

```text
SchedulerOutput
→ EngineCore
→ Executor / GPUWorker
→ GPUModelRunner request-state update
→ per-step index mapping
→ GPU input metadata
```

It is not a complete V1/MRV2 feature comparison.

### Hypothesis

MRV2 reduces async-scheduling bookkeeping by separating persistent
request-lifetime state from per-step batch order. Requests keep stable rows;
each step supplies an indirection that gathers those rows into the required
execution order.

### Observables

- ownership and lifetime of request state;
- new/cached/finished/preempted request handling;
- row allocation and reuse;
- CPU-to-GPU state update mechanism;
- per-step request ordering and `idx_mapping`;
- buffers that can be touched by CPU step N+1 while GPU step N is in flight.

### Decision criteria

The map is sufficient when it can answer the six Day 2 questions without
claiming unmeasured runtime state:

1. How does `SchedulerOutput` reach `GPUModelRunner.update_requests`?
2. How does a request obtain a persistent row?
3. What long-lived state is stored in `RequestState`?
4. Why does `StagedWriteTensor` require UVA?
5. How does `idx_mapping` form the execution order?
6. How can the CPU prepare step N+1 without corrupting GPU step N?

## Stable outer control flow

The high-level Engine Core loop remains recognizable across both revisions:

```text
Scheduler.schedule()
→ SchedulerOutput
→ model_executor.execute_model(..., non_block=True)
→ GPUWorker.execute_model()
→ GPUModelRunner.execute_model()
→ Scheduler.update_from_output()
```

In v0.26, the direct single-step path is:

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

`EngineCore.step_with_batch_queue` extends this with a queue of in-flight
batches. It tries to schedule another batch before blocking on the oldest
result. The major V1 → MRV2 change is therefore inside the worker/model-runner
state preparation, rather than a completely new Scheduler API.

## SchedulerOutput → MRV2 request update

`Scheduler.schedule()` constructs a `SchedulerOutput` containing:

| Field | Meaning for the model runner |
|---|---|
| `scheduled_new_reqs` | Full data for first-time or re-added requests |
| `scheduled_cached_reqs` | Incremental data for requests already cached |
| `num_scheduled_tokens` | Per-request token count for this step |
| `total_num_scheduled_tokens` | Global scheduled token count |
| `finished_req_ids` | State that can be released |
| `preempted_req_ids` | MRV2 state treated as finished |
| cached `new_block_ids` | KV block-table extension |

The v0.26 call order inside `GPUModelRunner.execute_model` is:

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

Important distinction:

- `add_requests()` handles full state for new/re-added requests and applies
  their staged request/model/sampler writes;
- `update_requests()` handles cached-request diffs, especially
  `num_computed_tokens`, new KV block IDs, fresh-block zeroing, and KV
  copy-on-write operations;
- the call to `update_requests()` is not made by the Scheduler directly. The
  SchedulerOutput crosses Engine Core, Executor, and GPUWorker first.

## Persistent state: V1 versus MRV2

| Concern | v0.15 V1 | v0.26 MRV2 |
|---|---|---|
| Python request backup | `dict[req_id, CachedRequestState]` | No equivalent general backup object |
| Persistent batch | `InputBatch` mixes state and direct model inputs | Persistent state is separate from per-step `InputBatch` |
| Row lifetime | Rows may be removed, filled and condensed | Fixed row for active lifetime |
| Unscheduled request | Removed from V1 persistent batch but retained in cached request dict | State stays in its slot unless finished/preempted |
| Preemption | Cached state supports later resume/reinsert | Treated as finish; resume is a fresh add |
| Per-step order | Persistent `InputBatch` is reordered/condensed | `idx_mapping` gathers fixed rows in step order |
| Token state | CPU tensor/list centric, then copied/prepared | GPU/UVA persistent tensors plus CPU mirrors where needed |
| KV block table | Part of V1 `InputBatch` layout | Separate persistent `BlockTables`, gathered using `idx_mapping` |
| Async protection | `synchronize_input_prep` barrier around preprocessing | Async-first buffers, staged writes, and round-robin UVA pools |

V1's `_update_states()` removes unscheduled rows, re-adds requests, calls
`InputBatch.condense()`, may reorder the batch for the backend, and refreshes
metadata. `CachedRequestState` is needed because active state may no longer be
present in the persistent batch.

MRV2 keeps request-lifetime state independent of the current step's compact
batch. This removes the need to move every request's persistent row when the
step order changes.

## How a request obtains a persistent row

MRV2 `RequestState` initializes:

```text
req_id_to_index
index_to_req_id
free_indices = [0, ..., max_num_reqs - 1]
```

When `add_requests()` receives a new request:

1. `RequestState.add_request()` pops one free index.
2. Both request-ID mappings are populated.
3. Request-lifetime fields are initialized at that index.
4. Model-specific, block-table, LoRA, and sampler state use the same request
   index.
5. Staged writes are applied before the forward pass.

The row remains assigned until `remove_request()` is called. Finish and
preemption both release it. A resumed preempted request is added again and may
receive a different row.

“Persistent row” therefore means stable for one active lifetime, not stable
forever across preemption/resume.

## What RequestState stores

The v0.26 `vllm/v1/worker/gpu/states.py::RequestState` owns:

| State | Representation | Purpose |
|---|---|---|
| request ID ↔ row | Python dicts | Stable row lookup |
| free rows | Python list | Slot allocation/reuse |
| `all_token_ids` | UVA-backed `StagedWriteTensor` | Full token history without a huge GPU duplicate |
| `prompt_len` | `UvaBackedTensor` | Original user prompt length |
| `prefill_len` | `UvaBackedTensor` | Tokens that must pass through prefill |
| `total_len` | GPU `StagedWriteTensor` | Prompt plus generated length |
| computed prefill count | NumPy array | CPU-side prefill-phase decisions |
| `num_computed_tokens` | GPU staged tensor + optimistic NumPy mirror | Progress visible to GPU and CPU prep |
| last sampled tokens | GPU tensor | Next decode/input preparation |
| maximum sequence length | NumPy array | Request stopping/PP metadata |
| draft tokens | GPU tensor | Speculative path |
| next prefill tokens | GPU tensor | Prefill input preparation |

KV block tables, sampler configuration, LoRA state, multimodal state, and
model-specific recurrent state are separate components keyed by the same row.
They should not be described as fields of `RequestState`.

## Why StagedWriteTensor uses UVA

`StagedWriteTensor` avoids copying an entire persistent tensor when only a few
rows or slices change:

```text
stage row/start/content diffs on CPU
→ pack write metadata
→ expose/copy metadata through an available UVA buffer
→ copy packed contents asynchronously
→ launch a Triton kernel that patches the persistent tensor
```

UVA is used in two distinct ways:

1. The write indices, starts, and cumulative lengths are supplied through
   `UvaBufferPool`, so the GPU kernel can read pinned CPU memory without a
   blocking metadata copy.
2. Very large, infrequently accessed state such as `all_token_ids` can itself
   be UVA-backed, avoiding a multi-gigabyte GPU duplicate.

This explains the Day 1 WSL gate: MRV2 constructs these buffers during
initialization, so disabled pinned memory makes UVA unavailable before model
execution begins.

## How idx_mapping defines per-step order

Persistent row order and execution order are intentionally different.

In `prepare_inputs()`:

1. `sort_batch_req_ids()` chooses the request order required for the step,
   including the decode-query-length ordering rule.
2. Each request ID is looked up in `req_states.req_id_to_index`.
3. The resulting NumPy vector is copied asynchronously to GPU as
   `idx_mapping`.

Example:

| Persistent row | Active request |
|---:|---|
| 2 | B |
| 7 | A |
| 11 | C |

If the step order is `[A, C, B]`, then:

```text
idx_mapping = [7, 11, 2]
```

GPU preparation kernels use this mapping to read persistent token/progress
state and create `input_ids`, positions, sequence lengths, and sampling
metadata. `BlockTables.gather_block_tables()` uses the same mapping to produce
the compact block-table view for the forward pass.

With speculative decoding, `expanded_idx_mapping` repeats/expands the mapping
so multiple logits rows still refer back to the correct persistent request
state.

## Why step N+1 does not corrupt step N

MRV2's async-first design relies on several cooperating mechanisms:

1. Engine Core submits model execution non-blockingly and may enqueue another
   batch before consuming the oldest result.
2. Persistent CPU source state is generally not itself the pinned buffer being
   read by an in-flight GPU copy. Temporary pinned copies separate CPU mutation
   from GPU reads.
3. UVA metadata uses a round-robin `UvaBufferPool`. Its depth is configured to
   be at least the maximum number of concurrent in-flight batches, so step
   N+1 does not immediately overwrite step N's metadata buffer.
4. Large persistent GPU state is updated by staged-write kernels. Updates and
   subsequent consumers are enqueued on the CUDA stream, preserving device
   execution order without a CPU synchronization barrier.
5. Per-step `idx_mapping` and compact input buffers describe that step; the
   fixed request rows do not need to be reshuffled under the GPU.

This does not mean all races are impossible by construction. Any new MRV2
feature that reuses a pinned CPU buffer, calls `.item()`, performs an unpinned
blocking transfer, or writes a shared surface outside these lifetime rules can
reintroduce a barrier or race.

## Source map for Scheduler Trace

| Question | Primary v0.26 source |
|---|---|
| Scheduler decision/output | `vllm/v1/core/sched/scheduler.py`, `output.py` |
| Async Engine Core submission | `vllm/v1/engine/core.py` |
| Worker dispatch | `vllm/v1/worker/gpu_worker.py` |
| Request add/update/execute | `vllm/v1/worker/gpu/model_runner.py` |
| Persistent request state | `vllm/v1/worker/gpu/states.py` |
| Staged writes/UVA pools | `vllm/v1/worker/gpu/buffer_utils.py` |
| Persistent/gathered block tables | `vllm/v1/worker/gpu/block_table.py` |
| Architectural intent | `docs/design/model_runner_v2.md` |

For the later trace patch, Scheduler state should be captured before this
worker path. MRV2 row/index information belongs to an optional worker-side
extension; it must not force GPU synchronization just to enrich a log line.

## Performance-comparison warning

The historical v0.15 GPU result used a different runner and a different
measured attention backend from the v0.26 path:

```text
v0.15 historical lane: V1 runner + FLASHINFER
v0.26 measured lane:    MRV2 + TRITON_ATTN
```

Therefore, a v0.15/v0.26 latency difference cannot be attributed to MRV2
alone. A causal runner comparison would need a controlled v0.26 MRV1/MRV2
experiment with the same model, backend, workload, memory settings, and
machine state.

## Five-minute explanation

The outer vLLM loop is still Scheduler schedule, non-blocking model execution,
then Scheduler update from output. The important MRV2 change is how the model
runner stores and prepares request state.

V1 has two coupled layers: a Python `CachedRequestState` backup and an
`InputBatch` whose rows are also direct model inputs. When requests disappear
from a step, V1 removes rows, later re-adds them, condenses holes, and refreshes
metadata. This becomes complicated under asynchronous scheduling because row
movement and CPU/GPU buffer lifetimes must be protected.

MRV2 assigns each active request a fixed row. Token history, lengths and
computed-token progress live in persistent GPU or UVA-backed state. The
request can appear in any execution order because each step builds
`idx_mapping`, which maps compact batch order back to persistent rows. GPU
kernels then prepare input IDs, positions, sequence lengths and gathered block
tables from that mapping.

Incremental changes use `StagedWriteTensor`: CPU code records only changed
rows, then a GPU kernel applies the diffs. UVA supplies small write metadata
and can hold very large token-history state. To remain async-safe, pinned/UVA
buffers are pooled with at least the in-flight batch depth, so CPU preparation
for step N+1 does not overwrite the buffer GPU step N is reading.

For Scheduler Trace, I will log Scheduler decisions on the CPU without
`.item()` or synchronization. Persistent-row and index-mapping evidence will
be an optional MRV2-side extension, not mixed into claims about Scheduler
policy.

## Unresolved questions for later days

- Which row/index fields can be exposed for tracing without adding a worker RPC
  or synchronization?
- How does the measured `TRITON_ATTN` backend's ordering constraint affect
  `sort_batch_req_ids()` for mixed prefill/decode batches?
- Under the 6 GiB GPU limit, will the 8K/16K workload run directly, or should
  GPU evidence use a 2K/4K scale while Scheduler tests preserve 8K/16K logic?
- Does long-running WSL pinned-memory use remain stable beyond the Day 1 smoke
  duration?
