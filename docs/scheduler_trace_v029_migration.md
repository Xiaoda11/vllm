# Scheduler Trace on v0.29

This personal-fork port is based on v0.29.0 (`98dff2a81d747d1dba01a47f939f48c3526d4206`).
It carries forward the Scheduler trace from personal PR #3
(`4e29a11f9c5deb93d8d3a3462c9f71ea7a5e963a`) and restores the core MRV2
batch trace on the modular v0.29 GPU model runner. It does not port the v0.26
waiting-bypass policy.

## Scope

- Preserve opt-in background JSONL writing, legacy environment-variable support,
  schema version 1, dual budgets, request/queue snapshots, prefix hits,
  allocation failures and KV-delivery-aware preemption.
- Add `kv_connector.has_sync_kv_loads` to scheduler-step events.
- Copy `kv_connector.offered_block_state` before building connector metadata.
  This records per-request grouped block IDs and Mamba boundary hand-offs as
  `[group_id, block_id, boundary_tokens]`. Copying nested CPU lists prevents later
  connector consumption or mutation from changing records queued for writing.
- Preserve upstream clearing of `SchedulerOutput.kv_connector_block_state`
  before dispatch to workers. Trace-disabled steps do not construct this copy.
- Restore `model_runner_batch` events in `vllm/v1/worker/gpu/model_runner.py`
  from existing CPU/Python batch metadata only. The v0.26 join fields
  (`request_ids`, persistent rows and per-request scheduled tokens) remain, while
  v0.29 adds explicit scheduler-logical, runner-effective and graph-padded token
  counts.
- Propagate a Scheduler/runner correlation step ID only when Scheduler tracing is
  enabled. Trace-disabled `SchedulerOutput` objects keep their ordinary shape.
- Preserve the correlation ID on the `InputBatch` instance through optional PCP
  partitioning rather than storing an in-flight step ID on `GPUModelRunner`.
  This avoids a runner-global correlation slot that could be overwritten by
  overlapping execute/sample batches.
- Add `sampler_batch_shard` events for v0.29 TP batch-sharded sampling. They
  record the CPU-derived global/local request layouts, persistent-row ownership,
  per-rank request counts and per-rank logit splits. GPU gather/sort tensors are
  intentionally not read back for tracing.
- Give every MRV2 worker process its own JSONL file using
  `<scheduler-stem>.mrv2.pid<PID><suffix>`. Each MRV2 event also carries the
  vLLM global `worker_rank`, so files can be merged semantically without making
  multiple TP/PP/PCP workers compete for one exclusive file path.

The connector fields are additive schema-1 fields; older records omit them.
`offered_block_state: null` means no connector block state was offered, whereas
empty dictionaries mean an empty state was offered. An offer does not prove that
transfer started, completed, or persisted successfully. This is not a complete
Mamba checkpoint lifecycle trace. Request block differences remain mapping
changes, not counts of physical allocation/free operations.

The MRV2 batch event distinguishes three quantities that are no longer
interchangeable on v0.29:

- `scheduler_logical`: Scheduler token budget for the step;
- `runner_effective_unpadded`: tokens the runner intends to execute after
  CPU-visible trimming such as adaptive verification;
- `model_input_after_padding`: model-input rows after CUDA-graph padding.

This prevents adaptive-verification trimming and graph padding from being
misreported as Scheduler decisions.

For batch-sharded sampling, v0.29 derives TP ownership as
`persistent_row % tp_size`. The trace records that ownership together with the
local sampler shard. It does not read `gathered_src_indices`,
`sorted_logits_indices`, or other GPU-only shard-plan tensors.

## Validation

On 2026-09-14, the branch CI runs two CPU jobs on Python 3.12.

The isolated trace suite reports **22 passed**. It covers:

- JSONL writer behavior, including process-local MRV2 file naming;
- Scheduler event semantics and extracted Scheduler integration contracts;
- MRV2 batch event semantics, including worker rank, adaptive-verification
  trimming and graph padding;
- TP sampler-shard ownership semantics and validation of local ownership/logit
  splits;
- Scheduler -> runner step-correlation serialization through a Python pickle
  round trip;
- MRV2 lifecycle/wiring invariants, including trace-only worker-rank lookup,
  per-`InputBatch` step correlation, writer close, and sampler-shard emission;
- static checks that trace event arguments do not introduce `.item()`, `.cpu()`,
  `.tolist()` or `synchronize()` calls, and that GPU-only sampler shard-plan
  tensors are not pulled back for tracing.

The real-Scheduler/IPC CPU job reports **7 passed** (`VLLM_TARGET_DEVICE=cpu`).
It covers:

- trace disabled on the ordinary scheduling path, including absence of the
  trace-only correlation attribute;
- trace enabled for a real WAITING -> RUNNING prefill with KV block allocation,
  including the Scheduler/runner step correlation ID;
- the dynamically attached `scheduler_trace_step_id` surviving an actual process
  boundary through vLLM's `MessageQueue`: the parent emits the same
  `("execute_model", (scheduler_output,), {}, output_rank)` RPC tuple used by
  `MultiprocExecutor.collective_rpc()`, and a spawned reader process attaches via
  `MessageQueue.create_from_handle()` and verifies the step ID and scheduled-token
  map after dequeue;
- a real waiting-path KV allocation failure with the request left queued;
- request completion through `update_from_output()`, followed by the next
  Scheduler step flushing the request through `finished_req_ids`;
- real KV-pressure preemption using the upstream 10-usable-block / two 80-token
  request construction, including `RUNNING -> PREEMPTED`, the preempted request
  ID and freed KV blocks in the trace event;
- a synchronous KV-load path through the v0.29 Scheduler using the repository's
  mock KV connector, verifying that `SchedulerOutput.has_sync_kv_loads` and the
  emitted `kv_connector.has_sync_kv_loads` trace field are both true.

The MessageQueue fixture exercises the actual vLLM SHM/ZMQ queue serialization,
reader attachment and cross-process dequeue path. This is stronger than a plain
pickle round trip and validates the transport used by the default multiprocess
executor for Scheduler RPC payloads. It still does **not** instantiate a full
`MultiprocExecutor`/`WorkerProc`, initialize a model runner, or prove a complete
engine Scheduler -> worker -> MRV2 event join.

The synchronous-KV fixture exercises real Scheduler connector integration but
uses a mock connector to make the remote match/load mode deterministic. It does
not validate an actual network transfer, remote persistence, transport failure,
or end-to-end P/D disaggregation.

Ruff check/format checks and the repository-pinned Ruff/typos hooks passed on the
original migration checkpoint. Full pre-commit execution remains blocked during
markdownlint environment installation because npm returns EALLOWGIT for the
hook's git+file package. Full hooks must be rerun in a supported environment;
this is not an all-checks-passed claim.

The current evidence establishes CPU contracts, real Scheduler integration and
real cross-process MessageQueue preservation of the trace correlation field for
the tested path. It does **not** establish real TP/PP/PCP worker execution, GPU
correctness, or performance. In particular, process-local MRV2 file naming and
TP sampler ownership are contract-tested but have not yet been exercised by a
live multi-GPU batch-sharded-sampling run. No v0.29 overhead claim is made from
the v0.26 benchmark results.

## Remaining gates

1. Run a full engine smoke test through `MultiprocExecutor`/`WorkerProc` and
   verify Scheduler and MRV2 records join by `step_id`, each worker writes a
   distinct MRV2 file, and lifecycle shutdown flushes all files cleanly. The
   underlying MessageQueue process boundary is now validated separately.
2. Run a real TP batch-sharded-sampling GPU test and validate recorded
   `worker_rank` / `tp_rank`, request ownership and logit splits against runtime
   behavior.
3. Exercise an actual KV connector transport rather than the mock connector,
   including synchronous-load completion/failure behavior and, where relevant,
   P/D disaggregation semantics.
4. Validate PCP semantics. `model_runner_batch` is emitted before
   `pcp.maybe_partition_pcp_batch`, so its request/token counts describe the
   pre-partition batch; the step ID is preserved onto the partitioned batch, but
   PCP-local execution counts are not yet a dedicated event.
5. Validate alternate executor paths such as Ray if they are part of the target
   support matrix.
6. Run trace-off/on real-model GPU comparisons for output equivalence and
   overhead before making any performance claim.

Implementation assisted by Codex. Human review is required before upstream
submission; this migration is for the personal experiment fork.
