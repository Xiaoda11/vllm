# Scheduler Trace on v0.29

This personal-fork port is based on v0.29.0 (`98dff2a81d747d1dba01a47f939f48c3526d4206`).
It carries forward the Scheduler trace from personal PR #3
(`4e29a11f9c5deb93d8d3a3462c9f71ea7a5e963a`) and restores the core MRV2
batch event on the modular v0.29 GPU model runner. It does not port the v0.26
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

The connector fields are additive schema-1 fields; older records omit them.
`offered_block_state: null` means no connector block state was offered, whereas
empty dictionaries mean an empty state was offered. An offer does not prove that
transfer started, completed, or persisted successfully. This is not a complete
Mamba checkpoint lifecycle trace. Request block differences remain mapping
changes, not counts of physical allocation/free operations.

The MRV2 event distinguishes three quantities that are no longer interchangeable
on v0.29:

- `scheduler_logical`: Scheduler token budget for the step;
- `runner_effective_unpadded`: tokens the runner intends to execute after
  CPU-visible trimming such as adaptive verification;
- `model_input_after_padding`: model-input rows after CUDA-graph padding.

This prevents adaptive-verification trimming and graph padding from being
misreported as Scheduler decisions.

## Validation

On 2026-09-13, the branch CI runs two CPU jobs.

The isolated trace suite uses Python 3.12 / pytest 9.1.1 and now reports
**15 passed**. It covers the writer, Scheduler event semantics, extracted
Scheduler integration contracts, MRV2 event semantics and MRV2 wiring
invariants. Static wiring checks verify that the MRV2 trace call path does not
introduce `.item()`, `.cpu()`, `.tolist()` or `synchronize()` calls around event
construction. The event builder itself accepts only CPU/Python values and has no
Torch dependency.

The real-Scheduler job uses `VLLM_TARGET_DEVICE=cpu` with the repository CPU
runtime dependencies and reports **3 passed**. It calls `Scheduler.schedule()`
directly and covers:

- trace disabled on the ordinary scheduling path, including absence of the
  trace-only correlation attribute;
- trace enabled for a real WAITING -> RUNNING prefill with KV block allocation,
  including the Scheduler/runner step correlation ID;
- a real waiting-path KV allocation failure with the request left queued.

Ruff check/format checks and the repository-pinned Ruff/typos hooks passed on the
original migration checkpoint. Full pre-commit execution remains blocked during
markdownlint environment installation because npm returns EALLOWGIT for the
hook's git+file package. Full hooks must be rerun in a supported environment;
this is not an all-checks-passed claim.

The current evidence establishes CPU contracts and real Scheduler integration
for the tested paths. It does **not** establish GPU correctness or performance.
No v0.29 overhead claim is made from the v0.26 benchmark results.

## Remaining gates

1. Verify that the trace-only step correlation survives the actual worker/executor
   transport, not only the in-process Scheduler fixture. A serialization contract
   can narrow this risk but is not a substitute for a real engine run.
2. Extend the real-Scheduler fixture to completion/preemption and a real connector
   path, including synchronous KV-load behavior where practical.
3. Add explicit sampler batch-shard tracing when batch-sharded sampling is
   enabled. The current event records that sharding is enabled but does not record
   per-rank sampler ownership/shard metadata.
4. Validate PCP semantics. `model_runner_batch` is currently emitted before
   `pcp.maybe_partition_pcp_batch`, so its request/token counts describe the
   pre-partition batch rather than PCP-local execution.
5. Run a real-model GPU smoke test and compare trace-off/on outputs and overhead.

Implementation assisted by Codex. Human review is required before upstream
submission; this migration is for the personal experiment fork.
