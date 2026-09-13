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

The pickle contract is relevant to the default v0.29 `MultiprocExecutor` path,
whose shared-memory `MessageQueue` serializes RPC payloads with Python
`pickle.Pickler`. This narrows the cross-process correlation risk, but it is not
a substitute for a real multi-process engine run.

The real-Scheduler CPU job reports **3 passed** (`VLLM_TARGET_DEVICE=cpu`). It
calls `Scheduler.schedule()` directly and covers:

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
for the tested paths. It does **not** establish real TP/PP/PCP worker execution,
GPU correctness, or performance. In particular, process-local MRV2 file naming
and TP sampler ownership are contract-tested but have not yet been exercised by
a live multi-GPU batch-sharded-sampling run. No v0.29 overhead claim is made from
the v0.26 benchmark results.

## Remaining gates

1. Run a real engine smoke test through the default multiprocess executor and
   verify Scheduler/MRV2 records join by `step_id`, that each worker writes a
   distinct MRV2 file, and that lifecycle shutdown flushes all files cleanly.
2. Run a real TP batch-sharded-sampling GPU test and validate recorded
   `worker_rank` / `tp_rank`, request ownership and logit splits against runtime
   behavior.
3. Extend the real-Scheduler fixture to completion/preemption and a real
   connector path, including synchronous KV-load behavior where practical.
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
