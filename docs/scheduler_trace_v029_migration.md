# Scheduler Trace on v0.29

This personal-fork port is based on v0.29.0 (`98dff2a81d747d1dba01a47f939f48c3526d4206`).
It carries forward the Scheduler-only trace from personal PR #3
(`4e29a11f9c5deb93d8d3a3462c9f71ea7a5e963a`). It does not port the v0.26
waiting-bypass policy or the complete MRV2 tracing chain.

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

The connector fields are additive schema-1 fields; older records omit them.
`offered_block_state: null` means no connector block state was offered, whereas
empty dictionaries mean an empty state was offered. An offer does not prove that
transfer started, completed, or persisted successfully. This is not a complete
Mamba checkpoint lifecycle trace. Request block differences remain mapping
changes, not counts of physical allocation/free operations.

## Validation

On 2026-09-13, Python 3.12 with pytest 9.1.1:

```bash
uv venv --python 3.12
uv pip install pytest==9.1.1
.venv/bin/python -m pytest -q lab_tests
```

Result: **9 passed**. Existing writer, event and extracted Scheduler-method
contracts pass on the port. The connector test verifies that clearing/mutating
the original block state does not change the emitted event. An additional test
distinguishes absent and empty connector state. Ruff check/format checks pass.
The repository-pinned Ruff and typos hooks also pass. Full pre-commit execution
is blocked during markdownlint environment installation: npm returns EALLOWGIT
for the hook's git+file package. Full hooks must be rerun in a supported
environment; the branch checkpoint is committed without rerunning that blocked
hook installation. This is not an all-checks-passed claim.

These are isolated CPU contracts, not a full Scheduler fixture, complete vLLM
CI, model execution, or GPU performance validation. No v0.29 overhead or
correctness claim is made from the v0.26 benchmark results.

## Remaining gates

1. Run a complete real-Scheduler fixture with trace enabled and disabled using
   supported vLLM dependencies; exercise normal scheduling, allocation failure,
   completion, and connector paths.
2. Run a real-model smoke test and compare trace-off/on output and overhead.
3. Restore MRV2 events while distinguishing logical scheduled tokens, graph
   padding, and sampler batch sharding. Validate cross-layer identities in each
   supported execution mode before making end-to-end claims.

Implementation assisted by Codex. Human review is required before upstream
submission; this migration is for the personal experiment fork.
