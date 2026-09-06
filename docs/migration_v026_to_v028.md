# Scheduler Trace Lab migration: v0.26 -> v0.28

## Goal

Port the long-lived observability infrastructure from the v0.26 experiment to a clean v0.28 baseline without rebasing the old experiment branch.

## Branch policy

- `exp/mrv2-scheduler-trace-v026`: frozen reproducible v0.26 baseline.
- `exp/mrv2-scheduler-trace-v028`: clean v0.28 baseline for the migration.
- `port/trace-infra-v028`: active implementation branch for the first migration slice.
- `docs/migration-audit-v026-v028`: audit/documentation branch.

The old v0.26 branch is treated as an evidence snapshot. New work starts from the v0.28 release tag.

## Migration invariants

The v0.26 trace design should preserve these semantic properties on v0.28:

1. Trace is opt-in and disabled by default.
2. Instrumentation observes scheduler decisions and existing CPU-side metadata only.
3. No GPU tensor readback, `.item()`, or extra CUDA synchronization is introduced.
4. Trace schema remains machine-readable and analyzers can detect schema/version mismatches.
5. Policy experiments are migrated only after the observability path is validated.

## Initial code audit

### Scheduler

The scheduler still lives at `vllm/v1/core/sched/scheduler.py`, so the primary observability boundary remains recognizable.

However, v0.28 has additional scheduler state and execution constraints that must be handled explicitly before reusing old instrumentation, including encoder-only handling, KV-delivery requirements, newer encoder-cache manager construction, and expanded speculative/prefill lookahead state.

### Trace module

The custom v0.26 file `vllm/v1/core/sched/trace.py` is not present in the clean v0.28 baseline. It should therefore be reintroduced as a new, isolated module rather than copied through a large scheduler diff.

### Migration order

1. Reintroduce trace schema/writer as an isolated utility.
2. Port scheduler instrumentation around scheduling inputs/outputs and request lifecycle events.
3. Add CPU-only contract tests for waiting/running/scheduled/preempted/finished request states.
4. Port workload generators and analyzers.
5. Audit Model Runner V2 instrumentation against v0.28 async execution semantics.
6. Only then reconsider any waiting-bypass policy code.

## Validation gates without a GPU

### Gate A: static/import

- import trace module
- lint/format
- schema serialization tests

### Gate B: scheduler contracts

Use scheduler/unit-test fixtures or mocks to verify that trace events agree with `SchedulerOutput` and request state transitions.

### Gate C: analyzer compatibility

Feed synthetic JSONL traces into existing analyzers and verify deterministic CSV/summary output.

### Gate D: GPU validation (deferred)

Real model load, CUDA execution, attention backend behavior, CUDA Graphs, performance, and profiler alignment remain unvalidated until GPU access is available.

## First implementation milestone

The first v0.28 milestone is complete when a CPU-only test can call the relevant scheduler path and emit a versioned JSONL event for a controlled request state transition without importing or touching GPU execution code.
