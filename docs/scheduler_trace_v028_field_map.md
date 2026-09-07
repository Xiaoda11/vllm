# v0.28 Scheduler Trace field migration map

This document records the semantic migration of Scheduler Trace from the frozen v0.26 experiment to the v0.28 integration branch.

## Result of the first audit

The core observability boundary survives the upgrade: `Scheduler.schedule()` still owns the request queues, token budgets, KV allocation decisions, preemption decisions, and the final `SchedulerOutput`. The trace should therefore remain scheduler-side and should continue to read CPU-side state only.

The old instrumentation must not be copied mechanically because v0.28 adds a second input budget, speculative-draft slot accounting, KV-delivery-aware preemption, encoder-only support, configurable encoder-cache management, and revised prefill lookahead semantics.

## Field map

| Trace concept | v0.26 source | v0.28 source / decision | Migration status |
| --- | --- | --- | --- |
| step id | `Scheduler.current_step` | `Scheduler.current_step` | stable |
| request id | `Request.request_id` | `Request.request_id` | stable |
| request status | `Request.status.name` | `Request.status.name` | stable |
| prompt tokens | `Request.num_prompt_tokens` | `Request.num_prompt_tokens` | stable |
| output tokens | `len(Request.output_token_ids)` | same | stable |
| computed tokens | `Request.num_computed_tokens` | same | stable |
| in-flight tokens | `Request.num_in_flight_tokens` | same | stable |
| scheduled tokens | `SchedulerOutput.num_scheduled_tokens` | same | stable |
| scheduled total | `SchedulerOutput.total_num_scheduled_tokens` | same | stable |
| running queue | `Scheduler.running` | same | stable |
| waiting queue | `Scheduler.waiting` | same | stable |
| skipped waiting queue | `Scheduler.skipped_waiting` | same | stable |
| KV usage | `KVCacheManager.usage` | same scheduler-side manager | wired, CPU contract covered |
| free KV blocks | block pool free count | block pool free count | wired, CPU contract covered |
| request block ids | KV cache manager lookup | KV cache manager lookup | wired, CPU contract covered |
| preempted ids | `SchedulerOutput.preempted_req_ids` | same output field | stable |
| finished ids | `SchedulerOutput.finished_req_ids` | same output field | stable |
| token budget | `max_num_scheduled_tokens` / remaining budget | same compute-token budget | stable |
| input budget | not traced | `max_num_batched_tokens` with draft-slot accounting | **new v0.28 field** |
| allocation failure | `allocate_slots() -> None` | same decision point | stable hook; payload must include both budgets |
| preemption semantics | `_preempt_request(req, ts)` | `_preempt_request(..., drop_stale_output=requires_kv_delivery)` | **changed** |
| prefix cache hit | waiting-flow local computed tokens | local-hit path remains, connector semantics evolved | audit during waiting-flow port |
| prefill lookahead | Eagle/spec lookahead | `num_prefill_lookahead` + `_reserve_prefill_lookahead()` | **changed** |

## v0.28 schema additions

The scheduler-step event should distinguish two budgets:

- `token_budget`: compute tokens available to the scheduling step.
- `input_budget`: input/draft-slot constrained budget introduced in the newer scheduler path.

This matters because a request can now be blocked even when compute-token budget remains. Recording only the old token budget would make some v0.28 scheduling decisions impossible to explain from the trace.

Preemption events should also record whether stale in-flight output must be dropped (`requires_kv_delivery`), because v0.28 can pass that semantic into `_preempt_request`.

## Port order

Completed in the CPU integration slice:

1. Wired `create_scheduler_trace_writer()` into Scheduler construction and shutdown.
2. Restored before/after queue + request snapshots using v0.28 state.
3. Emitted successful scheduling decisions from `SchedulerOutput`.
4. Added the v0.28 `input_budget` to the live step event.
5. Instrumented running-flow allocation failures and KV-delivery-aware preemption.
6. Instrumented waiting-flow allocation failures and admitted prefix-cache hits.
7. Added isolated CPU contracts for the live Scheduler snapshot/event methods.

The remaining validation step is an end-to-end Scheduler or GPU workload that
exercises the opt-in JSONL path under real model execution.

## Validation boundary

A green CPU CI now proves JSONL/schema/writer behavior plus the live Scheduler
snapshot/event methods against lightweight v0.28 state doubles. Until a real
Scheduler fixture or GPU workload runs, end-to-end scheduling and trace overhead
remain migration code under validation, not experimental evidence.
