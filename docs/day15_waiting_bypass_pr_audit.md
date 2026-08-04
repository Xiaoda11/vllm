# Day 15: bounded waiting bypass PR audit

## Decision

The one-admission bound is not ready for an upstream PR. It limits how many
later requests may pass a KV-blocked head, but it does not limit how long the
admitted request retains KV or how much pressure it creates later. The long-
lived burst produced three preemptions and regressed fairness, makespan, and
throughput. The short-lived burst safely improved only C1; median and tail
latency were effectively unchanged.

This is a useful negative scheduling result, not evidence that the feature must
ship. The current patch remains opt-in and should not be proposed as a default.

## Compared policies

Strict admission preserves the waiting policy's order: when B cannot reserve
its full input, the scheduler stops examining younger requests. This is
non-work-conserving, but protects B from KV allocations made by younger work and
keeps FCFS behavior predictable.

Bounded bypass gives each blocked waiting request one opportunity for a later
request to be admitted. It prevents the unbounded burst from Day 14, but the
bound applies only to admission count. Once admitted, C1 is a running request
and may retain KV throughout a long decode.

## Current-commit GPU evidence

All completed pairs used MRV2, `TRITON_ATTN`, full-ISL reservation, a 1450-block
KV override, eager execution, and Prefix Cache disabled.

### Three-request workload

| Metric | Strict | Bounded |
|---|---:|---:|
| B TTFT | 33.240 s | 31.757 s |
| C TTFT | 33.407 s | 0.179 s |
| B first scheduled step | 517 | 517 |
| Preemptions | 0 | 0 |

This confirms the basic mechanism, but a single small C does not test the
admitted request's downstream lifetime.

### Long-lived burst: eight 1K prompt / 512 output requests

| Metric | Strict | Bounded | Change |
|---|---:|---:|---:|
| B TTFT | 31.229 s | 31.863 s | +2.03% |
| C TTFT median | 31.931 s | 32.142 s | +0.66% |
| Makespan | 52.603 s | 53.241 s | +1.21% |
| Output throughput | 88.208 tok/s | 87.150 tok/s | -1.20% |
| TTFT Jain index | 0.930977 | 0.831934 | -10.64% |
| B first scheduled step | 517 | 517 | unchanged |
| C first-step range | 525-557 | 72-554 | C1 only is very early |
| Preemptions | 0 | 3 | regression |

Bounded C1 TTFT was 0.174 seconds, while the other seven C requests remained at
roughly 31.8-33.1 seconds. C1's retained KV contributed to pressure that
preempted C6, C7, and C8. Therefore “B first step is unchanged” is insufficient
as a safety claim: the admission changes later system behavior.

### Short-lived burst: eight 1K prompt / 32 output requests

| Metric | Strict | Bounded | Change |
|---|---:|---:|---:|
| B TTFT | 31.243 s | 31.440 s | +0.63% |
| C1 TTFT | 31.472 s | 0.173 s | -99.45% |
| C TTFT median | 31.938 s | 32.018 s | +0.25% |
| Makespan | 40.693 s | 40.771 s | +0.19% |
| Preemptions | 0 | 0 | unchanged |
| B first scheduled step | 517 | 517 | unchanged |

When C1 finishes before B can enter, the bypass is safe in this run. It still
benefits one arbitrarily positioned request rather than improving the queue's
median or tail. The small aggregate differences are one-run descriptive values,
not stable performance claims.

### Existing alternative: disable full-ISL reservation

The D13 alternative run was manually terminated at scheduler step 166 after 12
B preemptions. B repeatedly reached about 14,329 computed tokens, was preempted,
and restarted. The run is censored and must not be reported as a successful
comparison. It does show that removing full-input reservation is not a safe
substitute under this memory pressure.

## Risks and semantics

- FCFS becomes conditional rather than strict. Exactly one younger request gets
  a large advantage based on queue position.
- The bound is not a starvation or wall-clock bound. A single admitted request
  may decode for a long time, wait on remote KV, or hold substantial KV.
- Later preemption and recomputation can offset the apparent utilization gain.
- Priority scheduling already permits a later higher-priority request to pass
  B, so the simple “one later request” description only holds cleanly for FCFS
  or equal/lower-priority requests.
- A token-count threshold would be hardware- and workload-dependent. Adding it
  as another heuristic does not provide a principled no-delay guarantee.

A true conservative backfill policy would admit C only when it can prove that C
will not delay B. vLLM does not know C's completion time, and reserving future KV
or forcibly preempting bypassed work would be a substantially larger scheduling
design.

## Upstream relevance

Upstream draft PR `vllm-project/vllm#33499` already proposes the unbounded skip
direction. Opening another equivalent PR would be duplicative. More useful
upstream input would be the counterexample: retrying the blocked head on every
step does not prevent younger admitted requests from retaining KV and delaying
or destabilizing later work.

No external comment is posted by this work. The evidence can be offered to that
PR only after deciding that publishing the local benchmark results is desired.

## Reproduction artifacts

```text
/home/xiaoda/vllm-lab/outputs/prgate-d13-strict-r1-20260804
/home/xiaoda/vllm-lab/outputs/prgate-d13-bounded-r1-20260804
/home/xiaoda/vllm-lab/outputs/prgate-d13-partial-r1-20260804
/home/xiaoda/vllm-lab/outputs/prgate-d14b-strict-r1-20260804
/home/xiaoda/vllm-lab/outputs/prgate-d14b-bounded-r1-20260804
/home/xiaoda/vllm-lab/outputs/prgate-d14b-bounded-analysis-20260804
/home/xiaoda/vllm-lab/outputs/prgate-d15-short-strict-r1-20260804
/home/xiaoda/vllm-lab/outputs/prgate-d15-short-bounded-r1-20260804
/home/xiaoda/vllm-lab/outputs/prgate-d15-short-analysis-20260804
```
