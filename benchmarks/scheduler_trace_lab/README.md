# vLLM Scheduler Trace Lab

This lab turns vLLM V1 scheduling questions into reproducible workloads,
per-step traces, small policy changes, and benchmark evidence.

The four-week project is intentionally frozen at vLLM 0.15.0 commit
`f176443446f659dbab5315e056e605d8984fd976`. Results are not mixed with newer
vLLM versions.

## Milestones

1. Establish a stable long-prefill baseline.
2. Generate controlled single- and dual-request workloads.
3. Trace request state, scheduled tokens, token budget, KV usage, and
   preemption at every Scheduler step.
4. Follow Scheduler output through KV Cache Manager and GPU Model Runner.
5. Implement one opt-in scheduling policy change.
6. Compare latency, throughput, fairness, KV usage, and GPU execution.

## Current status

- [x] [Day 1: single-request long-prefill baseline](day1)
- [ ] Day 2: controlled workload generator
- [ ] Day 3: Scheduler call-chain map and S3 prediction
- [ ] Day 4: opt-in Scheduler JSONL instrumentation
- [ ] Day 5: 8K + 16K scheduling report

Every experiment records a hypothesis, observables, decision criteria,
reproduction command, raw result, conclusion, and one unresolved question.
