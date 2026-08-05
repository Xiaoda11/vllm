# Day 16 personal acceptance and interview Q&A

## Acceptance rule

Pass when the learner can answer questions 1-8 without reading the report and
can deliver the 30-second explanation. Exact kernel symbols and exact decimal
values are not memorization requirements. The required anchors are:

- strict: 20 single Decode steps;
- bounded: 17 single Decode + 1 mixed Prefill/Decode + 2 dual Decode;
- mixed step: A Decode 1 + C Prefill 1024 = 1025 scheduled tokens;
- same attention call count, changed shape and duration;
- Decode is GEMV-heavy, Prefill introduces GEMM;
- profiler evidence explains work placement but is not a stable benchmark.

## Questions

1. Why were the first two Nsight Systems reports rejected as GPU evidence even
   though they contained about 111k `cudaLaunchKernel` API calls?
2. How did the minimal matmul smoke test narrow the Nsight failure?
3. Why is the PyTorch Profiler fallback acceptable, and what can it not prove?
4. How do we know the profile covers Scheduler steps 60-79 and not merely an
   approximate wall-clock interval?
5. Derive the bounded mixed step's 1025 scheduled tokens from the execution
   annotation.
6. What is the execution-composition difference between strict and bounded?
7. Why are there 480 unified-attention calls in both profiles?
8. Why does bounded introduce GEMM while strict is GEMV-heavy?
9. Does 299 ms versus 162 ms prove bounded scheduling is 84% slower? Why not?
10. Why is “B first scheduled step stayed at 517” insufficient as a complete
    fairness/safety claim?
11. Why did C first schedule at step 77 here but step 68 in the earlier run?
12. What additional gate must pass before using Nsight Compute metrics?

## Reference answers

1. API launch records prove the CPU issued launches, not that Nsight recorded
   kernel start/end activity. `cuda_gpu_kern_sum` had no kernel table, so GPU
   gaps, overlap, names, and durations were unobserved.
2. A single CUDA matmul failed the same way, excluding vLLM's multiprocessing
   and complex workload as the sole cause. Explicit software trace also failed,
   narrowing the problem to the local Nsight/WSL/host path.
3. It is a vLLM-supported profiler and the smoke test showed real kernel names
   and CUDA time. It has medium/high overhead, so it can establish observed
   execution shape and kernel composition, not stable end-to-end performance.
4. The profiler was configured with delay 60 and max 20. The analyzer pairs the
   20 execution annotations with Scheduler steps 60-79 and rejects any mismatch
   between annotation token totals and Scheduler scheduled tokens.
5. `context_1(sq1024...)` means one context/Prefill request with 1024 query
   tokens. `generation_1(sq1...)` means one Decode request with one query token.
   Therefore the step executes `1024 + 1 = 1025` tokens.
6. Strict is 20 single Decode steps. Bounded is 17 single Decode, one mixed
   Prefill/Decode, and two multi-Decode steps.
7. Qwen has 24 attention layers and both windows contain 20 forward steps, so
   `24 × 20 = 480`. The batch shape changes duration, not layer invocation
   count.
8. Single-token Decode performs matrix-vector-like linear work. Processing a
   1024-token Prefill exposes larger matrices and invokes Turing FP16/CUTLASS
   GEMMs.
9. No. Bounded performs C work that strict postpones, and the profiler adds
   overhead. The numbers explain the cost/location of the extra mixed work;
   repeated unprofiled runs decide policy performance.
10. Day 15's long-lived burst kept B's first step unchanged but introduced
    three later preemptions and regressed fairness/makespan. Admission can alter
    later KV pressure after B starts.
11. Arrivals are wall-clock controlled. Profiling overhead changes how many
    Scheduler steps occur before C's 7.01-second arrival. Same-run trace
    alignment is authoritative.
12. A one-kernel NCU smoke must succeed with authorized GPU performance-counter
    access. The current run fails with `ERR_NVGPUCTRPERM`; do not collect or
    interpret vLLM NCU metrics before that gate passes.

## Thirty-second answer

I aligned a 20-step PyTorch GPU profile with Scheduler trace. Strict admission
ran 20 single-token Decode steps. Bounded admission inserted one step containing
A's one Decode token plus C's 1024-token Prefill, followed by two dual-Decode
steps. Both profiles still called attention 480 times because the model has 24
layers across 20 steps, but the mixed shape introduced Tensor Core GEMMs and
made that step much longer. So the policy changes when and with what shape
existing kernels execute; it does not modify kernel code, and profiler timing
is not used as a stable throughput benchmark.

## Two-minute answer outline

1. Start with the D13 HOL problem and strict/full-ISL behavior.
2. Explain why Nsight API-only traces failed the kernel-evidence gate.
3. State the bounded PyTorch profile window and Scheduler-token validation.
4. Walk through `1 Decode + 1024 Prefill = 1025` at step 77.
5. Explain GEMV-heavy Decode versus GEMM-heavy Prefill.
6. Separate kernel-composition evidence from repeated policy benchmarks.
7. Close with the Day 15 fairness counterexample and no-PR decision.
