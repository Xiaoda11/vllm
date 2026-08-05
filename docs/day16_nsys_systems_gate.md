# Day 16: profiling gate and Scheduler-to-kernel evidence

## Decision

Day 16 passed through a documented fallback path.

- Nsight Systems 2025.6.3 recorded CUDA API calls but no GPU kernel activity
  for vLLM or a minimal PyTorch matmul. Those reports are not GPU-timeline
  evidence.
- Explicit `--trace=cuda-sw` produced the same result.
- The vLLM-supported PyTorch Profiler did record CUDA kernels. A bounded
  step-60-to-79 window was collected for strict and bounded waiting admission.
- Scheduler traces and profiler annotations agree on scheduled token counts for
  all 40 profiled steps.

The result answers the Day 16 controlled question: bounded admission changes
the execution shape and kernel mix by placing C's Prefill into A's Decode
window. It does not modify an individual kernel implementation.

The upstream PR path remains paused. No external comment or PR update was made.

## Source facts and local gates

NVIDIA's CUDA on WSL guide lists Nsight Systems CLI and CUPTI trace support for
Volta and later GPUs, with additional Windows/driver requirements for
profilers. The current GPU is Turing and the Windows driver is R596, but the
exact Windows build was not recovered from the WSL session.

NVIDIA's Nsight Systems release notes state that virtualized environments may
not support CUDA hardware trace and may fall back to legacy software trace. The
local report diagnostics confirmed that fallback. However, both hardware and
explicit software modes still omitted GPU kernel activity locally.

Official references:

- <https://docs.nvidia.com/cuda/wsl-user-guide/index.html>
- <https://docs.nvidia.com/nsight-systems/ReleaseNotes/index.html#cuda-trace-issues>
- `docs/contributing/profiling.md` in this repository

A minimal Nsight Compute run reached the GPU but failed with
`ERR_NVGPUCTRPERM`. Enabling system-wide performance-counter access was not
authorized, so Day 17 remains behind its own one-kernel permission gate.

## Controlled question

> Does bounded waiting admission change only Scheduler/request ordering, or
> does it also change the GPU work composition around C's admission?

Hypothesis:

- Strict admission keeps the profile window as A-only Decode.
- Bounded admission inserts C's 1024-token Prefill and then runs A/C Decode
  together.
- The mixed step introduces Prefill-oriented GEMM work; this is a batch/shape
  change, not a new kernel implementation.

Decision criteria:

1. Both runs use the same commit, model, backend, KV capacity, token budget,
   eager mode, and profiler window.
2. Every profiler execution annotation matches the corresponding Scheduler
   step's scheduled-token total.
3. Both traces contain actual CUDA kernel events.
4. Conclusions distinguish observed shape/kernel composition from stable
   performance claims.

## Implementation

Commit `1e2f11bf54` adds a lab-only bounded PyTorch profiler mode:

```text
--torch-profile
--torch-profile-delay-iterations 60
--torch-profile-max-iterations 20
```

The effective profiler config records shapes, disables stacks/memory/FLOPs,
ignores the AsyncLLM frontend, and uses detailed worker annotations. The
profile starts after 60 worker iterations and records 20 steps, avoiding model
load and initial Triton JIT.

The automated analyzer is `scripts/lab_day16_profile_analyze.py`. It validates
the D13 HOL witness, request completion, no preemption, MRV2 shape invariants,
the exact profile window, Scheduler/profile token agreement, and non-empty
kernel activity. It writes a step CSV and JSON summary.

## Workload and commands

Both runs used MRV2, `TRITON_ATTN`, full-ISL reservation, Prefix Cache off,
eager execution, a 2048-token budget, and a 1450-block KV override. The only
policy variable was waiting bypass off/on.

```bash
VLLM_WSL2_ENABLE_PIN_MEMORY=1 \
VLLM_USE_V2_MODEL_RUNNER=1 \
/home/xiaoda/vllm-lab/.venv-v026/bin/python \
  scripts/lab_v026_workload.py \
  --config benchmarks/scheduler_trace/configs/d13_waiting_hol_blocking.json \
  --model /home/xiaoda/vllm-lab/models/Qwen2.5-0.5B-Instruct \
  --num-gpu-blocks-override 1450 \
  --waiting-bypass off \
  --scheduler-trace --token-timing \
  --torch-profile \
  --torch-profile-delay-iterations 60 \
  --torch-profile-max-iterations 20 \
  --run-id day16-torch-d13-strict-20260805
```

The bounded command changes only:

```bash
--waiting-bypass on --run-id day16-torch-d13-bounded-20260805
```

## Scheduler-to-execution alignment

| Profile window | Strict | Bounded |
|---|---:|---:|
| Scheduler steps | 60-79 | 60-79 |
| Single Decode steps | 20 | 17 |
| Prefill/Decode mixed steps | 0 | 1 |
| Multi-Decode steps | 0 | 2 |
| Profile/Scheduler token mismatches | 0 | 0 |

The bounded mixed step is Scheduler step 77:

```text
A Decode: 1 token
C Prefill: 1024 tokens
total scheduled: 1025 tokens
annotation: execute_1025_context_1(...sq1024...)_generation_1(...sq1...)
```

Steps 78 and 79 schedule two generation tokens, one each for A and C. Strict
steps 60-79 schedule only one A Decode token per step.

C first appears at step 77 in the profiled bounded run, not step 68 as in the
earlier Nsight run. Arrival is wall-clock controlled, and profiler overhead
changes how many Scheduler iterations occur before 7.01 seconds. This is why
the conclusion uses same-run Scheduler/profile alignment rather than assuming
a fixed step number across tools.

## Observed kernel composition

These values describe one 20-step profiled window per mode. PyTorch Profiler
adds substantial overhead, so they are not throughput benchmark results.

| Metric | Strict | Bounded |
|---|---:|---:|
| CUDA kernel calls | 5,661 | 5,643 |
| Sum of kernel durations | 162.173 ms | 299.080 ms |
| Unified-attention calls | 480 | 480 |
| Unified-attention duration | 71.569 ms | 169.749 ms |
| GEMV duration | 83.282 ms | 70.649 ms |
| GEMM duration | 0 ms | 46.068 ms |
| Longest annotated GPU range | 30.292 ms | 143.275 ms |

Interpretation:

- Both windows execute 24 attention-layer calls per step: `24 × 20 = 480`.
  The call count stays fixed while the bounded mixed step changes query/context
  shape, so attention duration changes.
- Strict is pure single-token Decode and is GEMV-heavy.
- Bounded replaces three single-Decode steps with one 1025-token mixed step and
  two dual-Decode steps. The Prefill introduces Turing FP16/CUTLASS GEMMs.
- The mixed step's GPU range is 143.275 ms; strict's longest step in the same
  window is 30.292 ms.

This does not prove the policy is globally slower. Bounded performs useful C
work that strict intentionally postpones. The profile explains where the work
moved and why A can see an ITL disturbance; stable latency/throughput decisions
still come from the repeated Day 14/15 benchmarks without profiler overhead.

## Request and correctness checks

Both profiled runs:

- completed A/B/C with exact requested token counts;
- contained a same-step B-blocked/C-fits HOL witness;
- had zero preemptions;
- had zero MRV2 input-shape mismatches;
- kept B's first scheduled step at 517.

Profiled request timing is intentionally excluded from performance claims. For
example, C TPOT is heavily distorted by profiler flush/overhead.

## Artifacts

```text
/home/xiaoda/vllm-lab/outputs/day16-torch-d13-strict-20260805
/home/xiaoda/vllm-lab/outputs/day16-torch-d13-bounded-20260805
/home/xiaoda/vllm-lab/outputs/day16-torch-analysis-20260805
/home/xiaoda/vllm-lab/outputs/day16-nsys-20260805
```

The two `.pt.trace.json.gz` files can be opened directly in Perfetto. The
Nsight `.nsys-rep` files are retained only as failed-gate evidence.

## Conclusion

Bounded waiting admission changes more than a CPU queue decision. In the
observed window it changes the Model Runner batch from 20 single-token Decode
steps to a window containing one 1024-Prefill-plus-Decode step and two
dual-Decode steps. That shape change replaces part of the GEMV-heavy execution
with GEMM work and lengthens the mixed step. The policy code does not alter
attention/GEMM kernels; it changes when and with which shape existing kernels
run.

## Unresolved question

Can Windows/driver settings make Nsight Systems kernel tracing and Nsight
Compute performance counters available without weakening machine-wide security
more than is acceptable for this learning environment?

## 30-second interview explanation

I aligned a bounded 20-step GPU profile with Scheduler JSONL rather than
profiling the whole server. Strict admission produced 20 single-token Decode
steps. Bounded admission produced 17 single Decode steps, one mixed step with A
Decode plus C's 1024-token Prefill, and two dual-Decode steps. The mixed step
introduced Tensor Core GEMMs and extended the annotated GPU range from a
roughly 30-millisecond Decode maximum to 143 milliseconds. This shows the
scheduler policy changes batch shape and kernel composition, not kernel code.
