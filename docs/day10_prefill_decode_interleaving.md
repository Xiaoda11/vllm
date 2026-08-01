# Day 10: Prefill/Decode Interleaving

## Engineering question

When request A is already decoding and request B arrives with an 8K or 16K
prompt, how does the v0.26 unified Scheduler divide the global per-step token
budget, and what latency tradeoff is visible for A and B?

## Version and execution path

- Source lane: v0.26.0-based `exp/mrv2-scheduler-trace`.
- Environment: `/home/xiaoda/vllm-lab/.venv-v026`.
- Intended measured path: Path A, MRV2 with
  `VLLM_WSL2_ENABLE_PIN_MEMORY=1` and `VLLM_USE_V2_MODEL_RUNNER=1`.
- Model: `/home/xiaoda/vllm-lab/models/Qwen2.5-0.5B-Instruct`, FP16.
- Measured commit: `7c3664f0ce` (`bench: add decode prefill timing matrix`).
- Measured runner/backend: MRV2 and `TRITON_ATTN`; the trace reports one
  24-layer `FullAttentionSpec` group using NHD layout.

## Pre-run hypotheses

1. Once A is decoding, its next output token consumes one token from the
   global `max_num_batched_tokens` budget before B receives the remaining
   budget for a partial prefill.
2. A mixed step therefore schedules one token for A and at most `budget - 1`
   tokens for B, subject to the Scheduler's other limits and KV allocation.
3. A larger token budget should reduce the number of partial-prefill steps and
   tend to reduce B's TTFT, while the larger mixed batch may increase A's ITL.
   This is only a prediction; a single cold run is insufficient for a stable
   performance claim.
4. B=16K should require more partial-prefill steps than B=8K at the same
   budget, extending the interval in which A decode and B prefill overlap.

## Controlled workload

- A: 1024 prompt tokens, 512 output tokens, arrival at 0 seconds.
- B: 8192 or 16384 prompt tokens, 32 output tokens, nominal arrival at 1.0
  seconds.
- Global token budget: 2048, 4096, or 8192.
- Concurrency: 2; chunked prefill enabled; prefix caching disabled; eager mode.
- Matrix size: two B prompt lengths by three budgets, for six measured runs.

The original 0.25-second delay failed calibration: B was submitted at
0.250434 seconds, before A's first token at 0.338839 seconds. The corrected
1.0-second delay is still only a workload input, not proof of interleaving. A run is
valid for the matrix only if the measured artifacts show that A emitted a token
before B was submitted and at least one Scheduler step contains A decode plus B
prefill.

## Observables

- Scheduler JSONL/CSV: per-request scheduled tokens, initial/remaining budget,
  running/waiting order, KV allocation, and preemption.
- MRV2 JSONL: execution order, persistent row, `idx_mapping`,
  `query_start_loc`, and input tensor shapes.
- Request timing: A TPOT/E2E and B TTFT/E2E.
- Token timing: CPU-observed streaming output event time and exact
  single-token ITL where adjacent chunks each contain one token.
- Batch classes: A decode-only, A-decode/B-prefill mixed, B prefill-only, and
  decode-only after B's prefill.

The traces establish batch composition and shape. They do not establish GPU
kernel time or hardware causality; those claims require the later Nsight work.

## Decision criteria

A run passes workload validation when:

1. both requests finish with their requested output-token counts;
2. A's first-token time precedes B's submitted time;
3. at least one step schedules A with one token and B with prompt tokens;
4. scheduled tokens never exceed the recorded global budget;
5. Scheduler scheduled-token totals agree with the MRV2 input shapes;
6. bundled streaming chunks are not mislabeled as exact single-token ITL.

## Artifacts and validation

The six formal run IDs are:

- `day10-s5-b8k-budget{2048,4096,8192}-20260801`
- `day10-s5-b16k-budget{2048,4096,8192}-20260801`

Each directory under `/home/xiaoda/vllm-lab/outputs` contains
`run_metadata.json`, `startup.log`, `request_timing.csv`, `token_timing.csv`,
both JSONL traces, and the flattened Scheduler CSV. The combined analysis is
under `/home/xiaoda/vllm-lab/outputs/day10-matrix-analysis-20260801`.

All six runs passed the predeclared checks:

- both requests produced their exact requested output-token count;
- A emitted its first token before B was submitted;
- every run contained A-decode/B-prefill mixed steps;
- no step exceeded its global token budget;
- every actual forward step matched Scheduler scheduled tokens against MRV2
  `input_ids.shape[0]`, batch token count, and `query_start_loc[-1]`;
- there were zero preemptions and zero allocation failures.

## Measured Scheduler behavior

In every full mixed-prefill step, A consumed one token and B consumed the
remaining `budget - 1` tokens. The final prompt-tail step used only the tokens
needed to reach B's exact prompt length.

| B prompt | Budget | Full mixed shape `[A,B]` | Tail shape | Mixed prefill steps |
|---:|---:|---:|---:|---:|
| 8K | 2048 | `[1,2047]` × 4 | `[1,4]` | 5 |
| 8K | 4096 | `[1,4095]` × 2 | `[1,2]` | 3 |
| 8K | 8192 | `[1,8191]` × 1 | `[1,1]` | 2 |
| 16K | 2048 | `[1,2047]` × 8 | `[1,8]` | 9 |
| 16K | 4096 | `[1,4095]` × 4 | `[1,4]` | 5 |
| 16K | 8192 | `[1,8191]` × 2 | `[1,2]` | 3 |

For a full 4096-budget step, for example, the corresponding MRV2 evidence is
`num_scheduled_tokens=[1,4095]`, `query_start_loc=[0,1,4096]`, and
`input_ids.shape=[4096]`. This closes the step-level chain from the Scheduler's
global budget to the MRV2 input batch without reading a GPU tensor.

## Measured latency and throughput

These are one cold-start run per matrix point under eager mode. They describe
the observed runs; they are not a stable performance benchmark.

| B prompt | Budget | A TPOT ms | B TTFT ms | A ITL during B prefill mean / P95 ms | Output token/s |
|---:|---:|---:|---:|---:|---:|
| 8K | 2048 | 34.281 | 5336.749 | 764.127 / 2067.815 | 30.719 |
| 8K | 4096 | 34.808 | 5327.839 | 892.135 / 3245.029 | 30.108 |
| 8K | 8192 | 35.204 | 5341.096 | 1339.300 / 4488.222 | 29.814 |
| 16K | 2048 | 64.431 | 20718.605 | 1884.249 / 4472.850 | 16.380 |
| 16K | 4096 | 64.643 | 20670.714 | 2586.731 / 8024.546 | 16.331 |
| 16K | 8192 | 64.884 | 20689.070 | 4142.045 / 13304.196 | 16.267 |

Outside B's prefill phase, A's mean ITL remained close to 24–25 ms in every
run: before B arrived, during B's 32-token decode, and after B finished. During
B's prefill, the mean and P95 ITL rose sharply. Increasing the budget reduced
the number of partial-prefill mixed steps but made individual stalls longer.

B's TTFT did not improve materially with a larger budget in this matrix: the
8K range was 13.257 ms across the three runs, and the 16K range was 47.891 ms.
The observation here is a change in stall granularity, not a demonstrated TTFT
gain. Average A TPOT also hides the multi-second Prefill-phase tail stalls, so
ITL segmentation is the more informative metric for this workload.

The reported throughput is completed output tokens divided by workload
makespan. It is not input throughput and should not be compared across B=8K
and B=16K as if the work were identical.

## Batch and kernel evidence boundary

The trace directly establishes four batch phases: A decode-only,
A-decode/B-prefill mixed, two-request decode, and A decode-only after B
finishes. All attention metadata entries identify `TritonAttentionMetadata`,
and the startup log records JIT of `kernel_unified_attention` and
`reduce_segments`.

This evidence does not contain per-kernel duration. It therefore supports the
batch-shape and backend statements above, but not a claim that a particular
kernel caused an ITL spike. Kernel timing and hardware causality remain Day
16/17 Nsight questions.

## Source facts, measurements, and inference

- Source/trace fact: v0.26's unified Scheduler charged A's decode token and B's
  prefill chunk against one global budget in every mixed step.
- Measurement: larger budgets produced fewer, larger mixed-prefill batches;
  B TTFT stayed nearly flat within each prompt length while A's Prefill-phase
  ITL tail increased.
- Inference to test later: for this model/backend, total prefill work rather
  than the number of partial-prefill steps dominated B TTFT. Nsight or repeated
  warm runs are required before making a causal performance statement.

## Unresolved question

Would repeated warm runs preserve the nearly flat B TTFT and increasing ITL
tail trend, or are parts of the single-run differences caused by cold Triton
JIT, temperature, or normal run-to-run variance?

## 30-second interview explanation

I constructed a vLLM v0.26 MRV2 workload where a 1K/512-token request was
already decoding when an 8K or 16K Prefill arrived. The Scheduler trace showed
that each mixed step first charged one decode token, then gave the remaining
global budget to Prefill: for a 4096 budget the MRV2 batch was `[1,4095]` with
`query_start_loc=[0,1,4096]`. Larger budgets reduced the number of Prefill
chunks but created fewer, longer ITL stalls for the decoding request, while B's
single-run TTFT stayed almost flat. This shows why average TPOT alone can hide
tail latency and why the next policy should be selected from ITL evidence.
