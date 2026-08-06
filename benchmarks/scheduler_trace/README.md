# Scheduler Trace Workloads

This directory contains deterministic inputs for the v0.26 Scheduler Trace
Lab. The generator submits pre-tokenized prompts through `AsyncLLM`, so prompt
lengths are exact and request arrival times are controlled independently.

## Validate without a GPU

```bash
V026_PYTHON=/home/xiaoda/vllm-lab/.venv-v026/bin/python
MODEL=/home/xiaoda/vllm-lab/models/Qwen2.5-0.5B-Instruct

"${V026_PYTHON}" scripts/lab_v026_workload.py \
  --config benchmarks/scheduler_trace/configs/s3_dual_simultaneous.json \
  --model "${MODEL}" \
  --validate-only
```

## Run the canonical S3 workload

The WSL pin-memory opt-in is required for the measured MRV2 Path A.

```bash
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export VLLM_USE_V2_MODEL_RUNNER=1

/home/xiaoda/vllm-lab/.venv-v026/bin/python \
  scripts/lab_v026_workload.py \
  --config benchmarks/scheduler_trace/configs/s3_dual_simultaneous.json \
  --model /home/xiaoda/vllm-lab/models/Qwen2.5-0.5B-Instruct \
  --scheduler-trace
```

Each run creates a unique directory under
`/home/xiaoda/vllm-lab/outputs/`. It contains:

- `run_metadata.json`: commit, exact command, environment, resolved scenario,
  prompt digests, and request results.
- `request_timing.csv`: planned/submitted/first-token/finished times, TTFT,
  E2E latency, exact token counts, and errors.
- `token_timing.csv`: opt-in CPU-observed streaming output events for ITL
  analysis; enable it with `--token-timing`.
- `scheduler_trace.jsonl`: opt-in, one CPU-side Scheduler decision per line.
- `scheduler_trace.mrv2.jsonl`: opt-in, MRV2 execution order and persistent
  request rows. Warmup records use `step_id=0`.

Existing run directories are never overwritten.

Flatten and join both trace streams with:

```bash
/home/xiaoda/vllm-lab/.venv-v026/bin/python \
  scripts/lab_scheduler_trace_to_csv.py \
  --scheduler-trace /home/xiaoda/vllm-lab/outputs/RUN_ID/scheduler_trace.jsonl \
  --model-runner-trace \
    /home/xiaoda/vllm-lab/outputs/RUN_ID/scheduler_trace.mrv2.jsonl \
  --output /home/xiaoda/vllm-lab/outputs/RUN_ID/scheduler_trace.csv
```

Tracing is disabled unless `--scheduler-trace` is passed. The implementation
copies Scheduler and MRV2 CPU state only; it does not read a GPU tensor or add
a CUDA synchronization.

Token timing is disabled unless `--token-timing` is passed. A
`single_token_itl_s` value is emitted only when both adjacent streaming chunks
contain exactly one token, so bundled chunks are not mislabeled as exact ITL.

## Run the Day 5 dual-prefill matrix

Day 5 crosses three global per-step token budgets (`2048`, `4096`, and `8192`)
with three arrival orders:

| Arrival order | Config |
|---|---|
| A and B simultaneous | `s3_dual_simultaneous.json` |
| A before B | `s4_staggered_prefill.json` |
| B before A | `d5_dual_b_first.json` |

Pass `--token-budget 2048`, `--token-budget 4096`, or
`--token-budget 8192` together with `--scheduler-trace`. The resolved budget is
recorded under `effective_scenario.engine.max_num_batched_tokens` in
`run_metadata.json`; the source config remains unchanged.

The measured matrix and its source/measurement boundaries are documented in
`docs/day5_dual_prefill.md`.

## Run the Day 6 KV allocation workloads

Use S1 with `--token-budget 2048` for the single-request block-table growth
baseline, and S3 for two requests without a shared prefix.

Run the same S6 token inputs with Prefix Cache disabled and enabled:

```bash
/home/xiaoda/vllm-lab/.venv-v026/bin/python \
  scripts/lab_v026_workload.py \
  --config benchmarks/scheduler_trace/configs/s6_shared_prefix.json \
  --model /home/xiaoda/vllm-lab/models/Qwen2.5-0.5B-Instruct \
  --token-budget 2048 \
  --prefix-caching off \
  --scheduler-trace

/home/xiaoda/vllm-lab/.venv-v026/bin/python \
  scripts/lab_v026_workload.py \
  --config benchmarks/scheduler_trace/configs/s6_shared_prefix.json \
  --model /home/xiaoda/vllm-lab/models/Qwen2.5-0.5B-Instruct \
  --token-budget 2048 \
  --prefix-caching on \
  --scheduler-trace
```

The flattened CSV includes the KV block size, Prefix Cache hit blocks, and
block-table sizes before and after each step. `block_table_blocks_added`
counts blocks newly attached to that request's table. With Prefix Cache
enabled, it includes reused blocks and must not be reported as the number of
new physical allocations.

The measured allocation chain and Prefix Cache comparison are documented in
`docs/day6_kv_cache_allocation.md`.

## Run the Day 7 preemption workload

Day 7 keeps the requests, token budget, and scheduler settings fixed while
changing only the KV block count. The scenario disables
`scheduler_reserve_full_isl` in both groups so the Scheduler can admit both
requests and expose the running-request preemption path.

Run the default-capacity baseline:

```bash
/home/xiaoda/vllm-lab/.venv-v026/bin/python \
  scripts/lab_v026_workload.py \
  --config benchmarks/scheduler_trace/configs/d7_preemption_pressure.json \
  --model /home/xiaoda/vllm-lab/models/Qwen2.5-0.5B-Instruct \
  --scheduler-trace
```

Then restrict the KV pool to 1450 blocks:

```bash
/home/xiaoda/vllm-lab/.venv-v026/bin/python \
  scripts/lab_v026_workload.py \
  --config benchmarks/scheduler_trace/configs/d7_preemption_pressure.json \
  --model /home/xiaoda/vllm-lab/models/Qwen2.5-0.5B-Instruct \
  --num-gpu-blocks-override 1450 \
  --scheduler-trace
```

The trace records waiting and running allocation failures. For running
failures, it also records the victim and its computed-token count immediately
before `_preempt_request()` resets that progress to zero. `request_timing.csv`
includes TPOT as `(finished - first token) / (output tokens - 1)`.

With the v0.26 default `scheduler_reserve_full_isl=true`, the pressured B
request remains waiting until its full input can fit. That admission guard is
a separate behavior and must not be reported as a measured preemption.

The measured failure, preemption, recomputation, and request-timing evidence is
documented in `docs/day7_preemption.md`.

## Run the Day 8 Scheduler-to-MRV2 trace

Day 8 reuses canonical S3 at a 2048-token budget and extends the opt-in MRV2
JSONL with input tensor metadata:

```bash
VLLM_WSL2_ENABLE_PIN_MEMORY=1 VLLM_USE_V2_MODEL_RUNNER=1 \
  /home/xiaoda/vllm-lab/.venv-v026/bin/python \
  scripts/lab_v026_workload.py \
  --config benchmarks/scheduler_trace/configs/s3_dual_simultaneous.json \
  --model /home/xiaoda/vllm-lab/models/Qwen2.5-0.5B-Instruct \
  --token-budget 2048 \
  --scheduler-trace
```

`model_runner_inputs` records CPU arrays already used by input preparation and
tensor shape/dtype/device metadata. It never reads GPU tensor contents. The
measured persistent-row, index-mapping, input-shape, block-table, slot-mapping,
and attention-metadata chain is documented in
`docs/scheduler_to_model_runner.md`.

## Scale prompts for the RTX 2060

If canonical 8K/16K requests do not fit reliably, preserve the 1:2 ratio with:

```bash
/home/xiaoda/vllm-lab/.venv-v026/bin/python \
  scripts/lab_v026_workload.py \
  --config benchmarks/scheduler_trace/configs/s3_dual_simultaneous.json \
  --model /home/xiaoda/vllm-lab/models/Qwen2.5-0.5B-Instruct \
  --prompt-scale 0.5
```

This changes S3 to 4K/8K. The metadata and CSV retain both canonical and
effective prompt lengths. A scaled GPU run must not be reported as an 8K/16K
measurement.

## Run the Day 10 decode/prefill matrix

Day 10 crosses B prompt lengths (`8192` and `16384`) with the three global
per-step token budgets (`2048`, `4096`, and `8192`). A remains fixed at a
1024-token prompt and 512-token output. Use
`s5_decode_then_prefill_8k.json` for B=8K and
`s5_decode_then_prefill.json` for B=16K, together with
`--scheduler-trace --token-timing`.

The configured 1.0-second arrival delay is not itself proof that A is already
decoding. Each measured run must verify from request timing and Scheduler trace
that A emitted a token before B was submitted and that mixed A-decode/B-prefill
steps actually occurred.

Validate and summarize the completed six-run matrix with
`scripts/lab_day10_analyze.py`. It writes a matrix summary CSV, segmented ITL
CSV, and dependency-free SVG tradeoff chart. The measured 2026-08-01 matrix and
its evidence boundaries are documented in
`docs/day10_prefill_decode_interleaving.md`.

## Run the Day 13 waiting head-of-line baseline

Day 13 ends the Prefill-quantum code path after Gate A/B and uses a three-request
KV-pressure workload to test waiting admission head-of-line blocking. Keep the
default full-input reservation enabled and restrict the KV pool to 1450 blocks:

```bash
VLLM_WSL2_ENABLE_PIN_MEMORY=1 VLLM_USE_V2_MODEL_RUNNER=1 \
  /home/xiaoda/vllm-lab/.venv-v026/bin/python \
  scripts/lab_v026_workload.py \
  --config benchmarks/scheduler_trace/configs/d13_waiting_hol_blocking.json \
  --model /home/xiaoda/vllm-lab/models/Qwen2.5-0.5B-Instruct \
  --num-gpu-blocks-override 1450 \
  --scheduler-trace --token-timing
```

The baseline must prove from the same Scheduler step that B fails full-ISL
allocation while C remains behind it even though C's full input would fit in the
reported free blocks. The measured Gate A/B decision and the policy hypothesis
are documented in `docs/day13_prefill_quantum_gate.md`.

Run the opt-in policy with the same workload and capacity by adding:

```bash
--waiting-bypass on
```

The corresponding core CLI flag is `--scheduler-allow-waiting-bypass`. It is
disabled by default. Baseline/Modified trace evidence, implementation details,
and boundary risks are documented in `docs/day13_waiting_hol_policy.md`.

## Run the Day 14 waiting HOL PR Gate

Repeat the D13 workload in interleaved Baseline/Modified order at least three
times per mode, then validate the six directories with
`scripts/lab_day14_analyze.py`. The analyzer rejects runs without ordered B/C
submission, a same-step HOL witness, successful requests, or consistent MRV2
input shapes.

Use `d14b_waiting_hol_burst.json` for the fairness counterexample. It places
eight 1K/512 short requests behind B while A is decoding. Run one descriptive
Baseline/Modified pair with the same 1450-block override and validate it with
`scripts/lab_day14_burst_analyze.py`. This pair is intended to expose long-head
delay, not to provide a stable performance percentage.

The measured repeated benefit, burst tradeoff, and upstream PR Gate are
documented in `docs/day14_waiting_hol_benchmark.md`.

For the bounded one-admission variant, compare both lifetime extremes. Re-run
`d14b_waiting_hol_burst.json` for the long-lived case and use
`d15_waiting_hol_short_burst.json` for the 32-output-token case. Analyze D15 by
passing `--scenario-id D15` to `scripts/lab_day14_burst_analyze.py`. The policy
audit and current no-PR decision are documented in
`docs/day15_waiting_bypass_pr_audit.md`.

For Day 16, run a one-kernel CUDA smoke before interpreting profiler output.
The local Nsight Systems path recorded CUDA API calls but no kernel activity,
so the controlled fallback uses `--torch-profile` with a bounded worker-step
window. `scripts/lab_day16_profile_analyze.py` aligns execution annotations
with Scheduler token totals and rejects missing kernel activity. Commands,
artifacts, results, and the evidence boundary are in
`docs/day16_nsys_systems_gate.md`; personal review questions are in
`docs/day16_profiling_acceptance.md`.

Day 17 initially stopped at `ERR_NVGPUCTRPERM`. After explicit user authorization
and enabling NVIDIA performance-counter access, the minimal CUDA matmul gate and
a targeted real-vLLM Prefill GEMM collection both succeeded. The measured SM,
memory, occupancy, and warp-stall counters—and the important limitation that the
target launch is not yet uniquely aligned to a Scheduler step—are documented in
`docs/day17_ncu_gate.md`.

The end-to-end engineering narrative, project overview, and final PR Gate are
collected in:

- `docs/scheduler_trace_lab_final_report.md`;
- `docs/scheduler_trace_lab_project_overview.md`;
- `docs/day20_project_review_and_pr_gate.md`.

## Scenario intent

| Scenario | Requests | Controlled question |
|---|---|---|
| S1 | A=8K | Single long prefill baseline |
| S2 | B=16K | Longer single prefill baseline |
| S3 | A=8K, B=16K at `t=0` | Dual-prefill scheduling |
| S4 | A=8K, then B=16K | Waiting/running transition |
| S5 | A decodes, then B=16K | Decode/prefill overlap |
| S6 | Two 8K prompts share 6K | Prefix-cache reuse |

Request-level timing cannot establish scheduler step order, token-budget
allocation, KV-block allocation, or preemption. Those are Day 4 trace fields.
