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
