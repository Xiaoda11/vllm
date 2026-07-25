# Scheduler Trace Lab — Day 1

Day 1 establishes a reproducible single-request long-prefill baseline before
Scheduler instrumentation or policy changes are introduced.

## Question

Can Qwen2.5-0.5B-Instruct execute 8K and 16K prompts reliably on an RTX 2060
Laptop GPU, and what are the baseline TTFT, TPOT, memory, temperature, and power
characteristics?

## Fixed configuration

- vLLM 0.15.0 at commit `f176443446f659dbab5315e056e605d8984fd976`
- Qwen2.5-0.5B-Instruct, FP16
- FlashInfer attention backend
- `max_model_len=32768`
- `max_num_batched_tokens=4096`
- `max_num_seqs=2`
- `gpu_memory_utilization=0.75`
- eager execution
- Chunked Prefill enabled
- Prefix Cache disabled

The workload contains one excluded cold 2K request, followed by three
steady-state repetitions at 2K, 4K, 8K, and 16K prompt tokens. Every request
generates exactly 16 output tokens.

Prompt token IDs are constructed to an exact length and passed directly to
vLLM. This avoids confusing text length with tokenizer token length.

## Run

From the repository root:

```bash
source /home/xiaoda/vllm-lab/.venv/bin/activate

benchmarks/scheduler_trace_lab/day1/run_baseline.sh
```

To store outputs outside the repository:

```bash
DAY1_OUTPUT_DIR=/path/to/day1-results \
  benchmarks/scheduler_trace_lab/day1/run_baseline.sh
```

For a quick validation run:

```bash
DAY1_OUTPUT_DIR=/tmp/vllm-day1-smoke \
  benchmarks/scheduler_trace_lab/day1/run_baseline.sh \
  --prompt-lengths 2048 --steady-repeats 1
```

The run produces:

- `environment.json`: source, runtime, GPU, and engine configuration
- `baseline.csv`: normalized per-request measurements
- `baseline.jsonl`: lossless per-request records
- `summary.csv`: grouped steady-state statistics

The checked-in [`artifacts/2026-07-24`](artifacts/2026-07-24) directory contains
the first verified run's summary and concise report. Large logs and duplicate
raw formats stay outside Git.

## Metrics

- TTFT is `RequestStateStats.first_token_latency`.
- TPOT is `(last_token_ts - first_token_ts) / (output_tokens - 1)`.
- E2E is wall time around one blocking `LLM.generate` call.
- Input tok/s is `actual_input_tokens / TTFT`, a prefill throughput proxy.
- GPU metrics are sampled with NVML every 20 ms.

NVML memory is device-level usage and can include WSLg or desktop processes; it
is not vLLM-exclusive allocation.

## Interpretation boundary

This baseline verifies that the configured prompt lengths run and records their
latency/resource scale. It does not provide per-step Scheduler evidence.

For a single 16K prompt and a 4096-token budget, four 4096-token prefill steps
are the working prediction. Day 4 JSONL Scheduler tracing must verify that
prediction; it must not be reported as an observed trace yet.
