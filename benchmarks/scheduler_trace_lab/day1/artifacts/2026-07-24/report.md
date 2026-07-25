# Day 1: Long-Prefill Baseline

Date: 2026-07-24

## Result

The RTX 2060 Laptop GPU completed 2K, 4K, 8K, and 16K single-request
long-prefill workloads. Each length has three steady-state repetitions after a
separate excluded 2K cold request.

- All 13 requests completed successfully with 16 generated tokens.
- Every completion ended with `finish_reason=length`.
- Every request reported `num_cached_tokens=0`.
- No OOM or preemption occurred.
- vLLM selected the explicitly configured FlashInfer attention backend.
- Startup logs confirmed Chunked Prefill with a 4096-token scheduler budget.
- The engine reported 3.4 GiB available KV cache and capacity for 297,296
  tokens.

## Configuration

| Item | Value |
|---|---|
| vLLM | 0.15.0 |
| Commit | `f176443446f659dbab5315e056e605d8984fd976` |
| Model | Qwen2.5-0.5B-Instruct |
| GPU | RTX 2060 Laptop GPU, 6 GiB, SM 7.5 |
| dtype | FP16 |
| Attention backend | FlashInfer |
| `max_model_len` | 32768 |
| `max_num_batched_tokens` | 4096 |
| `max_num_seqs` | 2 |
| `gpu_memory_utilization` | 0.75 |
| Prefix Cache | disabled |
| Execution | eager |

## Steady-state means

| Input tokens | TTFT | TPOT | E2E | Input tok/s |
|---:|---:|---:|---:|---:|
| 2,048 | 0.115632 s | 16.225 ms | 0.361343 s | 17,712 |
| 4,096 | 0.228837 s | 15.723 ms | 0.466970 s | 17,903 |
| 8,192 | 0.511617 s | 15.941 ms | 0.753719 s | 16,012 |
| 16,384 | 1.334912 s | 16.672 ms | 1.587313 s | 12,276 |

The excluded cold 2K request had a TTFT of 0.252873 seconds, approximately
2.19x the 2K steady-state mean.

## Interpretation

TTFT grows with prompt length while TPOT remains near 16 ms. Prefill throughput
is roughly stable from 2K to 4K, then drops at 8K and 16K as long-context
attention cost grows.

For a single 16K request and a 4096-token budget, four 4096-token prefill steps
are the working Scheduler prediction. This result is not a per-step trace.
Scheduler JSONL instrumentation must verify the prediction before it is
reported as observed behavior.
