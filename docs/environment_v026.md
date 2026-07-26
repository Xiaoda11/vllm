# vLLM v0.26.0 environment and MRV2 gate

Date: 2026-07-26

Run ID: `v026-gate-20260726`

Result: **Path A passed, with the official WSL pinned-memory opt-in enabled**

## Decision

This RTX 2060 Laptop / WSL2 environment can run vLLM v0.26.0 with Model
Runner V2. It does not work with the WSL defaults: `pin_memory=False` makes
UVA unavailable and a minimal `UvaBuffer` raises
`RuntimeError: UVA is not available`.

No patch was applied. v0.26.0 includes the merged
`VLLM_WSL2_ENABLE_PIN_MEMORY` opt-in. With
`VLLM_WSL2_ENABLE_PIN_MEMORY=1`, pinned memory and a minimal UVA buffer both
work, three independent MRV2 engine starts complete generation, and the
OpenAI-compatible server returns successful health and chat-completion
responses.

All subsequent MRV2 GPU experiments on this machine must set:

```bash
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export VLLM_USE_V2_MODEL_RUNNER=1
```

The second variable is explicit evidence control: it prevents a future
silent fallback from being reported as an MRV2 measurement.

## Fixed source and environment

| Item | Measured value |
|---|---|
| Branch | `exp/mrv2-scheduler-trace` |
| Commit/tag | `f2654939e69b4069b13977e9aef3e31d4dcaf051` / `v0.26.0` |
| Environment | `/home/xiaoda/vllm-lab/.venv-v026` |
| Python | 3.12.13 |
| vLLM | 0.26.0 editable source + exact release precompiled binaries |
| PyTorch | 2.11.0+cu129 |
| PyTorch CUDA runtime | 12.9 |
| NVIDIA driver | 596.49 |
| Driver-reported CUDA | 13.2 |
| Local CUDA toolkit (`nvcc`) | 13.2 |
| GPU | NVIDIA GeForce RTX 2060 Laptop, 6 GiB |
| Compute capability | 7.5 |
| WSL kernel | 6.18.33.2-microsoft-standard-WSL2 |
| Model | local Qwen2.5-0.5B-Instruct, FP16 |

The release does not publish a `cu130` wheel asset. The installed native
binaries therefore use the exact v0.26.0 `cu129` x86_64 release wheel. The
596.49 driver runs this older CUDA runtime successfully.

The historical `/home/xiaoda/vllm-lab/.venv` was removed on 2026-07-26 after
the v0.26 baseline was committed and its script references were checked. v0.15
is retained only as the `exp/scheduler-trace` Git branch/commits and archived
raw outputs. Source comparisons use `git show`; v0.15 is no longer a runnable
project lane.

## Measured gate results

### WSL, pinned memory, and UVA

| Configuration | pin memory | UVA | Result |
|---|---:|---:|---|
| WSL default | false | false | `RuntimeError: UVA is not available` |
| `VLLM_WSL2_ENABLE_PIN_MEMORY=1` | true | true | UVA allocation and readback passed |

The upstream UVA issue and proposed automatic V1 fallback PR were still open
when this test was performed:

- https://github.com/vllm-project/vllm/issues/47292
- https://github.com/vllm-project/vllm/pull/47579

The test did not use that unmerged fallback change.

### Model Runner and attention backend

Each offline and server startup logs:

```text
Using V2 Model Runner
Cannot use FA version 2 ... compute capability >= 8
Using TRITON_ATTN attention backend
```

Measured execution path:

- Model Runner: MRV2.
- Attention backend: `TRITON_ATTN`.
- FlashAttention 2: unavailable on SM 7.5.
- FlashInfer top-p/top-k sampler: unavailable on SM 7.5; vLLM fallback used.
- Multiprocessing: vLLM forces `spawn` because WSL NVML is incompatible with
  `fork`.
- `--enforce-eager` was used for the compatibility gate, so this run did not
  validate torch.compile or CUDA Graphs.

### Offline generation

Three independent engine processes started successfully. The first process
made three consecutive generation calls; the next two made one call each.
All five calls returned generated tokens.

With `gpu_memory_utilization=0.70` and `max_model_len=512`, representative
offline logs report:

- model weights: about 0.93 GiB;
- available KV cache: about 2.95 GiB;
- KV cache capacity: 257,824 tokens;
- first generation: 2.154 s, including first-shape Triton JIT;
- subsequent calls in the same engine: 0.203 s and 0.211 s;
- later cold-start process generation calls: 0.290 s and 0.290 s after engine
  initialization.

The generated wording is not an accuracy evaluation. Gate success means the
engine initialized and returned valid output token IDs without UVA, pinned
memory, worker-startup, ABI, or out-of-memory errors.

### OpenAI-compatible server

The server started on `127.0.0.1:8000`.

- `GET /health`: HTTP 200.
- `POST /v1/chat/completions`: HTTP 200.
- Response content: `Ready.`

After the successful response, manual SIGINT shutdown logged an
`EngineDeadError` in the async output handler and one leaked semaphore. This
was teardown-only in this run, but should be rechecked before treating service
shutdown/restart as production-clean.

## Reproduction

Offline:

```bash
cd /home/xiaoda/vllm-lab/vllm
VLLM_WSL2_ENABLE_PIN_MEMORY=1 \
VLLM_USE_V2_MODEL_RUNNER=1 \
/home/xiaoda/vllm-lab/.venv-v026/bin/python \
  scripts/lab_v026_offline_smoke.py \
  --model /home/xiaoda/vllm-lab/models/Qwen2.5-0.5B-Instruct \
  --output results/offline_smoke.json \
  --repetitions 3
```

Server:

```bash
cd /home/xiaoda/vllm-lab/vllm
./scripts/serve_v026.sh
```

## Evidence and remaining limits

Summary: `results/smoke_test.json`.

Raw logs and responses are preserved outside the Git repository under
`/home/xiaoda/vllm-lab/outputs/v026-gate-20260726/`. Source facts, measured
results, and current inference are separated as follows:

- Source fact: SM 7.5 is accepted by vLLM but does not meet the logged FA2
  requirement.
- Measured: MRV2, TRITON_ATTN, UVA readback, three engine starts, five
  generation calls, and the HTTP responses.
- Inference: Path A is suitable for the next Scheduler Trace tasks only when
  the pinned-memory opt-in remains explicit.

Not yet validated:

- torch.compile and CUDA Graph execution;
- long-prefill or high-concurrency stability;
- performance cost and WSL stability of opt-in pinned memory over long runs;
- graceful OpenAI server shutdown;
- 8K/16K workloads on the 6 GiB GPU.

## 30-second interview explanation

On vLLM v0.26.0, Model Runner V2 requires UVA-backed staged buffers. WSL
defaults to disabled pinned memory, so I first reproduced the exact UVA
failure. I then used v0.26's official pinned-memory opt-in, without patching
the source, and verified UVA readback, three independent MRV2 starts, offline
generation, and an OpenAI HTTP request. The RTX 2060 cannot use FA2, so the
measured backend is TRITON_ATTN. I therefore classify the machine as Path A
with an explicit WSL environment requirement, not as an unconditional MRV2
pass.
