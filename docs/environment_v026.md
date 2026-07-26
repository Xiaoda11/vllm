# vLLM v0.26.0 环境与 MRV2 Gate

日期：2026-07-26

运行 ID：`v026-gate-20260726`

结果：**Path A 通过，但必须启用 vLLM 官方提供的 WSL pinned memory 开关**

## 结论

当前 RTX 2060 Laptop / WSL2 环境可以运行 vLLM v0.26.0 的 Model Runner
V2，但不能直接使用 WSL 默认配置。默认的 `pin_memory=False` 会导致 UVA
不可用，最小 `UvaBuffer` 测试会抛出
`RuntimeError: UVA is not available`。

本次没有修改 vLLM 源码。v0.26.0 已包含正式合入的
`VLLM_WSL2_ENABLE_PIN_MEMORY` 开关。设置
`VLLM_WSL2_ENABLE_PIN_MEMORY=1` 后，pinned memory 和最小 UVA buffer
均可正常工作；三个独立 MRV2 引擎进程都能完成生成，OpenAI 兼容服务也能
成功返回健康检查和聊天补全响应。

因此，后续在这台机器上进行 MRV2 GPU 实验时，必须设置：

```bash
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export VLLM_USE_V2_MODEL_RUNNER=1
```

第二个变量用于明确控制实验证据：它可以避免未来发生静默 fallback，
从而把非 MRV2 路径误报为 MRV2 实测。

## 固定源码与环境

| 项目 | 实测值 |
|---|---|
| 分支 | `exp/mrv2-scheduler-trace` |
| Commit/tag | `f2654939e69b4069b13977e9aef3e31d4dcaf051` / `v0.26.0` |
| 运行环境 | `/home/xiaoda/vllm-lab/.venv-v026` |
| Python | 3.12.13 |
| vLLM | 0.26.0 editable 源码 + 对应 release 预编译二进制 |
| PyTorch | 2.11.0+cu129 |
| PyTorch CUDA runtime | 12.9 |
| NVIDIA 驱动 | 596.49 |
| 驱动报告的 CUDA | 13.2 |
| 本地 CUDA toolkit（`nvcc`） | 13.2 |
| GPU | NVIDIA GeForce RTX 2060 Laptop，6 GiB |
| Compute capability | 7.5 |
| WSL kernel | 6.18.33.2-microsoft-standard-WSL2 |
| 模型 | 本地 Qwen2.5-0.5B-Instruct，FP16 |

v0.26.0 release 没有提供 `cu130` wheel，因此当前安装的 native binaries
来自完全对应 v0.26.0 的 `cu129` x86_64 release wheel。596.49 驱动可以
正常运行这个较旧的 CUDA runtime。

历史环境 `/home/xiaoda/vllm-lab/.venv` 已于 2026-07-26 删除。删除前已
确认 v0.26 基线完成提交，并检查了脚本中的环境引用。v0.15 目前只保留
`exp/scheduler-trace` Git 分支、相关 commits 和归档的原始输出。源码对比
统一使用 `git show`，v0.15 不再作为可运行的项目分支。

## Gate 实测结果

### WSL、pinned memory 与 UVA

| 配置 | pin memory | UVA | 结果 |
|---|---:|---:|---|
| WSL 默认配置 | false | false | `RuntimeError: UVA is not available` |
| `VLLM_WSL2_ENABLE_PIN_MEMORY=1` | true | true | UVA 分配与回读通过 |

执行本次测试时，上游 UVA issue 和自动 fallback 到 V1 的候选 PR 仍处于
打开状态：

- https://github.com/vllm-project/vllm/issues/47292
- https://github.com/vllm-project/vllm/pull/47579

本次测试没有使用该尚未合入的 fallback 修改。

### Model Runner 与 attention backend

每次离线推理和服务启动日志均包含：

```text
Using V2 Model Runner
Cannot use FA version 2 ... compute capability >= 8
Using TRITON_ATTN attention backend
```

实测执行路径：

- Model Runner：MRV2。
- Attention backend：`TRITON_ATTN`。
- FlashAttention 2：SM 7.5 不可用。
- FlashInfer top-p/top-k sampler：SM 7.5 不可用，实际使用 vLLM fallback。
- 多进程：由于 WSL NVML 与 `fork` 不兼容，vLLM 强制使用 `spawn`。
- 兼容性 Gate 使用了 `--enforce-eager`，所以本次没有验证 torch.compile
  或 CUDA Graphs。

### 离线生成

三个独立引擎进程均成功启动。第一个进程连续执行三次生成，后两个进程各
执行一次生成，五次调用均返回生成 token。

在 `gpu_memory_utilization=0.70`、`max_model_len=512` 配置下，代表性离线
日志显示：

- 模型权重约 0.93 GiB；
- 可用 KV cache 约 2.95 GiB；
- KV cache 容量为 257,824 tokens；
- 首次生成耗时 2.154 秒，其中包含首个 shape 的 Triton JIT；
- 同一引擎内后续调用分别耗时 0.203 秒和 0.211 秒；
- 后续冷启动进程完成引擎初始化后，生成调用均耗时 0.290 秒。

生成文本本身不构成准确率评估。这里的 Gate 通过仅表示：引擎成功初始化并
返回有效输出 token IDs，过程中没有出现 UVA、pinned memory、worker
启动、ABI 或显存不足错误。

### OpenAI 兼容服务

服务成功启动在 `127.0.0.1:8000`：

- `GET /health`：HTTP 200。
- `POST /v1/chat/completions`：HTTP 200。
- 响应内容：`Ready.`。

成功响应后，手动通过 SIGINT 关闭服务时，异步输出处理器记录了一次
`EngineDeadError`，并出现一个 leaked semaphore。本次只在退出阶段观察到
该问题，但在把服务关闭和重启视为 production-clean 之前需要重新验证。

## 复现方法

离线推理：

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

启动服务：

```bash
cd /home/xiaoda/vllm-lab/vllm
./scripts/serve_v026.sh
```

## 证据与剩余边界

精简摘要：`results/smoke_test.json`。

原始日志和响应保存在 Git 仓库外：
`/home/xiaoda/vllm-lab/outputs/v026-gate-20260726/`。源码事实、实测结果与
当前推断区分如下：

- 源码事实：vLLM 支持 SM 7.5，但该架构不满足日志中显示的 FA2 要求。
- 实测结果：MRV2、`TRITON_ATTN`、UVA 回读、三个独立引擎启动、五次生成
  调用和 HTTP 响应。
- 当前推断：只有持续显式启用 pinned-memory 开关时，Path A 才适用于后续
  Scheduler Trace 任务。

尚未验证：

- torch.compile 和 CUDA Graph 执行；
- long-prefill 或高并发稳定性；
- 长时间启用 pinned memory 的性能成本和 WSL 稳定性；
- OpenAI 服务的优雅关闭；
- 6 GiB GPU 上的 8K/16K workload。

## 30 秒面试表达

vLLM v0.26.0 的 Model Runner V2 需要基于 UVA 的 staged buffers。WSL
默认关闭 pinned memory，所以我先复现了明确的 UVA 初始化失败。随后我在
不修改源码的情况下启用 v0.26 官方 pinned-memory 开关，验证了 UVA 回读、
三个独立 MRV2 进程启动、离线生成和一次 OpenAI HTTP 请求。RTX 2060
无法使用 FA2，实测 backend 是 `TRITON_ATTN`。因此我把当前机器判定为
具有明确 WSL 环境前置条件的 Path A，而不是无条件通过 MRV2。
