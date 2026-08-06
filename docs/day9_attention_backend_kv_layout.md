# Day 9：Attention backend 与 KV Cache layout

## 问题、假设与判定标准

本日只回答与当前模型和 RTX 2060 直接相关的三个问题：

1. v0.26 怎样为当前 KV-cache group 选择 attention backend？
2. Qwen2.5-0.5B-Instruct 的 KV cache 在 GPU 上是什么逻辑 shape 和物理布局？
3. Scheduler/MRV2 的 block table、slot mapping 和 attention metadata 怎样连接到
   `TRITON_ATTN`？

实验前假设：

- SM 7.5 不满足 FA2 的 compute-capability 要求，自动选择会落到
  `TRITON_ATTN`；
- 当前 Qwen 虽然 HF config 含 `sliding_window=32768`，但
  `use_sliding_window=false`，运行时应产生一个 full-attention KV group；
- FP16、2 个 KV heads、64 head size 的 Triton KV cache 逻辑 shape 应为
  `[num_blocks, 2, 16, 128]`，其中 K/V 合并进最后一个 content dimension；
- 默认 NHD 布局应体现在 tensor stride，而不改变上述逻辑 shape；
- 一次性读取 tensor shape、stride 和 storage offset 不会触发 GPU-to-CPU copy。

判定标准：

- clean run 明确记录 MRV2、backend、spec type、group/layer 数和 sliding-window
  状态；
- trace 中的 KV token 容量、block 数、page bytes 和启动日志能够互相反算；
- trace 默认关闭，且不调用 `.item()`、`.cpu()` 或显式 synchronize；
- v0.15 与 v0.26 的差异只陈述为 shape/layout 接口变化，不虚构容量收益。

## 环境与正式产物

- vLLM：v0.26.0；
- instrumentation commit：`5dea775d39b2e308a1538941df32daac116ec5ba`；
- 正式运行的 `git_status_short` 为空；
- runner：MRV2 Path A，`VLLM_WSL2_ENABLE_PIN_MEMORY=1`；
- GPU：RTX 2060 Laptop 6 GiB，compute capability 7.5；
- 模型：Qwen2.5-0.5B-Instruct，FP16；
- workload：canonical S3，A=8K、B=16K、token budget=2048；
- Prefix Cache：关闭；CUDA Graph：关闭。

正式产物：

```text
/home/xiaoda/vllm-lab/outputs/day9-s3-attention-layout-clean-20260801
```

正式命令：

```bash
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export VLLM_USE_V2_MODEL_RUNNER=1

/home/xiaoda/vllm-lab/.venv-v026/bin/python \
  scripts/lab_v026_workload.py \
  --config benchmarks/scheduler_trace/configs/s3_dual_simultaneous.json \
  --model /home/xiaoda/vllm-lab/models/Qwen2.5-0.5B-Instruct \
  --token-budget 2048 \
  --run-id day9-s3-attention-layout-clean-20260801 \
  --scheduler-trace
```

## Opt-in KV layout trace

MRV2 在 `init_kv_cache()` 完成后增加一次性 `kv_cache_layout` 事件，记录：

- backend 和 KV spec 类型；
- KV-cache group、attention group 和 layer 数；
- block size、kernel block size、page size；
- KV heads、K/V head size 和 sliding window；
- tensor shape、dtype、device、stride 和 storage offset；
- NHD/HND layout。

Tracing 未开启时 writer 为 `None`，记录函数立即返回。开启时只读取 tensor
对象的 Python metadata；不读取 tensor 内容，不调用 `.item()`、`.cpu()` 或
显式 synchronize。

## Backend 选择链

### 源码事实

`Attention` 层根据 attention 类型、sliding-window 状态、KV dtype、head size
等参数调用 `get_attn_backend()`。如果配置了 `backend_per_kind`，匹配
`KVCacheSpecKind` 的条目优先于全局 backend；未匹配的 kind 回退到全局配置或
自动选择。

CUDA 自动选择会验证候选 backend 的 dtype、KV dtype、block size、head size、
attention type 和 compute capability，然后选择有效候选中优先级最高者。指定
backend 时不做静默 fallback：不兼容就直接报错。

### 本机实测

正式启动日志记录：

```text
Using V2 Model Runner
FA2 is only supported on devices with compute capability >= 8
Using TRITON_ATTN attention backend out of potential backends:
['TRITON_ATTN', 'FLEX_ATTENTION']
```

因此本机链路是：

```text
Qwen decoder self-attention
→ FullAttentionSpec requirements
→ CUDA capability validation on SM 7.5
→ FLASH_ATTN/FA2 rejected
→ valid candidates TRITON_ATTN and FLEX_ATTENTION
→ priority selects TRITON_ATTN
```

不能把“本机选中 Triton”推广为 v0.26 在所有 CUDA GPU 上都优先选择 Triton。

## 当前模型的 KV group 与 layout

正式 trace 的唯一 `kv_cache_layout` 事件：

| 字段 | 实测值 |
|---|---:|
| KV-cache groups | 1 |
| Attention groups | 1 |
| Layers | 24 |
| Backend | `TRITON_ATTN` |
| Spec | `FullAttentionSpec` |
| Sliding window | `null` |
| Block / kernel block size | 16 / 16 |
| KV heads | 2 |
| K/V head size | 64 / 64 |
| Blocks | 17,243 |
| Per-layer page bytes | 8,192 |
| Cache dtype / tensor dtype | `auto` / FP16 |
| Logical shape | `[17243, 2, 16, 128]` |
| Logical stride | `[4096, 128, 256, 1]` |
| Layout | NHD |

`sliding_window=null` 是运行时 spec 证据。HF config 中存在
`sliding_window=32768` 不等于模型实际启用了 sliding-window attention；当前
模型同时设置 `use_sliding_window=false`。

### Logical shape 与 content dimension

v0.26 Triton backend 返回：

```text
[num_blocks, num_kv_heads, block_size, 2 × head_size]
```

代入实测值：

```text
[17243, 2, 16, 2 × 64]
= [17243, 2, 16, 128]
```

最后的 128 是统一 content dimension：`K[64] | V[64]`。这不是 128 维的
单独 K head。

### NHD 物理布局与 stride

逻辑视图为 `[B,H,N,D]`，NHD 要求实际连续次序为 `[B,N,H,D]`。对
`N=16, H=2, D=128`，逻辑视图的预期 stride 是：

```text
B: 16 × 2 × 128 = 4096
H: 128
N: 2 × 128 = 256
D: 1
```

实测正是 `[4096,128,256,1]`。因此 shape 与 stride 一起证明了“逻辑
`[B,H,N,D]`、物理 NHD”，不能只看 shape 判断物理布局。

### 容量闭环

```text
单层单 block
= 2 KV heads × 16 tokens × (64 K + 64 V) × 2 bytes
= 8,192 bytes

24 层单 block
= 8,192 × 24
= 196,608 bytes

总 KV bytes
= 17,243 × 196,608
= 3,390,111,744 bytes
= 3.1573 GiB

token capacity
= 17,243 × 16
= 275,888 tokens
```

这与启动日志的 `Available KV cache memory: 3.16 GiB` 和
`GPU KV cache size: 275,888 tokens` 闭合。

## v0.15 与 v0.26 的布局差异

历史 commit `f176443446f659dbab5315e056e605d8984fd976` 中，Triton backend
返回：

```text
v0.15: [num_blocks, 2, block_size, num_kv_heads, head_size]
v0.26: [num_blocks, num_kv_heads, block_size, 2 × head_size]
```

v0.15 用独立的 K/V 维；v0.26 将 K/V 统一进 content dimension，并通过
stride order 表达 NHD/HND。对当前未量化模型，两者每 block 的元素数相同，
所以这项变化本身不意味着 KV 容量翻倍或显存减半。

## Scheduler metadata 怎样到达 backend

当前证据链是：

```text
SchedulerOutput.num_scheduled_tokens / block IDs
→ MRV2 RequestState 与 persistent block tables
→ idx_mapping gather 本轮 request rows
→ per-step block_tables + slot_mappings
→ TritonAttentionMetadata
→ TRITON_ATTN unified attention kernel
```

- block table 将 request 的逻辑 token blocks 映射到物理 KV pages；
- slot mapping 指定本轮新 K/V 应写入的物理 slot；
- `query_start_loc` 划分扁平 query tensor 中的不同请求；
- sequence lengths 和 computed-token state 限定每个 query 可见的上下文；
- backend-specific metadata builder 将这些公共状态整理为
  `TritonAttentionMetadata`；
- Triton kernel 按 block table 访问 paged KV cache，并按 slot mapping 写入新
  token 的 K/V。

Day 8 正式 trace 已证明 24 个 attention layer 的 metadata 类型均为
`TritonAttentionMetadata`。Day 9 新事件进一步证明这些 metadata 对应的真实
cache 是上述 FullAttention/NHD tensor。

## 验证结果与边界

静态验证：

```text
30 tests passed
Ruff passed
git diff --check passed
```

正式 S3 两个请求均完成，但运行中出现 `kernel_unified_attention` 和
`reduce_segments` 的 cold JIT warning。本日目标是 backend/layout 证据链，
不是性能 benchmark，因此不使用本次 TTFT/E2E 推导性能结论，也不将它与
Day 8 单次运行直接比较。

未解释问题：相同 workload 在不同独立进程中仍可能发生 Triton JIT，后续若要
比较 Day 10 的 TTFT/TPOT，需要固定 warmup、缓存状态和运行顺序。
