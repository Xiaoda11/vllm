# Day 3：可控 Workload Generator

## 工程问题

在修改 Scheduler 前，先固定一套能精确控制 prompt/output token 数、请求
到达时间、并发、共享前缀和 request ID 的输入。请求级时间用于确认输入
是否按计划进入引擎，不代替 Day 4 的逐 step Scheduler Trace。

## 假设、观测量与判定标准

假设：

1. 预分词 token ID 可以消除自然语言 prompt 的长度误差。
2. 独立的异步 request task 可以控制相对到达时间。
3. 相同 shared-prefix ID 和长度会产生逐 token 相同的前缀。

观测量：

- 每个请求的 canonical/effective prompt tokens。
- planned arrival、submitted、first token 和 finished 时间。
- TTFT、E2E、实际 output tokens。
- 整个 prompt 和 shared prefix 的 SHA-256。

判定标准：

- S1–S6 配置均通过 schema 和 token 构造验证。
- S3 的 A/B 在 `t≈0` 提交，长度精确为 8192/16384。
- S4 的 B 在 `t≈0.5s` 提交。
- S6 的两个 shared-prefix 摘要相同、suffix 不同。
- Path A GPU smoke 使用 V2 Model Runner，输出 token 数符合配置。

## 实现

- `scripts/lab_v026_workload.py`
  - 使用 `AsyncLLM` 并发提交请求。
  - 使用预分词 `TokensInput` 保证 prompt 长度。
  - `ignore_eos=True` 保证固定 output 长度。
  - `--prompt-scale` 只缩放 prompt/shared prefix，并同时保留原始长度。
  - 每个 run 写入唯一目录，不覆盖旧证据。
- `benchmarks/scheduler_trace/configs/`
  - 固定 S1–S6。
- `request_timing.csv`
  - 只包含 request-level 可观测量。
- `run_metadata.json`
  - 保存 commit、dirty 状态、生成器哈希、实际命令、环境与结果。

## 验证结果

静态/CPU 验证：

- 六个配置全部通过。
- S3 token 构造为 8192/16384；`--prompt-scale 0.5` 后为
  4096/8192。
- S6 两个 6144-token 前缀的摘要一致，suffix 不同。
- `AsyncEngineArgs` 能接受固定 engine 配置。
- Python 语法、手工纯函数断言和 `git diff --check` 通过。
- `pytest --confcutdir=tests/lab tests/lab/test_v026_workload.py -q`：
  9 passed。
- Ruff lint 和 format check 均通过。

GPU 实测均使用：

- vLLM v0.26.0。
- Path A：`VLLM_USE_V2_MODEL_RUNNER=1`。
- `VLLM_WSL2_ENABLE_PIN_MEMORY=1`。
- 日志确认 `Using V2 Model Runner` 和 `TRITON_ATTN`。

### S1 缩放 smoke

Run：`day3-s1-2k-smoke-20260727`

| Request | Prompt | Output | TTFT | E2E |
|---|---:|---:|---:|---:|
| A | 2048 | 32 | 0.610s | 1.360s |

这是 2K smoke，不是 8K 测量。

### S3 canonical

Run：`day3-s3-canonical-20260727`

| Request | Submitted | Prompt | Output | TTFT | E2E |
|---|---:|---:|---:|---:|---:|
| A | 0.000225s | 8192 | 32 | 5.693s | 28.076s |
| B | 0.000408s | 16384 | 32 | 27.430s | 28.179s |

两个请求均在 `t≈0` 提交，canonical 8K/16K workload 可在 RTX 2060
上执行。本次是冷启动后的首个 workload，日志出现 Triton kernel JIT，
因此这些延迟只用于功能验收，不作为稳定 benchmark。

仅凭 request timing 不能断言 A/B 每个 Scheduler step 分到了多少 token，
也不能断言 B 等待的具体内部原因。

### S4 arrival smoke

Run：`day3-s4-arrival-smoke-20260727`

| Request | Planned | Submitted | Effective prompt |
|---|---:|---:|---:|
| A | 0.000s | 0.000261s | 2048 |
| B | 0.500s | 0.501429s | 4096 |

B 的提交偏差为 1.429ms，arrival 控制生效。这是 2K/4K 缩放验证，不是
canonical S4 性能数据。

## 结论与未解释问题

工程结论：生成器已经能够精确构造请求、控制并发与到达时间，并留下可
追溯的 request-level CSV/JSON 证据，足以作为 Day 4 Trace 的固定输入。

未解释问题：S3 中 B 的 TTFT 明显晚于 A，究竟对应怎样的 waiting/running
变化、每 step token budget 分配和 KV block 分配，必须由 Day 4 Trace
回答，不能从本日 timing 反推。

## 30 秒表达

我先用预分词 token ID 构建了 vLLM 的可控 workload，精确控制
prompt/output 长度、arrival time、并发、request ID 和共享前缀。请求通过
AsyncLLM 独立提交，每次实验保存 commit、dirty 状态、命令、prompt 摘要
和 TTFT/E2E。canonical 8K/16K 双请求已在 RTX 2060 的 MRV2 路径跑通；
但 request timing 只能证明输入和外部延迟，内部 token budget、KV 分配和
调度顺序要靠下一步逐 step trace。

## 个人验收

进入 Day 4 前，本人需要不看本文回答：

1. 为什么不用两个自然语言字符串声称它们是 8K/16K？
2. `concurrency=2` 和两个请求同时到达分别控制什么？
3. 为什么 CSV 中同时保存 planned arrival 和 submitted time？
4. S3 的 timing 能证明什么，不能证明什么？
5. `--prompt-scale 0.5` 后，哪些结论仍成立，哪些不能沿用？
6. S6 如何证明共享的是 token prefix，而不是相似文本？
