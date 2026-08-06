# Day 6：KV Cache 分配链

## 问题、假设与判定标准

本日追踪 Scheduler 何时请求 KV block、partial prefill 后 block table
如何增长，以及 Prefix Cache 命中如何改变 computed tokens 和物理分配。

实验前假设：

1. 对普通 attention group，新增 KV blocks 约为
   `ceil((computed + scheduled) / block_size) - current_blocks`。
2. 单请求 8K prompt、2048 token budget、16-token block 下，前四个
   prefill step 各增加 128 blocks。
3. 两个无共享前缀请求使用不相交的 block IDs。
4. S6 的第二个请求会命中 6144 tokens，即复用 384 blocks，只为本轮
   2047 个新 tokens 取得 128 个新物理 blocks。

判定同时使用 request block table、Prefix Cache hit、全局 free-block pool
和请求 timing。加入请求 block table 的 block 不一定是新取得的物理 block：
Prefix Cache hit 会把已有 block 加入第二个请求的 table。

## 环境与产物

- vLLM：v0.26.0；
- 正式实验提交：`d403324df8f360113b919ae37b80f1fe4b895325`；
- 四组 metadata 的 `git_status_short` 均为空；
- CSV 语义修正提交：`fd1508bf8f`；
- GPU：RTX 2060 Laptop 6 GiB / WSL2；
- `VLLM_USE_V2_MODEL_RUNNER=1`；
- `VLLM_WSL2_ENABLE_PIN_MEMORY=1`；
- attention backend：`TRITON_ATTN`；
- block size：16 tokens；
- token budget：2048；
- chunked prefill 开启，MRV2 async scheduling 开启。

正式产物：

```text
/home/xiaoda/vllm-lab/outputs/day6-s1-single-b2048-20260729
/home/xiaoda/vllm-lab/outputs/day6-s3-dual-no-prefix-b2048-20260729
/home/xiaoda/vllm-lab/outputs/day6-s6-prefix-off-b2048-20260729
/home/xiaoda/vllm-lab/outputs/day6-s6-prefix-on-b2048-20260729
```

每个目录包含 `run_metadata.json`、`request_timing.csv`、
`scheduler_trace.jsonl`、`scheduler_trace.mrv2.jsonl` 和
`scheduler_trace.csv`。

## 源码调用链

### 1. Scheduler 计算本轮工作

`Scheduler.schedule()` 不维护独立的 prefill/decode 阶段，而是让
`num_computed_tokens` 追赶当前 `num_tokens_with_spec`。

对 running 请求：

```text
计算 num_new_tokens
→ 受全局 token_budget 截断
→ KVCacheManager.allocate_slots()
→ 成功：记录 scheduled tokens
→ 失败：从最低优先级 running 请求开始 preempt
```

对 waiting 请求：

```text
KVCacheManager.get_computed_blocks()
→ 得到本地 Prefix Cache hit
→ 计算 num_new_tokens = num_tokens - cached/computed tokens
→ KVCacheManager.allocate_slots()
→ 成功：WAITING → RUNNING
→ 失败：停止接纳，本请求继续等待
```

所以 token budget 不足和 KV 空间不足是两条不同路径。前者先截断
`num_new_tokens`；后者由 `allocate_slots()` 返回 `None`。

### 2. Prefix Cache 查找

`KVCacheManager.get_computed_blocks()`：

1. Prefix Cache 关闭时直接返回 0 hit。
2. 开启时按 block hash 查找最长完整 block 前缀。
3. 即使整段 prompt 命中，也至少重算最后一个 token 以获得 logits。
4. 返回 cached blocks 和 block-aligned computed-token 数。

### 3. token 数转换为 block 分配

`KVCacheManager.allocate_slots()`：

```text
total_computed = existing computed + local hit + external hit
num_tokens_need_slot = total_computed + num_new_tokens + lookahead
→ coordinator.get_num_blocks_to_allocate()
→ required_blocks > available_blocks：返回 None
→ 接入 Prefix Cache 命中的 computed blocks
→ coordinator.allocate_new_blocks()
→ cache 已完成的完整 blocks
```

真正从物理 free queue 取块发生在 `BlockPool.get_new_blocks()`。Prefix
Cache hit 则通过 `BlockPool.touch()` 增加已有 block 的引用计数，而不是复制
KV 内容。

请求释放 block 时，未 hash 的 block 优先进入 free queue；带 hash 的 block
作为可驱逐 Prefix Cache 留在队列另一端。

## GPU 实测

### 单请求 partial prefill

S1 为 8192-token prompt、32-token output。

| Step | computed before | scheduled | table before | table after | table added |
|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 2048 | 0 | 128 | 128 |
| 2 | 2048 | 2048 | 128 | 256 | 128 |
| 3 | 4096 | 2048 | 256 | 384 | 128 |
| 4 | 6144 | 2048 | 384 | 512 | 128 |
| 5 | 8192 | 1 | 512 | 513 | 1 |
| 21 | 8208 | 1 | 513 | 514 | 1 |

前四步严格满足 `2048 / 16 = 128`。step 5 为第一次 decode 分配新槽位；
之后 16 个 decode tokens 共享该 block，直到 step 21 再跨入下一个 block。

### 两请求无共享前缀

S3 中 A=8K、B=16K 同时到达：

- A 在 step 1 首次分配 128 blocks。
- B 在 step 5 首次调度 2047 tokens，并建立 128-block table。
- 当 B 首次进入 running 时，A/B block ID 交集为 0。
- A 的最大 table 为 514 blocks，B 为 1026 blocks。
- 无分配失败、无 preemption。

### Prefix Cache on/off

S6 的 A/B 均为 8K prompt，共享前 6144 tokens，B 延迟 0.5 秒到达。

| Prefix Cache | B hit tokens/blocks | B 首次 scheduled | B 首次 table | A/B 共享 IDs | 本 step free blocks 减少 | B TTFT |
|---|---:|---:|---:|---:|---:|---:|
| off | 0 / 0 | 2047 | 128 | 0 | 129 | 10.895199 s |
| on | 6144 / 384 | 2047 | 512 | 384 | 129 | 7.748444 s |

Prefix Cache 开启时，B 的 512-block table 由 384 个复用 blocks 和 128 个
新物理 blocks 组成。本 step 的 free pool 减少 129，是因为 A 同时获得 1 个
decode block，B 获得 128 个新 blocks。

`block_table_blocks_added` 表示 block 加入该请求的 table，不能直接当作物理
新分配量；这也是 CSV 不再使用含混的 `allocated_blocks` 计数名的原因。

单次独立 Engine run 都包含 Triton JIT，因此表中的 TTFT 只说明本次运行的
描述性结果。虽然 B TTFT 从 10.90 秒降到 7.75 秒，但没有重复/warmup 前不能
把 28.9% 差异报告为稳定性能收益。

## 分配失败的源码结论与实测边界

源码事实：

- waiting 请求分配失败时，`allocate_slots()` 返回 `None`，Scheduler 停止
  本轮 waiting admission，请求保持 waiting。
- running 请求扩容失败时，Scheduler 逐个 preempt 低优先级 running 请求，
  释放其 blocks 后重试。
- preempted 请求清空 computed-token 进度并回到 waiting，后续需要重计算。

本日四组运行最高 KV usage 很低，没有实测分配失败或 preemption。降低 KV
容量并稳定复现上述路径是 Day 7，而不是用本日低压力结果声称已经测到。

## 结论

Scheduler 先用 computed-token 差值决定本 step 工作量，再让
`KVCacheManager.allocate_slots()`把 token 范围映射到 block table。普通
partial prefill 只扩展 table 尾部；Prefix Cache hit 会把已有物理 blocks
接入新请求的 table，并只为未命中部分取得新 blocks。判断物理分配必须联合
看 cached blocks、请求间 block-ID 交集和全局 free pool，不能只看请求 table
增加量。

未解决问题：KV 容量不足时，running request 的扩容失败、抢占、释放和重算
能否在固定小容量配置下逐 step 稳定复现。
