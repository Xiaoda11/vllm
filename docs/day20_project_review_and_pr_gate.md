# Day 20：PR Gate

## 最终 PR 决策

当前 waiting bypass patch 只保留在实验 fork，不向 upstream 提交，不发布外部
评论。这个决定不是因为问题不存在，而是因为已有数据证明当前 bound 不足以支撑
安全的通用策略。

| Gate | 结果 | 依据 |
|---|---|---|
| 问题真实且可稳定观察 | 通过 | 同 step B-blocked/C-fits HOL witness |
| 默认行为不变 | 通过 | feature 默认关闭，strict tests 覆盖 |
| 修改范围可解释 | 通过 | waiting admission 与少量生命周期状态 |
| 有明确局部收益 | 通过 | C TTFT median 33.250 s → 0.182 s |
| 有 starvation/delay bound | 不通过 | 无界 burst 推迟 B；count bound 不限制 C 生命周期 |
| 无新增 preemption/fairness 退化 | 不通过 | 长生命周期 bounded case 新增 3 次 preemption |
| 不重复 upstream 工作 | 不通过 | Day 15 审计时已有相近无界 skip draft 方向 |
| NCU 单 kernel 微架构证据 | 通过 | 真实 Prefill GEMM 的 full/stall counter 完整 |
| NCU 策略级归因 | 未通过但非必要 | launch 未与 mixed step 唯一对齐，无同-shape 对照 |

所以更准确的项目成果是一个有代码、有正反实验、有执行证据的调度负结果，而不是
待发布功能。
