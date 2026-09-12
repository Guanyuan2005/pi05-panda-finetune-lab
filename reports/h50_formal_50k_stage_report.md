# π0.5 Panda h50 正式模型：50k 阶段评测报告

日期：2026-09-12  
模型配置：`pi05_panda_h50_formal`  
实验名：`panda_h50_formal_v1`  
Checkpoint：`/root/shared-nvme/checkpoints/pi05_panda_h50_formal/panda_h50_formal_v1/50000`

## 1. 阶段结论

50k checkpoint 已经学会四类物体的基本抓取和放置。在 40 个冻结的普通测试场景上，最佳执行参数为 `execute_actions=10`、`smooth_alpha=0.2`，成功率为 **34/40（85%）**。

同一模型在 40 个恢复测试场景上使用 `alpha=0.2` 时成功率为 **22/40（55%）**；关闭 EMA（`alpha=1.0`）后提高到 **30/40（75%）**。这说明模型具备一定恢复能力，但强平滑产生的响应滞后明显妨碍了快速纠错。

因此，50k 可以作为当前基线和候选部署模型，但暂不应认定为最终模型。部署参数需要在普通场景的平稳性和恢复场景的响应速度之间权衡；建议测试较弱平滑或自适应平滑，再训练到 75k 并在完全相同的测试集上复测。

## 2. 评测设置

- 模型动作长度：`action_horizon=50`
- 滚动执行长度：`execute_actions=10`
- 单回合最大规划步数：`max_steps=100`
- 成功后继续执行：`post_success_cycles=5`（早期单场景验证曾使用 30）
- 普通评测集：冻结测试集中的 40 个 `recovery_type=none` 场景，四种目标各 10 个
- 恢复评测集：冻结测试集中的 40 个 recovery 场景，四种恢复类型各 10 个
- 普通场景的不同 `smooth_alpha` 使用同一批场景，以保证横向可比

评测脚本：`scripts/run_mujoco_policy_eval_receding.py`

## 3. 普通场景：平滑系数对照

| smooth_alpha | apple | box | can | orange | 总成功数 | 成功率 |
|---:|---:|---:|---:|---:|---:|---:|
| **0.2** | 8/10 | 8/10 | 10/10 | 8/10 | **34/40** | **85.0%** |
| 0.3 | 9/10 | 6/10 | 9/10 | 7/10 | 31/40 | 77.5% |
| 0.5 | 6/10 | 7/10 | 9/10 | 8/10 | 30/40 | 75.0% |
| 1.0（无 EMA 平滑） | 8/10 | 6/10 | 10/10 | 8/10 | 32/40 | 80.0% |

### 3.1 解读

- `alpha=0.2` 当前总体最好，比无平滑多成功 2 个回合。
- `can` 最稳定：在 `alpha=0.2` 和无平滑时均为 10/10。
- 相对无平滑，`alpha=0.2` 的成功率提升主要来自 `box`（6/10 → 8/10）。
- 四组只有每组 40 回合，5 个百分点的总体差距还不足以证明统计上的稳定优势；但 `alpha=0.2` 同时具有最高成功率和更强的抑制抖动效果，因此适合作为当前默认值。
- 成功率没有随 alpha 单调变化，说明结果还受到场景难度、闭环状态和模型采样的共同影响。

## 4. Recovery 场景结果

统一使用 `execute_actions=10`，对比 `smooth_alpha=0.2` 与关闭 EMA 的 `smooth_alpha=1.0`。

| recovery_type | alpha=0.2 | alpha=1.0 | 变化 |
|---|---:|---:|---:|
| approach_offset | 8/10 | 10/10 | +2 |
| object_slip | 5/10 | 8/10 | +3 |
| place_offset | 5/10 | 5/10 | 0 |
| weak_grasp | 4/10 | 7/10 | +3 |
| **合计** | **22/40（55%）** | **30/40（75%）** | **+8（+20 pp）** |

`alpha=1.0` 的 recovery 结果按目标物体划分为：apple 8/10、box 7/10、can 10/10、orange 5/10。

### 4.1 解读

- 关闭 EMA 后，`approach_offset`、`object_slip` 和 `weak_grasp` 分别多成功 2、3、3 个回合；`place_offset` 没有变化。
- 8/40 的提升幅度远大于普通场景中 `alpha=0.2` 相对无平滑的 2/40 优势，说明强 EMA 平滑对恢复动作的负面影响不可忽略。
- `place_offset` 在两种设置下均为 5/10，更可能是模型能力或数据覆盖问题，而不是平滑造成。
- `orange` 在无平滑 recovery 中只有 5/10，是当前需要重点检查失败视频的目标物体。
- 若普通与 recovery 场景等权，`alpha=1.0` 合计为 62/80（77.5%），`alpha=0.2` 为 56/80（70%）。因此包含较多扰动和失败恢复的部署环境更适合弱平滑或自适应平滑。

## 5. 动作平滑原理

评估器对模型输出的关节与夹爪目标使用指数移动平均（EMA）：

```text
smoothed_target[t] = alpha * raw_target[t]
                   + (1 - alpha) * smoothed_target[t-1]
```

其中：

- `raw_target[t]` 是模型当前输出的 8 维目标动作；
- `smoothed_target[t-1]` 是上一次平滑后的目标；
- `alpha` 越小，历史目标权重越大，动作越平滑，但响应滞后也越明显；
- `alpha=1.0` 时，平滑目标等于模型原始输出，相当于关闭 EMA。

在每个 100 ms 动作周期内部，评估器还会从前一目标向新目标做线性插值。EMA 用于过滤相邻模型动作之间的高频跳变，周期内插值用于避免控制目标瞬间阶跃，两者共同减少机械臂可见抖动。

平滑不能提升模型本身的认知或恢复能力。过强平滑还可能延迟夹爪闭合、快速重定位等纠错动作，因此普通场景的最佳 alpha 不一定也是 recovery 场景的最佳 alpha。

## 6. 与训练指标的关系

50k 附近观测到的 `loss` 和 `grad_norm` 仍有随机波动，`param_norm` 只有极小幅度增长，没有发现数值发散证据。小 batch 的 flow-matching 训练中，单步 loss 和梯度范数通常不会平滑下降到 0；闭环成功率比单步训练指标更能反映当前任务效果。

本次评测证明模型已经取得有效行为能力，因此不能根据 loss 没有继续明显下降就判断训练失败。另一方面，即使无平滑 recovery 已达到 75%，放置偏移和橘子场景仍明显偏弱，说明低 loss 并不等价于闭环鲁棒性已经充分收敛。

## 7. 下一阶段决策

1. 保留 50k checkpoint，作为固定基线，不覆盖、不删除。
2. 对同一批场景统计 `alpha=0.2` 与 `alpha=1.0` 的逐 seed 成功翻转，并重点查看 `orange` 和 `place_offset` 失败视频。
3. 为兼顾抖动与纠错速度，优先测试较弱平滑（如 `alpha=0.8`）或将机械臂与夹爪使用不同 alpha；不要直接把 `alpha=0.2` 作为所有状态下的固定部署参数。
4. 从 50k checkpoint 恢复训练至 75k，并确保终端输出通过 `tee` 写入持久日志。
5. 对 75k 使用同一批普通 40 场景和 recovery 40 场景复测，采用逐场景配对比较。
6. 若 75k/100k 在无平滑或弱平滑条件下仍集中失败于放置偏移或重抓，应采集模型自身失败状态上的 teacher 修正轨迹，构建 policy-induced recovery/DAgger 数据。

建议把“明确提升”暂定为 recovery 至少多成功 4～5 个回合（约提升 10～12.5 个百分点），并结合相同 seed 上的成功/失败翻转及视频行为判断，避免只比较一个波动较大的总体比例。

## 8. 当前日志位置

- 普通 alpha 矩阵：`/root/shared-nvme/logs/h50_formal_50k_quick_matrix/`
- 无平滑普通场景：`/root/shared-nvme/logs/h50_formal_50k_no_smoothing/alpha_1.0/`
- Recovery alpha=0.2：`/root/shared-nvme/logs/h50_formal_50k_recovery/alpha_0.2/`
- Recovery alpha=1.0：`/root/shared-nvme/logs/h50_formal_50k_recovery/alpha_1.0/`
- 50k checkpoint：`/root/shared-nvme/checkpoints/pi05_panda_h50_formal/panda_h50_formal_v1/50000`

## 9. 尚未完成

- 按失败视频标注失败阶段：接近、首次夹取、抬升、滑落、重新抓取、放置。
- 75k 与 100k checkpoint 的同场景配对评测。
- 从动作日志计算速度变化、动作差分或 jerk；当前“更平滑”的判断主要来自 EMA 机制和视频观察，尚未形成独立的定量抖动指标。
