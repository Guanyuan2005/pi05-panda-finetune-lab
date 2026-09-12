# π0.5 Panda h50：50k 与 100k 配对评测报告

日期：2026-09-12  
配置：`pi05_panda_h50_formal`  
实验：`panda_h50_formal_v1`  
评测设置：相同冻结场景、`execute_actions=10`、`max_steps=100`、`post_success_cycles=5`

> **方法更正（2026-09-12）：**训练教师会真实注入四类 recovery 事件；但当前 policy evaluator 只读取 `recovery_type` 标签，没有在 rollout 中执行对应扰动。因此下文将原“Recovery 场景”称为“recovery 标签子集”。它是另一组未见随机 seed，不是真正的在线扰动恢复 benchmark。

## 1. 结论

100k 是当前最佳 checkpoint。在无 EMA（`alpha=1.0`）下，普通场景为 **35/40（87.5%）**，recovery 标签子集为 **36/40（90%）**，两组未见随机场景合计 **71/80（88.75%）**；50k 在相同设置下为 62/80（77.5%），100k 净增 9 个成功回合。

逐 seed 配对结果为 11 个“50k 失败、100k 成功”，2 个“50k 成功、100k 失败”。合并 80 个配对回合做双侧精确 McNemar 检验，`p≈0.0225`；在本次固定测试集上，100k 的总体提升不是单纯由一两个场景造成。

100k 下 `alpha=1.0` 同时优于 `alpha=0.2` 的两个测试子集。当前不应继续把 `alpha=0.2` 作为默认部署参数。`alpha=1.0` 只关闭跨动作 EMA；每个 100 ms 控制周期内的线性插值仍然保留。

recovery 标签子集的 4 个失败全部带有 **yellow box + weak_grasp** 标签。这说明方块相关 seed 偏难，但不能证明失败是由未实际注入的 weak-grasp 事件导致。真正的 DAgger 优先级应由新的扰动注入评测确认。

## 2. 总体结果

| 场景 | alpha | 50k | 100k | 变化 |
|---|---:|---:|---:|---:|
| 普通 | 0.2 | 34/40（85%） | 32/40（80%） | -2（-5 pp） |
| 普通 | 1.0 | 32/40（80%） | 35/40（87.5%） | +3（+7.5 pp） |
| Recovery 标签 | 0.2 | 22/40（55%） | 28/40（70%） | +6（+15 pp） |
| Recovery 标签 | 1.0 | 30/40（75%） | 36/40（90%） | +6（+15 pp） |
| 两子集合计 | 0.2 | 56/80（70%） | 60/80（75%） | +4（+5 pp） |
| 两子集合计 | 1.0 | 62/80（77.5%） | 71/80（88.75%） | +9（+11.25 pp） |

## 3. 100k 按目标物体

| 场景/alpha | apple | box | can | orange |
|---|---:|---:|---:|---:|
| 普通 / 0.2 | 8/10 | 6/10 | 10/10 | 8/10 |
| 普通 / 1.0 | 9/10 | 7/10 | 10/10 | 9/10 |
| Recovery 标签 / 0.2 | 7/10 | 6/10 | 8/10 | 7/10 |
| Recovery 标签 / 1.0 | 10/10 | 6/10 | 10/10 | 10/10 |

在 100k、`alpha=1.0` 下合并两组未见场景：apple 19/20、box 13/20、can 20/20、orange 19/20。方块是唯一明显偏弱的目标物体。

## 4. Recovery 标签分组（未注入扰动）

| recovery_type | 50k / 0.2 | 100k / 0.2 | 50k / 1.0 | 100k / 1.0 |
|---|---:|---:|---:|---:|
| approach_offset | 8/10 | 8/10 | 10/10 | 10/10 |
| object_slip | 5/10 | 7/10 | 8/10 | 10/10 |
| place_offset | 5/10 | 7/10 | 5/10 | 10/10 |
| weak_grasp | 4/10 | 6/10 | 7/10 | 6/10 |

这些行只是按 metadata 分组的不同 seed，数字变化不能归因于对应 recovery 能力。四类训练扰动的代码定义、画面表现和应学能力见项目 [README 第 2.5 节](../README.md#25-四类-recovery-训练情况是什么)。

## 5. 逐 seed 配对结果

### 5.1 普通场景

- 匹配：40
- 失败→成功：4
- 成功→失败：1
- 两者均成功：31
- 两者均失败：4
- 净提升：3

100k 新增成功：

| seed | target | 50k steps | 100k steps |
|---:|---|---:|---:|
| 50008 | apple | 100（失败） | 31 |
| 50035 | box | 100（失败） | 20 |
| 50025 | orange | 100（失败） | 24 |
| 50029 | orange | 100（失败） | 22 |

100k 回退：seed 50005，orange；50k 在 60 steps 成功，100k 跑满 100 steps 失败。

### 5.2 Recovery 标签子集

- 匹配：40
- 失败→成功：7
- 成功→失败：1
- 两者均成功：29
- 两者均失败：3
- 净提升：6

100k 新增成功：

| seed | target | recovery | 100k steps |
|---:|---|---|---:|
| 50056 | apple | object_slip | 100 |
| 50076 | apple | object_slip | 27 |
| 50053 | orange | place_offset | 100 |
| 50069 | orange | place_offset | 19 |
| 50077 | orange | place_offset | 35 |
| 50085 | orange | place_offset | 20 |
| 50089 | orange | place_offset | 20 |

100k 回退：seed 50071，box + weak_grasp；50k 在 20 steps 成功，100k 跑满 100 steps 失败。

seed 50056 和 50053 恰好在 `max_steps=100` 时才成功，属于边界成功。应通过视频或使用稍长上限复核，但它们仍按预先设定的评测规则计为成功。

## 6. 完成效率

100k 成功回合的平均规划周期：

| 场景 | alpha=0.2 | alpha=1.0 | 变化 |
|---|---:|---:|---:|
| 普通 | 36.06 | 26.77 | -25.8% |
| Recovery 标签 | 35.61 | 33.86 | -4.9% |

无 EMA 不仅成功率更高，普通场景完成速度也更快。当前没有提供 50k 的同口径平均成功 steps，因此不对 checkpoint 间效率作无依据比较。

## 7. 下一步

1. 首先实现 runtime perturbation injector，对同一 seed 成对运行 nominal 和真实扰动回合。
2. 固定 100k 为当前主模型，保留 50k 作为基线；暂不继续训练到 150k。
3. 默认评测和部署先使用 `alpha=1.0`；若视频抖动不可接受，再只补测一个较弱平滑值（如 0.8）。
4. 根据真扰动失败集确定 DAgger 优先级；使用新训练 seed，不使用 50000–50099 冻结测试 seed。
5. DAgger 数据必须记录模型到达的状态和 teacher 的正确纠正动作，并与原始 nominal 数据混合，避免普通能力遗忘。
6. 文本改写鲁棒性作为独立实验进行，不与 checkpoint、平滑参数或 DAgger 同时改变。

## 8. 云端结果位置

- 50k 普通无 EMA：`/root/shared-nvme/logs/h50_formal_50k_no_smoothing/alpha_1.0/`
- 50k Recovery 无 EMA：`/root/shared-nvme/logs/h50_formal_50k_recovery/alpha_1.0/`
- 100k 完整矩阵：`/root/shared-nvme/logs/h50_formal_100k_matrix/`
- 50k checkpoint：`/root/shared-nvme/checkpoints/pi05_panda_h50_formal/panda_h50_formal_v1/50000`
- 100k checkpoint：`/root/shared-nvme/checkpoints/pi05_panda_h50_formal/panda_h50_formal_v1/100000`
