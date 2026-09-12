# π0.5 Panda/MuJoCo Fine-tuning Lab

在 MuJoCo 中使用规划教师采集 Panda 多物体抓放数据，在单卡 GPU 上对 OpenPI π0.5 做 LoRA 微调，并用 receding-horizon 闭环控制评测普通与恢复场景。

任务指令模板：

```text
Pick up the {target} and place it in the green box.
```

目标物体包括 red apple、orange、blue can 和 yellow box。恢复数据包括接近偏移、弱抓、物体滑落和放置偏移。

## 当前结果

冻结测试集使用相同 seed 对 50k 与 100k checkpoint 做配对评测，每组包含 40 个普通场景和 40 个恢复场景。

| Checkpoint | EMA alpha | 普通 | Recovery | 合计 |
|---|---:|---:|---:|---:|
| 50k | 1.0 | 32/40（80%） | 30/40（75%） | 62/80（77.5%） |
| **100k** | **1.0** | **35/40（87.5%）** | **36/40（90%）** | **71/80（88.75%）** |

80 个同 seed 回合中，100k 有 11 个失败转成功、2 个成功转失败。剩余 4 个 recovery 失败全部集中在 `yellow box + weak_grasp`。

完整结果：[50k vs 100k 配对评测报告](./reports/h50_formal_50k_vs_100k_report.md)。

## 数据

- train：400 episodes / 94,064 frames
- 300 个普通回合
- 100 个恢复回合，四种 recovery 各 25 个
- validation：60 episodes / 14,875 frames
- frozen test：50 个普通场景 + 50 个恢复场景，不含 teacher action
- 输入：外部 RGB、腕部 RGB、8 维关节/夹爪 state、英文 prompt
- 输出：8 维绝对关节/夹爪目标，`action_horizon=50`

原始 LeRobot 数据约 18 GB，不存入 Git；验证结果和数据构造配置保留在仓库中。

## 仓库结构

```text
configs/          任务、相机和数据集配置
pi05_local/       MuJoCo 环境、教师和数据集实现
openpi_overlay/   OpenPI Panda 输入映射与训练配置注入
scripts/          采集、校验、转换、训练辅助和闭环评测脚本
tests/            本地流水线检查
reports/          实验报告、数据审计和阶段结论
artifacts/        精选评测 JSON、训练日志和关键视频
data/             冻结测试场景；大规模数据被忽略
```

## 环境与运行

本地仿真依赖 Python 3.11、MuJoCo、LeRobot 及同级目录中的 `mujoco_arm_lab`。OpenPI 训练使用上游 commit `215abfb217dbac7d5f1273282331b9b1866c0479`。

```bash
./scripts/setup_local_env.sh
```

采集并校验数据：

```bash
python scripts/collect_panda_apple.py \
  --split train --mode mixed --episodes 20 \
  --dataset-root data/staging/demo --headless

python scripts/validate_panda_dataset.py \
  --dataset-root data/staging/demo \
  --report-md reports/demo-validation.md
```

100k checkpoint 的闭环评测示例：

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
python scripts/run_mujoco_policy_eval_receding.py \
  --checkpoint /path/to/panda_h50_formal_v1/100000 \
  --config pi05_panda_h50_formal \
  --scenes data/test_scenes.json \
  --output reports/model_eval \
  --episodes 40 --max-steps 100 \
  --execute-actions 10 --smooth-alpha 1.0 \
  --post-success-cycles 5
```

## 文档入口

- [微调复现与部署指南](./π0.5微调复现与部署指南.md)
- [云端训练工作日志](./云端训练工作日志.md)
- [本地实现与验证说明](./本地实现与验证说明.md)
- [数据集校验报告](./reports/dataset_validation.md)
- [实验 artifacts 说明](./artifacts/README.md)

## 不包含的文件

Git 仓库不保存虚拟环境、缓存、原始数据集或 Orbax checkpoint。当前单个 checkpoint 约 8.9 GB，应使用对象存储或 GitHub Release 单独发布，而不是普通 Git commit。
