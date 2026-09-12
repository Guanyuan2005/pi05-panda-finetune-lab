# π0.5 Panda 项目当前工作日志与 CLI 交接指南

## 2026-09-10 正式数据就绪状态（覆盖下方早期计划）

当前任务已升级为 `panda_multi_object_box_v2`：Panda 根据英文指令抓取苹果、橙子、蓝色罐或黄色盒子，并放入绿色箱子。

- train：400 episodes（300 正常；四类恢复各 25），94,064 frames，完整校验通过；
- validation：60 episodes（40 正常；四类恢复各 5），14,875 frames，完整校验通过；
- test：100 个冻结场景（50 正常、50 恢复），seed 50000–50099，不含教师动作；
- train / validation / test seed 无交叉；
- 正式动作回放 5/5 成功，60-episode 抽样审计通过；
- 正式数据根目录：`data/lerobot/panda_multi_object_box_v2`；
- 最终上传目录：`upload/pi05_panda_multi_object_box_v2`。

下方旧的 apple/tray v1 内容保留为实现演进记录；执行命令和数据版本以本节、`configs/dataset_v1.yaml` 及最终 readiness 报告为准。


最后更新：2026-09-10

## 1. 项目目标

用 MuJoCo 中的 Franka Panda 自动生成 LeRobot 教师数据，训练 π0.5 完成：

> 抓取红色苹果并放入绿色托盘；发生抓偏、夹持不足、滑落或放偏时完成纠错。

项目目录：`/home/yuan/projects/pi05-finetune-lab`

## 2. 当前已完成状态

- 场景只保留 1 个随机干扰物，不使用额外固定水杯。
- 苹果显示网格为 642 顶点、1280 面的闭合平滑模型。
- 苹果高模只负责显示；居中球体负责 65 g 质量、惯量和碰撞。
- 苹果具有滚动阻力与自由关节阻尼；扰动速度测试在 5 秒内由 8.0025 衰减到 0。
- 正常及四类纠错教师视频最终全部成功。
- 数据集动作回放 `dataset_episode_0000.mp4` 最终成功。
- 六个样例视频均为双相机 1024×512、H.264、CRF 16。

样例视频目录：`reports/sample_videos/`

视频含义：

- `normal.mp4`：一次正常抓放。
- `approach_offset.mp4`：首次接近偏移，重新定位后成功。
- `weak_grasp.mp4`：首次夹持不足，重新抓取后成功。
- `object_slip.mp4`：搬运前模拟滑落，重新定位后成功。
- `place_offset.mp4`：首次放到托盘外，重新抓放后成功。
- `dataset_episode_0000.mp4`：读取 LeRobot 数据动作并独立回放成功。

## 3. 关键文件

- `pi05_local/environment.py`：场景、相机、规划、物理与成功判定。
- `pi05_local/scene_overrides.py`：pi05 专用场景简化和苹果物理参数。
- `pi05_local/teacher.py`：正常与四类纠错教师状态机。
- `pi05_local/dataset.py`：LeRobot 数据集创建、采集和元数据。
- `scripts/render_sample_videos.py`：生成五类教师样例视频。
- `scripts/replay_panda_episode.py`：独立回放数据集动作。
- `scripts/collect_panda_apple.py`：单进程或多进程批量采集。
- `scripts/merge_panda_shards.py`：合并 worker 分片。
- `scripts/validate_panda_dataset.py`：校验数据集。
- `configs/panda_apple_task.yaml`：任务、干扰物数量和轨迹参数。

注意：pi05 会读取相邻项目 `../mujoco_arm_lab` 中的 Panda 和苹果资产，不要移动这两个项目的相对位置。

## 4. 数据组成原则

正式训练集固定使用：

- 75% 正常成功回合。
- 25% 纠错成功回合。
- 四类纠错均分，各占总数据约 6.25%。
- 最终失败回合不得进入监督训练集，只保存在失败诊断元数据中。

当前旧数据集 `data/lerobot/panda_multi_object_box_smoke` 使用修改前的苹果动力学，只作历史对照。
当前成功回放样本来自 `data/lerobot/panda_multi_object_box_smoke_v2`。

## 5. 运行环境

当前可用 Python：

```bash
cd /home/yuan/projects/pi05-finetune-lab
../mujoco_arm_lab/.venv/bin/python --version
```

后续命令均使用 `../mujoco_arm_lab/.venv/bin/python`，不要假设 pi05 目录内存在 `.venv`。

## 6. 新 CLI 开始工作前必须执行

```bash
cd /home/yuan/projects/pi05-finetune-lab
sed -n '1,260p' CURRENT_WORK_LOG_AND_HANDOFF.md
git status --short
```

要求新 CLI：

1. 只修改 `pi05-finetune-lab`，除非用户明确要求修改其他项目。
2. 保留现有未提交文件，不要执行 `git reset --hard` 或覆盖旧数据集。
3. 修改物理、教师策略或相机后，必须重新生成五类视频并验证全部最终成功。
4. 修改动力学后，旧 action 回放可能不再一致，应新建版本化数据集，不要覆盖旧数据。

## 7. 下一步：批量生成教师数据

先采集 20 条冒烟数据：

```bash
../mujoco_arm_lab/.venv/bin/python scripts/collect_panda_apple.py \
  --split train --mode mixed --episodes 20 --seed-start 1000 \
  --dataset-root data/lerobot/panda_apple_smoke_v3 \
  --image-storage image --headless
```

人工抽查后，正式采集 300 条正常和 100 条纠错：

```bash
../mujoco_arm_lab/.venv/bin/python scripts/collect_panda_apple.py \
  --split train --mode nominal --episodes 300 --seed-start 10000 \
  --workers 4 --staging-root data/staging/train/nominal --headless

../mujoco_arm_lab/.venv/bin/python scripts/collect_panda_apple.py \
  --split train --mode recovery --episodes 100 --seed-start 20000 \
  --workers 4 --staging-root data/staging/train/recovery --headless
```

验证集采集 45 条正常和 15 条纠错：

```bash
../mujoco_arm_lab/.venv/bin/python scripts/collect_panda_apple.py \
  --split validation --mode nominal --episodes 45 --seed-start 30000 \
  --workers 4 --staging-root data/staging/validation/nominal --headless

../mujoco_arm_lab/.venv/bin/python scripts/collect_panda_apple.py \
  --split validation --mode recovery --episodes 15 --seed-start 40000 \
  --workers 4 --staging-root data/staging/validation/recovery --headless
```

合并并校验：

```bash
../mujoco_arm_lab/.venv/bin/python scripts/merge_panda_shards.py \
  --shards-root data/staging \
  --output-root data/lerobot/panda_multi_object_box_v2

../mujoco_arm_lab/.venv/bin/python scripts/validate_panda_dataset.py \
  --dataset-root data/lerobot/panda_multi_object_box_v2
```

## 8. 修改后的回归检查

重新生成教师视频：

```bash
../mujoco_arm_lab/.venv/bin/python scripts/render_sample_videos.py
```

检查 `reports/sample_videos/index.json`，五项 `success` 必须全部为 `true`。

回放一个数据集回合：

```bash
../mujoco_arm_lab/.venv/bin/python scripts/replay_panda_episode.py \
  --dataset-root data/lerobot/panda_multi_object_box_smoke_v2 \
  --episode 0 \
  --output reports/sample_videos/dataset_episode_0000.mp4
```

输出 JSON 中 `expected_success` 和 `replay_success` 必须同时为 `true`。

## 9. 当前建议的下一动作

先执行第 7 节的 20 条冒烟采集，不要直接生成全部 400 条。抽查正常/纠错比例、视频观感、苹果稳定性和失败日志后，再开始正式四 worker 采集。
