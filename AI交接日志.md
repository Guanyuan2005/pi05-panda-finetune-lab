# AI 交接（2026-09-11）

目标：完成步骤 3/4/5（Panda 适配、LeRobot 数据注册/校验、norm stats），再给冒烟训练命令。不要直接开训。

已完成：
- 远端 OpenPI：`/root/shared-nvme/workspace/openpi`；commit `215abfb217dbac7d5f1273282331b9b1866c0479`。
- LeRobot 固定 commit `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`；JAX/PyTorch 已识别 RTX4090。
- 本机脚本：`openpi_overlay/panda_policy.py`、`openpi_overlay/apply_panda_config.py`、`scripts/create_lerobot_v21_compat.py`，均已上传至远端 `/root/shared-nvme/workspace/`。
- `panda_policy.py` 已复制进 OpenPI；配置已注册并通过 py_compile：
  - `pi05_panda_multi_object_lora_smoke`（100 steps）
  - `pi05_panda_multi_object_lora`（20000 steps）

当前状态：
- 兼容注册脚本已修复并成功生成：train 400 episodes/94064 frames，validation 60 episodes/14875 frames。
- norm stats 已生成：state/actions 各 8 维，有限值检查通过。
- 直接 OpenPI 工厂校验会触发远端 tokenizer 下载，当前网络失败/等待；未启动训练。
- 原始数据：`/root/shared-nvme/pi05_panda_multi_object_box_v2/dataset/{train,validation}`。
- 原始 `meta/pi05_episodes.jsonl` 有 episode 顺序但无 length；应扫描 parquet 的 `episode_index` 分组，生成 v2.0 `meta/episodes.jsonl`，每项含 `episode_index`、`tasks`、`length`。不要改原始数据。

下一步：
1. 先缓存 tokenizer/base weights，再运行 OpenPI 数据工厂样本校验：
   - `/root/shared-nvme/lerobot/local/panda_multi_object_box_v2_train_compat`
   - `/root/shared-nvme/lerobot/local/panda_multi_object_box_v2_validation_compat`
2. 核实 `scripts/train.py --help` 后给用户 smoke 命令与成功标准。

模型→MuJoCo 已打通：上传 `/root/shared-nvme/workspace/pi05-finetune-lab/scripts/run_mujoco_policy_eval.py` 及 `pi05_local`、`mujoco_arm_lab`；缓存 Panda menagerie 资产到 `/root/.cache/mujoco_menagerie`。新容器需 `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl`。1 回合/30 步 smoke 已完成，报告 `/root/shared-nvme/logs/mujoco_model_eval_smoke/summary.json`，success_rate=0（短测未成功，不代表最终指标）。

远端登录信息见用户消息；密码禁止写入文件或回复。
