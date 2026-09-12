# AutoDL 服务器使用与 π0.5 训练教程

更新时间：2026-09-10

本教程对应正式上传包：

`/home/yuan/projects/pi05-finetune-lab/upload/pi05_panda_multi_object_box_v2`

数据包括 400 条 train、60 条 validation 和 100 个无教师动作 test scenes。模型输入为外部 RGB、腕部 RGB、8 维 state 和英文 task，输出为 8 维绝对关节目标 action。

官方参考：

- OpenPI 安装与训练：https://github.com/Physical-Intelligence/openpi
- 自定义机器人映射示例：https://github.com/Physical-Intelligence/openpi/tree/main/examples/ur5
- AutoDL SSH：https://www.autodl.com/docs/ssh/
- AutoDL 磁盘说明：https://www.autodl.com/docs/env/
- AutoDL 数据保留规则：https://www.autodl.com/docs/instance_data/

## 1. 按截图选择服务器

### 1.1 计费

选择“按量计费”。第一轮需要安装、调试和冒烟，不建议包年包月。

### 1.2 GPU 与主机

第三张截图中当前选中的第一行可以使用：

```text
1 × RTX4090 24GB
10 vCPU
60GB 内存
30GB 系统盘
山东一区
```

保持这一行即可。

不要选择 `vGPU-RTX4090`。前两张截图中的 vGPU 只有 2.4GB、6GB 或 12GB 显存，均不满足 OpenPI 官方给出的 LoRA 大于 22.5GB 显存要求。

也不需要选择两张 4090。单卡 LoRA 先跑通，双卡会提高费用并增加配置复杂度。

如果整卡 RTX5090 32GB 的价格只比 4090 略高，也可选择 5090；但需要 CUDA 12.8 镜像。第一次求稳，截图中这张整卡 4090 是更稳妥的官方示例配置。

### 1.3 镜像

选择最接近以下条件的开发镜像：

```text
Ubuntu 22.04
基础 Python 3.10（uv 将为 OpenPI 安装 Python 3.11）
CUDA 12.3.2
PyTorch 2.2.0（PyTorch-24.01-py3-20240415）
```

虽然基础镜像名字可能写 PyTorch，实际 LoRA 走 OpenPI 的 JAX 后端。OpenPI 官方当前说明：PyTorch 后端仍不支持 LoRA。

当前截图中的 PyTorch-24.01 镜像使用 CUDA 12.3.2，适合 RTX4090。不要选择 CUDA 11。

### 1.4 磁盘

系统盘 30GB 可以保留。代码、数据、依赖缓存和 checkpoint 全部放到 `/root/shared-nvme` 数据盘。

```text
数据盘最低：100GB
推荐：200GB
```

50GB 不建议。上传包已有约 6.29GB，基础模型、uv 环境、JAX 缓存和多个 checkpoint 还会占用大量空间。

### 1.5 创建前核对

```text
[ ] 按量计费
[ ] 1 × 整卡 RTX4090 24GB
[ ] 10 vCPU / 60GB 内存
[ ] PyTorch 2.2.0 / PyTorch-24.01-py3-20240415 / Ubuntu 22.04 / CUDA 12.3.2
[ ] 数据盘至少 100GB，推荐 200GB
```

确认后再创建实例。

## 2. 第一次登录与验收

实例进入“运行中”后，在 AutoDL 控制台复制 SSH 命令，格式类似：

```bash
ssh -p 12345 root@connect.example.com
```

在本机终端执行。密码输入时不会显示字符，这是正常的。

登录服务器后运行：

```bash
nvidia-smi
cat /etc/os-release
python3 --version
df -h
free -h
```

必须确认：

- GPU 是完整 RTX4090，显存约 24GB；
- Ubuntu 22.04；
- 内存约 60GB；
- `/root/shared-nvme` 至少有 100GB；
- `nvidia-smi` 正常。

如果显存显示 12GB 或更少，说明选错了 vGPU，应停止实例，不要继续花钱安装。

## 3. 上传正式数据

推荐 `rsync`，中断后可以续传。把 `<端口>` 和 `<主机>` 替换为控制台的值，在本机终端执行：

```bash
rsync -avP --partial \
  -e "ssh -p <端口>" \
  /home/yuan/projects/pi05-finetune-lab/upload/pi05_panda_multi_object_box_v2/ \
  root@<主机>:/root/shared-nvme/pi05_panda_multi_object_box_v2/
```

没有 `rsync` 时：

```bash
scp -P <端口> -r \
  /home/yuan/projects/pi05-finetune-lab/upload/pi05_panda_multi_object_box_v2 \
  root@<主机>:/root/shared-nvme/
```

上传完，在服务器运行：

```bash
cd /root/shared-nvme/pi05_panda_multi_object_box_v2
ls
du -sh .
```

应看到：

```text
dataset/
configs/
scripts/
reports/
test_scenes.json
SHA256SUMS.json
bundle_manifest.json
```

### 3.1 校验上传文件

在服务器执行：

```bash
cd /root/shared-nvme/pi05_panda_multi_object_box_v2
python3 - <<'VERIFY_UPLOAD'
import hashlib
import json
from pathlib import Path

root = Path('.').resolve()
records = json.loads((root / 'SHA256SUMS.json').read_text())
errors = []
for record in records:
    path = root / record['path']
    if not path.is_file():
        errors.append(f"missing: {record['path']}")
        continue
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != record['sha256']:
        errors.append(f"hash: {record['path']}")
print('checked:', len(records), 'errors:', errors)
raise SystemExit(bool(errors))
VERIFY_UPLOAD
```

正确结果：

```text
checked: 117 errors: []
```

出现任何错误都先重新上传对应文件，不要开始训练。

## 4. 安装官方 OpenPI

在服务器执行：

```bash
mkdir -p /root/shared-nvme/workspace
mkdir -p /root/shared-nvme/openpi-cache
mkdir -p /root/shared-nvme/uv-cache
mkdir -p /root/shared-nvme/huggingface
mkdir -p /root/shared-nvme/lerobot/local

export OPENPI_DATA_HOME=/root/shared-nvme/openpi-cache
export UV_CACHE_DIR=/root/shared-nvme/uv-cache
export HF_HOME=/root/shared-nvme/huggingface
export HF_LEROBOT_HOME=/root/shared-nvme/lerobot
```

国内网络慢时临时开启 AutoDL 学术加速：

```bash
source /etc/network_turbo
```

安装 uv：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=/root/.local/bin:$PATH
uv --version
```

克隆官方仓库并记录 commit：

```bash
cd /root/shared-nvme/workspace
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
cd openpi
git rev-parse HEAD | tee OPENPI_COMMIT_USED.txt
```

不要在适配和训练中途随意 `git pull`。

安装依赖：

```bash
cd /root/shared-nvme/workspace/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

检查 GPU：

```bash
uv run python -c "import jax; print(jax.devices())"
uv run python -c "import torch; print(torch.cuda.get_device_name(), torch.cuda.get_device_properties(0).total_memory / 1024**3)"
```

JAX 必须显示 `CudaDevice`。若只有 CPU，停止后续步骤并排查镜像/CUDA/JAX。

## 5. 注册本地 LeRobot 数据

服务器执行：

```bash
ln -s /root/shared-nvme/pi05_panda_multi_object_box_v2/dataset/train \
  /root/shared-nvme/lerobot/local/panda_multi_object_box_v2_train

ln -s /root/shared-nvme/pi05_panda_multi_object_box_v2/dataset/validation \
  /root/shared-nvme/lerobot/local/panda_multi_object_box_v2_validation
```

如果提示已存在，先检查：

```bash
ls -l /root/shared-nvme/lerobot/local/
```

不要直接删除不明目录。

读取数据测试：

```bash
cd /root/shared-nvme/workspace/openpi
export HF_LEROBOT_HOME=/root/shared-nvme/lerobot
uv run python - <<'CHECK_DATA'
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset('local/panda_multi_object_box_v2_train')
item = dataset[0]
print('episodes:', dataset.num_episodes)
print('frames:', len(dataset))
for key in (
    'observation.images.image',
    'observation.images.wrist_image',
    'observation.state',
    'action',
    'task',
):
    value = item[key]
    print(key, getattr(value, 'shape', value))
CHECK_DATA
```

期望：

```text
episodes: 400
frames: 94064
两路图像: 3×256×256
state: 8
动作: 8
```

## 6. 这一点最重要：不要直接跑 pi05_libero

官方 `pi05_libero` 配置针对 LIBERO，字段和动作语义都不同，而且是全量微调。直接运行既可能 OOM，也可能让模型学习错误动作。

需要针对服务器上实际克隆到的 OpenPI commit 新增：

```text
pi05_panda_multi_object_lora_smoke   # 100 steps
pi05_panda_multi_object_lora         # 正式训练
```

专用适配器必须满足：

- `observation.images.image` 映射到主相机；
- `observation.images.wrist_image` 映射到左腕相机；
- 未使用的第三相机填零，mask 为 false；
- state 为 8 维；
- action 为 8 维；
- `task` 作为 prompt，`prompt_from_task=True`；
- `action_dim=8`；
- `action_horizon=10`；
- `pi05=True`；
- 使用 LoRA model variants 及匹配的 freeze filter；
- `ema_decay=None`；
- 权重来自 `gs://openpi-assets/checkpoints/pi05_base/params`；
- 当前 action 是绝对关节目标，不能添加 LIBERO/UR5 的 delta-action 转换。

完成前五节后，把下面输出发给 Codex，再针对那个 commit 编写和验证适配器：

```bash
cd /root/shared-nvme/workspace/openpi
git rev-parse HEAD
nvidia-smi
uv run python -c "import jax; print(jax.devices())"
find /root/shared-nvme/pi05_panda_multi_object_box_v2 -maxdepth 2 -type f | head
```

## 7. 适配完成后的训练顺序

正确顺序：

```text
单 batch 加载检查
→ norm stats
→ 100-step LoRA 冒烟
→ 小数据约 1000-step 过拟合
→ 正式 LoRA
→ validation
→ 冻结 test 闭环评测
```

### 7.1 计算 norm stats

```bash
cd /root/shared-nvme/workspace/openpi
export OPENPI_DATA_HOME=/root/shared-nvme/openpi-cache
export HF_HOME=/root/shared-nvme/huggingface
export HF_LEROBOT_HOME=/root/shared-nvme/lerobot
uv run scripts/compute_norm_stats.py --config-name pi05_panda_multi_object_lora_smoke
```

确认 state/action 统计都是 8 维且无 NaN/Inf。

### 7.2 使用 tmux

```bash
tmux new -s pi05
```

退出但保持任务：先按 `Ctrl+B`，松开后按 `D`。

重新进入：

```bash
tmux attach -t pi05
```

### 7.3 100-step LoRA 冒烟

在 tmux 中执行：

```bash
cd /root/shared-nvme/workspace/openpi
mkdir -p /root/shared-nvme/train-logs
export OPENPI_DATA_HOME=/root/shared-nvme/openpi-cache
export HF_HOME=/root/shared-nvme/huggingface
export HF_LEROBOT_HOME=/root/shared-nvme/lerobot
export WANDB_MODE=offline
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

uv run scripts/train.py pi05_panda_multi_object_lora_smoke \
  --exp-name=smoke_100 2>&1 | \
  tee /root/shared-nvme/train-logs/smoke_100.log
```

另一 SSH 终端监控：

```bash
watch -n 1 nvidia-smi
```

冒烟通过标准：

- 不 OOM；
- loss 为有限值并能变化/下降；
- checkpoint 正常保存；
- 无 shape、mask、tokenizer、norm stats 错误。

4090 OOM 时依次处理：batch size 降到 1、确认启用 remat/gradient checkpointing、减少数据预取 worker。不要改用 12GB vGPU。

### 7.4 小数据过拟合

先取少量 episode 跑约 1000 steps。验证模型能明显降低 loss，并能在对应场景输出合理动作。失败则继续修数据映射，不要开始长训练。

### 7.5 正式训练

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_panda_multi_object_lora

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
uv run scripts/train.py pi05_panda_multi_object_lora \
  --exp-name=panda_multi_object_box_v2 2>&1 | \
  tee /root/shared-nvme/train-logs/panda_multi_object_box_v2.log
```

第一轮不要盲跑 30000 steps。先设置较短步数并每 1000 steps 保存，根据 validation 和本地闭环成功率决定是否继续。

## 8. 策略服务器与本地仿真

训练后在服务器运行：

```bash
cd /root/shared-nvme/workspace/openpi
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_panda_multi_object_lora \
  --policy.dir=/root/shared-nvme/workspace/openpi/checkpoints/pi05_panda_multi_object_lora/panda_multi_object_box_v2/<STEP> \
  --port=8000
```

在本机建立 SSH 隧道，不直接暴露公网端口：

```bash
ssh -N -L 8000:127.0.0.1:8000 -p <端口> root@<主机>
```

本地客户端连接 `ws://127.0.0.1:8000`。最终成绩必须使用冻结的 100 个 test scenes，不能使用 train seed。

## 9. 下载结果并停止计费

至少下载：

- 最佳 checkpoint；
- norm stats；
- 自定义 adapter/config；
- `OPENPI_COMMIT_USED.txt`；
- 训练日志；
- validation/test 结果。

从服务器拉回：

```bash
mkdir -p /home/yuan/projects/pi05-finetune-lab/checkpoints/cloud
rsync -avP --partial \
  -e "ssh -p <端口>" \
  root@<主机>:/root/shared-nvme/workspace/openpi/checkpoints/ \
  /home/yuan/projects/pi05-finetune-lab/checkpoints/cloud/
```

确认结果已下载后，必须在 AutoDL 控制台点击“关机”。关闭浏览器、SSH 或 JupyterLab 不会停止计费。

关机不等于释放。AutoDL 官方说明：按量实例连续关机 15 天可能自动释放，届时实例数据永久清空；云端本地盘不能当唯一备份。

## 10. 最短检查表

```text
[ ] 截图第一行：1×RTX4090 24GB、10 vCPU、60GB 内存
[ ] 数据盘 >=100GB
[ ] nvidia-smi 验收
[ ] rsync 上传，117 个文件哈希通过
[ ] 安装 OpenPI 并记录 commit
[ ] JAX 显示 CudaDevice
[ ] train/validation 数据可加载
[ ] 专用 8维绝对动作适配器完成
[ ] 单 batch 测试
[ ] norm stats
[ ] 100-step LoRA
[ ] 1000-step 小数据过拟合
[ ] 正式训练
[ ] validation 与冻结 test 闭环评测
[ ] 下载 checkpoint/日志/配置
[ ] 控制台关机
```
