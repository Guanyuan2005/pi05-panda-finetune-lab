from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import environment as env
from .teacher import EpisodeResult, run_episode


def safe_remove_tree(path: Path) -> None:
    resolved = path.resolve()
    allowed = [env.HERE.resolve(), Path("/tmp").resolve()]
    if not any(resolved == root or root in resolved.parents for root in allowed):
        raise ValueError(f"拒绝删除工作区之外的路径: {resolved}")
    if resolved in allowed:
        raise ValueError(f"拒绝删除过宽路径: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def dataset_features(image_storage: str = "image") -> dict[str, dict[str, Any]]:
    if image_storage not in {"image", "video"}:
        raise ValueError("image_storage 必须是 image 或 video")
    return {
        "observation.images.image": {
            "dtype": image_storage,
            "shape": (3, 256, 256),
            "names": ["channels", "height", "width"],
        },
        "observation.images.wrist_image": {
            "dtype": image_storage,
            "shape": (3, 256, 256),
            "names": ["channels", "height", "width"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": [f"joint{i}" for i in range(1, 8)] + ["gripper"],
        },
        "action": {
            "dtype": "float32",
            "shape": (8,),
            "names": [f"joint_target{i}" for i in range(1, 8)] + ["gripper_target"],
        },
    }


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def read_manifest(root: Path) -> dict[str, Any]:
    with (root/"meta"/"pi05_manifest.json").open(encoding="utf-8") as file:
        return json.load(file)


def choose_recovery(mode: str, ordinal: int) -> str:
    if mode == "nominal":
        return "none"
    if mode == "recovery":
        return env.RECOVERY_TYPES[ordinal % len(env.RECOVERY_TYPES)]
    if mode == "mixed":
        return "none" if ordinal % 4 else env.RECOVERY_TYPES[(ordinal//4) % len(env.RECOVERY_TYPES)]
    raise ValueError("mode 必须是 nominal、recovery 或 mixed")


def create_lerobot_dataset(root: Path, repo_id: str, image_storage: str):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    if image_storage == "video":
        import importlib.util
        import torchvision

        if importlib.util.find_spec("torchcodec") is None and not hasattr(torchvision.io, "VideoReader"):
            raise RuntimeError(
                "Video decoding is unavailable with this torchvision build. "
                "Use the verified --image-storage image mode or install compatible torchcodec."
            )

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=env.FPS,
        features=dataset_features(image_storage),
        root=root,
        robot_type="franka_panda_mujoco",
        use_videos=image_storage == "video",
        # PyAV is part of the verified local environment. Passing it explicitly
        # also avoids LeRobot probing for optional torchcodec for image datasets.
        video_backend="pyav",
        image_writer_threads=2 if image_storage == "image" else 0,
    )


def collect_shard(
    root: Path,
    split: str,
    requested_successes: int,
    seed_start: int,
    seed_stride: int = 1,
    worker_id: int = 0,
    mode: str = "mixed",
    teacher_source: str = "privileged",
    image_storage: str = "image",
    overwrite: bool = False,
    max_attempt_multiplier: int = 4,
    initial_joint_noise: float = 0.0,
) -> dict[str, Any]:
    root = root.resolve()
    if root.exists():
        if not overwrite:
            raise FileExistsError(f"输出目录已存在: {root}; 使用 --overwrite 明确覆盖")
        safe_remove_tree(root)
    repo_id = f"local/panda_multi_object_box_v2_{split}_worker_{worker_id:02d}"
    dataset = create_lerobot_dataset(root, repo_id, image_storage)
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started = time.time()
    attempt = 0
    max_attempts = max(requested_successes*max_attempt_multiplier, requested_successes)
    diagnostic_dir = root/"diagnostics"
    try:
        while len(successes) < requested_successes and attempt < max_attempts:
            seed = seed_start + attempt*seed_stride
            global_success_ordinal = worker_id + len(successes)*seed_stride
            recovery_type = choose_recovery(mode, global_success_ordinal)
            spec = env.make_task_spec(seed, recovery_type, split)
            result = run_episode(
                spec,
                teacher_source=teacher_source,
                record_images=True,
                diagnostic_dir=diagnostic_dir if teacher_source == "vision" else None,
                initial_joint_noise=initial_joint_noise,
            )
            record = dict(result.metadata)
            record["worker_id"] = worker_id
            record["attempt_index"] = attempt
            if result.success:
                record["episode_index"] = len(successes)
                for frame in result.frames:
                    dataset.add_frame(frame)
                dataset.save_episode()
                successes.append(record)
                print(
                    f"worker={worker_id} episode={len(successes)}/{requested_successes} "
                    f"seed={seed} recovery={recovery_type} frames={len(result.frames)}"
                )
            else:
                failures.append(record)
                print(f"worker={worker_id} seed={seed} FAILED: {record['failure_reason']}")
            attempt += 1
    finally:
        pass

    elapsed = time.time()-started
    for record in successes:
        append_jsonl(root/"meta"/"pi05_episodes.jsonl", record)
    for record in failures:
        append_jsonl(root/"meta"/"pi05_failures.jsonl", record)
    manifest = {
        "format": "LeRobotDataset-v3",
        "repo_id": repo_id,
        "split": split,
        "fps": env.FPS,
        "image_storage": image_storage,
        "teacher_version": env.TEACHER_VERSION,
        "camera_config_version": env.CAMERA_VERSION,
        "teacher_source": teacher_source,
        "worker_id": worker_id,
        "seed_start": seed_start,
        "seed_stride": seed_stride,
        "requested_successes": requested_successes,
        "successful_episodes": len(successes),
        "failed_attempts": len(failures),
        "elapsed_seconds": elapsed,
        "episodes_per_hour": len(successes)*3600/max(elapsed, 1e-6),
    }
    write_json(root/"meta"/"pi05_manifest.json", manifest)
    if len(successes) < requested_successes:
        raise RuntimeError(
            f"只采到 {len(successes)}/{requested_successes} 个成功回合; 失败 {len(failures)}"
        )
    return manifest


def tensor_image_to_uint8(value: Any) -> np.ndarray:
    array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    if array.shape[0] in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array*255.0, 0, 255)
    return array.astype(np.uint8)


def item_to_frame(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "observation.images.image": tensor_image_to_uint8(item["observation.images.image"]),
        "observation.images.wrist_image": tensor_image_to_uint8(item["observation.images.wrist_image"]),
        "observation.state": item["observation.state"].detach().cpu().numpy().astype(np.float32),
        "action": item["action"].detach().cpu().numpy().astype(np.float32),
        "task": item["task"],
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_checksums(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        records.append({
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return records
