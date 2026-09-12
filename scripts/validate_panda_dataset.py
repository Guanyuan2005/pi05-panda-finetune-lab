#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from pi05_local import environment as env
from pi05_local.dataset import read_jsonl, read_manifest, write_json

REQUIRED = {"observation.images.image", "observation.images.wrist_image", "observation.state", "action"}
PANDA_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
PANDA_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Panda apple LeRobot datasets and shards.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset-root", type=Path)
    group.add_argument("--shards-root", type=Path)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--report-md", type=Path)
    parser.add_argument("--max-action-jump", type=float, default=0.20)
    return parser.parse_args()


def discover_dataset_roots(root: Path) -> list[Path]:
    if (root/"meta"/"info.json").exists():
        return [root]
    return sorted(path.parent.parent for path in root.rglob("meta/info.json"))


def scalar(value) -> int | float:
    if hasattr(value, "item"):
        return value.item()
    return value


def validate_one(root: Path, max_jump: float) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    manifest = read_manifest(root)
    dataset = LeRobotDataset(manifest["repo_id"], root=root, video_backend="pyav")
    missing = REQUIRED-set(dataset.features)
    if missing:
        errors.append(f"缺少字段: {sorted(missing)}")
    expected_shapes = {
        "observation.state": (8,),
        "action": (8,),
        "observation.images.image": (3, 256, 256),
        "observation.images.wrist_image": (3, 256, 256),
    }
    for key, shape in expected_shapes.items():
        actual = tuple(dataset.features.get(key, {}).get("shape", ()))
        if actual != shape:
            errors.append(f"{key} schema shape={actual}, expected={shape}")

    episodes = read_jsonl(root/"meta"/"pi05_episodes.jsonl")
    if len(episodes) != dataset.num_episodes:
        errors.append(f"sidecar episodes={len(episodes)} != LeRobot episodes={dataset.num_episodes}")
    seeds = [int(record["scene_seed"]) for record in episodes]
    if len(seeds) != len(set(seeds)):
        errors.append("scene_seed 重复")
    if any(not record.get("success", False) for record in episodes):
        errors.append("训练数据 sidecar 中包含失败 episode")

    if int(manifest.get("successful_episodes", len(episodes))) != len(episodes):
        errors.append("manifest successful_episodes does not match sidecar")
    for record in episodes:
        recovery_type = record.get("recovery_type", "none")
        if recovery_type != "none":
            event_types = {event.get("type") for event in record.get("events", [])}
            if recovery_type not in event_types:
                errors.append(f"seed {record.get('scene_seed')} missing recovery event {recovery_type}")
            if int(record.get("retry_count", 0)) < 1:
                errors.append(f"seed {record.get('scene_seed')} recovery episode has no retry")
        if any(float(plan.get("collision_count", 0)) != 0 for plan in record.get("plans", [])):
            errors.append(f"seed {record.get('scene_seed')} has a colliding saved plan")

    frame_counts: Counter[int] = Counter()
    previous_action: dict[int, np.ndarray] = {}
    max_seen_jump = 0.0
    image_means = defaultdict(list)
    prompts = set()
    for index in range(len(dataset)):
        item = dataset[index]
        episode_index = int(scalar(item["episode_index"]))
        frame_index = int(scalar(item["frame_index"]))
        if frame_index != frame_counts[episode_index]:
            errors.append(f"episode {episode_index} frame_index 不连续: {frame_index}")
        frame_counts[episode_index] += 1
        state = item["observation.state"].detach().cpu().numpy()
        action = item["action"].detach().cpu().numpy()
        if state.shape != (8,) or action.shape != (8,):
            errors.append(f"frame {index} state/action shape 错误")
            continue
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            errors.append(f"frame {index} 含 NaN/Inf")
        if np.any(action[:7] < PANDA_LOWER-1e-4) or np.any(action[:7] > PANDA_UPPER+1e-4):
            errors.append(f"frame {index} 关节目标越界")
        if not 0.0 <= float(action[7]) <= 1.0:
            errors.append(f"frame {index} 夹爪动作不在 [0,1]")
        if episode_index in previous_action:
            jump = float(np.max(np.abs(action-previous_action[episode_index])))
            max_seen_jump = max(max_seen_jump, jump)
            if jump > max_jump:
                errors.append(f"episode {episode_index} frame {frame_index} action jump={jump:.4f}")
        previous_action[episode_index] = action
        prompts.add(item["task"])
        for key in ("observation.images.image", "observation.images.wrist_image"):
            image = item[key]
            if tuple(image.shape) != (3, 256, 256):
                errors.append(f"frame {index} {key} 解码 shape={tuple(image.shape)}")
            image_means[key].append(float(image.mean()))
    expected_prompts = {
        env.task_config()["task"].format(target=env.TARGET_PROFILES[row.get("target_type", "apple")]["label"])
        for row in episodes
    }
    if prompts != expected_prompts:
        errors.append(f"prompt 不一致: actual={sorted(prompts)}, expected={sorted(expected_prompts)}")
    for key, values in image_means.items():
        if values and float(np.std(values)) < 1e-5:
            warnings.append(f"{key} 所有帧均值几乎不变，请人工检查相机")
        if values and (min(values) < 0.005 or max(values) > 0.995):
            warnings.append(f"{key} 存在接近全黑/全白帧")

    recovery_counts = Counter(record.get("recovery_type", "none") for record in episodes)
    return {
        "root": root.as_posix(),
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "episodes": dataset.num_episodes,
        "frames": len(dataset),
        "scene_seeds": seeds,
        "recovery_counts": dict(recovery_counts),
        "max_action_jump": max_seen_jump,
        "mean_episode_frames": float(np.mean(list(frame_counts.values()))) if frame_counts else 0.0,
        "image_mean_ranges": {key: [min(values), max(values)] for key, values in image_means.items() if values},
    }


def markdown_report(report: dict) -> str:
    lines = ["# Panda 数据集校验报告", "", f"总体结果：{'通过' if report['ok'] else '失败'}", ""]
    for item in report["datasets"]:
        lines += [f"## {item['root']}", "", f"- episodes: {item['episodes']}", f"- frames: {item['frames']}", f"- max action jump: {item['max_action_jump']:.6f}", f"- recovery counts: `{item['recovery_counts']}`", ""]
        if item["errors"]:
            lines += ["错误：", ""] + [f"- {error}" for error in item["errors"]] + [""]
        if item["warnings"]:
            lines += ["警告：", ""] + [f"- {warning}" for warning in item["warnings"]] + [""]
    if report.get("test_scenes"):
        test = report["test_scenes"]
        lines += [
            "## Held-out test scenes",
            "",
            f"- scenes: {test['scenes']}",
            f"- unique seeds: {test['unique_seeds']}",
            "",
        ]
    if report["cross_split_errors"]:
        lines += ["## 跨划分错误", ""] + [f"- {error}" for error in report["cross_split_errors"]] + [""]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root = (args.dataset_root or args.shards_root).resolve()
    roots = discover_dataset_roots(root)
    if not roots:
        raise FileNotFoundError(f"未找到 LeRobot meta/info.json: {root}")
    datasets = [validate_one(path, args.max_action_jump) for path in roots]
    cross_errors = []
    by_split = defaultdict(set)
    for item in datasets:
        split = read_manifest(Path(item["root"]))["split"]
        overlap = by_split[split].intersection(item["scene_seeds"])
        if overlap:
            cross_errors.append(f"{split} 内分片 seed 重复: {sorted(overlap)}")
        by_split[split].update(item["scene_seeds"])
    split_names = sorted(by_split)
    for i, first in enumerate(split_names):
        for second in split_names[i+1:]:
            overlap = by_split[first] & by_split[second]
            if overlap:
                cross_errors.append(f"{first}/{second} seed 交叉: {sorted(overlap)}")
    test_summary = None
    test_path = PROJECT/"data"/"test_scenes.json"
    if test_path.exists():
        test_payload = json.loads(test_path.read_text(encoding="utf-8"))
        test_seeds = [int(scene["seed"]) for scene in test_payload.get("scenes", [])]
        if len(test_seeds) != len(set(test_seeds)):
            cross_errors.append("duplicate seeds in test_scenes.json")
        test_set = set(test_seeds)
        for split, seeds in by_split.items():
            overlap = seeds & test_set
            if overlap:
                cross_errors.append(f"{split}/test seed overlap: {sorted(overlap)}")
        test_summary = {
            "path": test_path.as_posix(),
            "scenes": len(test_seeds),
            "unique_seeds": len(test_set),
        }
    report = {
        "ok": all(item["ok"] for item in datasets) and not cross_errors,
        "datasets": datasets,
        "test_scenes": test_summary,
        "cross_split_errors": cross_errors,
    }
    json_path = args.report_json or PROJECT/"reports"/"dataset_validation.json"
    md_path = args.report_md or PROJECT/"reports"/"dataset_validation.md"
    write_json(json_path, report)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
