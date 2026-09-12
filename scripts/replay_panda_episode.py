#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import av
import mujoco
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from pi05_local import environment as env
from pi05_local.dataset import read_jsonl, read_manifest, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay one Panda LeRobot episode without the teacher.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def resolve_root(root: Path, split: str) -> Path:
    return root if (root/"meta"/"info.json").exists() else root/split


def main() -> None:
    args = parse_args()
    root = resolve_root(args.dataset_root.resolve(), args.split)
    records = read_jsonl(root/"meta"/"pi05_episodes.jsonl")
    record = next((item for item in records if int(item["episode_index"]) == args.episode), None)
    if record is None:
        raise IndexError(f"episode {args.episode} 不存在")
    spec = env.TaskSpec(
        seed=int(record["scene_seed"]),
        positions=record["positions"],
        tray_xy=record["tray_xy"],
        target_type=record.get("target_type", "apple"),
        recovery_type=record["recovery_type"],
        split=record["split"],
    )
    model, data, ids = env.build_task_model(spec)
    renderer = env.RenderPair(model, size=512)
    manifest = read_manifest(root)
    dataset = LeRobotDataset(
        manifest["repo_id"], root=root, episodes=[args.episode], video_backend="pyav"
    )
    events = {int(event["frame"]): event for event in record.get("events", []) if event["type"] == "object_slip"}
    physics_steps = int(round(env.DT/model.opt.timestep))
    output = args.output or PROJECT/"reports"/"sample_videos"/f"{args.split}_episode_{args.episode:04d}.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(output), mode="w")
    stream = container.add_stream("libx264", rate=env.FPS)
    stream.width = 1024
    stream.height = 512
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "16", "preset": "medium"}
    try:
        for local_index in range(len(dataset)):
            if local_index in events:
                env.set_apple_pose(model, data, ids, np.asarray(events[local_index]["apple_xyz"], dtype=float))
            item = dataset[local_index]
            action = item["action"].detach().cpu().numpy()
            external, wrist = renderer.render(data)
            video_frame = av.VideoFrame.from_ndarray(np.concatenate([external, wrist], axis=1), format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
            data.ctrl[ids.arm_actuator] = action[:7]
            data.ctrl[ids.gripper_actuator] = env.actuator_gripper(model, ids, float(action[7]))
            for _ in range(physics_steps):
                mujoco.mj_step(model, data)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        renderer.close()
        container.close()
    success = env.apple_in_tray(data, ids, np.asarray(spec.tray_xy))
    result = {
        "episode": args.episode,
        "expected_success": bool(record["success"]),
        "replay_success": success,
        "final_apple_xyz": data.xpos[ids.apple_body].astype(float).tolist(),
        "video": output.resolve().as_posix(),
    }
    write_json(output.with_suffix(".json"), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if success != bool(record["success"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
