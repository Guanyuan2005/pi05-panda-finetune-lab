#!/usr/bin/env python3
"""Run a trained OpenPI Panda policy inside the existing MuJoCo task."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import av
import flax.nnx as nnx
import jax
import mujoco
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from openpi.policies import policy_config
from openpi.policies import policy as policy_lib
from openpi import transforms
from openpi.shared import nnx_utils
from openpi.training import config as training_config
from openpi.training import checkpoints as checkpoint_lib
from openpi.training import weight_loaders
from openpi.models import model as model_lib
from pi05_local import environment as env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-params", type=Path, default=None)
    parser.add_argument("--config", default="pi05_panda_multi_object_lora")
    parser.add_argument("--scenes", type=Path, default=PROJECT / "data/test_scenes.json")
    parser.add_argument("--output", type=Path, default=PROJECT / "reports/model_eval")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--execute-actions", type=int, default=1, help="每次观测后执行预测序列前 N 个动作")
    parser.add_argument("--smooth-alpha", type=float, default=1.0, help="动作目标 EMA 系数，1 表示不平滑")
    parser.add_argument("--post-success-cycles", type=int, default=0, help="成功检测后继续录制的控制周期数")
    return parser.parse_args()


def create_base_policy(config, params_path: Path):
    model = config.model.create(jax.random.key(0))
    graphdef, state = nnx.split(model)
    loaded = weight_loaders.CheckpointWeightLoader(str(params_path)).load(state.to_pure_dict())
    state.replace_by_pure_dict(loaded)
    model = nnx.merge(graphdef, state)
    data_config = config.data.create(config.assets_dirs, config.model)
    norm_stats = checkpoint_lib.load_norm_stats(config.assets_dirs, data_config.asset_id)
    return policy_lib.Policy(
        model,
        transforms=[
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ],
        metadata=config.policy_metadata,
    )


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    payload = json.loads(args.scenes.read_text(encoding="utf-8"))
    scenes = payload["scenes"][: args.episodes]

    config = training_config.get_config(args.config)
    policy = (
        create_base_policy(config, args.base_params)
        if args.base_params is not None
        else policy_config.create_trained_policy(config, str(args.checkpoint))
    )
    results = []

    for index, scene in enumerate(scenes):
        spec = env.TaskSpec(**scene)
        model, data, ids = env.build_task_model(spec)
        renderer = env.RenderPair(model, size=224)
        output = args.output / f"episode_{index:03d}.mp4"
        container = av.open(str(output), mode="w")
        stream = container.add_stream("libx264", rate=env.FPS)
        stream.width = 448
        stream.height = 224
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "medium"}
        physics_steps = int(round(env.DT / model.opt.timestep))
        success = False
        infer_ms = []
        smoothed_target = None
        success_ever = False
        success_cycles = 0

        step_index = -1
        try:
            for step_index in range(args.max_steps):
                external, wrist = renderer.render(data)
                frame = av.VideoFrame.from_ndarray(
                    np.concatenate([external, wrist], axis=1), format="rgb24"
                )
                for packet in stream.encode(frame):
                    container.mux(packet)

                state = np.concatenate(
                    [data.qpos[ids.arm_qpos], [env.normalized_gripper_state(data, ids)]]
                ).astype(np.float32)
                observation = {
                    "observation/image": external,
                    "observation/wrist_image": wrist,
                    "observation/state": state,
                    "prompt": env.task_prompt(spec),
                }
                prediction = policy.infer(observation)
                infer_ms.append(float(prediction.get("policy_timing", {}).get("infer_ms", 0.0)))
                action_seq = np.asarray(prediction["actions"])
                if action_seq.ndim == 3:
                    action_seq = action_seq[0]
                action_seq = action_seq[..., :8]
                n_exec = min(max(1, args.execute_actions), len(action_seq))
                for action in action_seq[:n_exec]:
                    raw_target = np.asarray(action[:8], dtype=float)
                    if smoothed_target is None:
                        smoothed_target = np.concatenate([data.qpos[ids.arm_qpos], [env.normalized_gripper_state(data, ids)]])
                    alpha = float(np.clip(args.smooth_alpha, 0.0, 1.0))
                    target = alpha * raw_target + (1.0 - alpha) * smoothed_target
                    start_target = smoothed_target.copy()
                    for interp_step in range(physics_steps):
                        frac = (interp_step + 1) / physics_steps
                        ctrl_target = start_target + frac * (target - start_target)
                        data.ctrl[ids.arm_actuator] = ctrl_target[:7]
                        data.ctrl[ids.gripper_actuator] = env.actuator_gripper(model, ids, float(ctrl_target[7]))
                        mujoco.mj_step(model, data)
                    smoothed_target = target
                    # Record every executed 100 ms action; otherwise a chunk of N actions
                    # would be compressed into one video frame.
                    frame_ext, frame_wrist = renderer.render(data)
                    action_frame = av.VideoFrame.from_ndarray(
                        np.concatenate([frame_ext, frame_wrist], axis=1), format="rgb24"
                    )
                    for packet in stream.encode(action_frame):
                        container.mux(packet)
                    success = bool(env.apple_in_tray(data, ids, np.asarray(spec.tray_xy)))
                    if success:
                        success_ever = True
                if success:
                    success_ever = True
                if success_ever:
                    success_cycles += 1
                    if success_cycles > args.post_success_cycles:
                        break
            success = success_ever
            for packet in stream.encode():
                container.mux(packet)
        finally:
            renderer.close()
            container.close()

        result = {
            "episode": index,
            "seed": spec.seed,
            "target_type": spec.target_type,
            "recovery_type": spec.recovery_type,
            "success": success,
            "steps": step_index + 1,
            "execute_actions": args.execute_actions,
            "smooth_alpha": args.smooth_alpha,
            "post_success_cycles": args.post_success_cycles,
            "mean_infer_ms": float(np.mean(infer_ms)) if infer_ms else None,
            "video": output.resolve().as_posix(),
        }
        (args.output / f"episode_{index:03d}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        results.append(result)
        print(json.dumps(result, ensure_ascii=False))

    summary = {
        "config": args.config,
        "checkpoint": str(args.checkpoint),
        "episodes": len(results),
        "successes": sum(item["success"] for item in results),
        "success_rate": sum(item["success"] for item in results) / max(len(results), 1),
        "results": results,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
