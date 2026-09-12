from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import mujoco
import numpy as np

from . import environment as env


@dataclass
class EpisodeResult:
    success: bool
    frames: list[dict[str, Any]]
    metadata: dict[str, Any]


def episode_metrics(frames: list[dict[str, Any]], plan_metrics: list[dict[str, float]]) -> dict[str, float]:
    if not frames:
        return {"duration_s": 0.0, "joint_path_length": 0.0, "max_action_jump": 0.0, "mean_jerk": 0.0}
    actions = np.asarray([frame["action"][:7] for frame in frames], dtype=float)
    differences = np.diff(actions, axis=0)
    jerk = np.diff(actions, n=3, axis=0)*env.FPS**3 if len(actions) >= 4 else np.empty((0, 7))
    metrics = {
        "duration_s": len(frames)/env.FPS,
        "joint_path_length": float(np.linalg.norm(differences, axis=1).sum()) if len(differences) else 0.0,
        "max_action_jump": float(np.max(np.abs(differences))) if len(differences) else 0.0,
        "mean_jerk": float(np.mean(np.linalg.norm(jerk, axis=1))) if len(jerk) else 0.0,
    }
    if plan_metrics:
        metrics["planner_cost"] = float(sum(plan["cost"] for plan in plan_metrics))
        metrics["minimum_planned_clearance"] = float(min(plan["clearance"] for plan in plan_metrics))
    return metrics


def run_episode(
    spec: env.TaskSpec,
    teacher_source: str = "privileged",
    record_images: bool = True,
    max_frames: int = 700,
    diagnostic_dir: Path | None = None,
    render_size: int = 256,
    initial_joint_noise: float = 0.0,
) -> EpisodeResult:
    model, data, ids = env.build_task_model(spec)
    if initial_joint_noise > 0.0:
        rng = np.random.default_rng(int(spec.seed) + 99173)
        data.qpos[ids.arm_qpos] += rng.uniform(-initial_joint_noise, initial_joint_noise, size=len(ids.arm_qpos))
        mujoco.mj_forward(model, data)
    renderer = env.RenderPair(model, size=render_size) if record_images else None
    frames: list[dict[str, Any]] = []
    stages: list[str] = []
    events: list[dict[str, Any]] = []
    plans: list[dict[str, float]] = []
    max_retries = int(env.task_config()["max_retries"])
    tray = np.asarray(spec.tray_xy, dtype=float)
    gripper_open = 1.0
    gripper_hold = float(env.target_profile(spec)["gripper_hold"])
    physics_steps = int(round(env.DT/model.opt.timestep))

    def execute(q_targets: Iterable[np.ndarray], grip_targets: float | Iterable[float], stage: str) -> None:
        targets = list(q_targets)
        grips = [float(grip_targets)]*len(targets) if isinstance(grip_targets, (float, int)) else list(grip_targets)
        if len(targets) != len(grips):
            raise ValueError("q_targets 与 grip_targets 长度不同")
        for q_ref, grip in zip(targets, grips, strict=True):
            if len(frames) >= max_frames:
                raise TimeoutError(f"episode 超过 {max_frames} 帧")
            state = np.concatenate([
                data.qpos[ids.arm_qpos].copy(),
                [env.normalized_gripper_state(data, ids)],
            ]).astype(np.float32)
            action = np.concatenate([
                np.asarray(q_ref, dtype=np.float32),
                [np.float32(grip)],
            ]).astype(np.float32)
            if renderer is None:
                image = np.zeros((256, 256, 3), dtype=np.uint8)
                wrist = image.copy()
            else:
                image, wrist = renderer.render(data)
            frames.append({
                "observation.images.image": image,
                "observation.images.wrist_image": wrist,
                "observation.state": state,
                "action": action,
                "task": env.task_prompt(spec),
            })
            stages.append(stage)
            data.ctrl[ids.arm_actuator] = q_ref
            data.ctrl[ids.gripper_actuator] = env.actuator_gripper(model, ids, grip)
            for _ in range(physics_steps):
                mujoco.mj_step(model, data)

    def hold(q: np.ndarray, start_grip: float, end_grip: float, seconds: float, stage: str) -> None:
        count = max(1, int(math.ceil(seconds*env.FPS)))
        grips = [start_grip + env.base.smoothstep((index+1)/count)*(end_grip-start_grip) for index in range(count)]
        execute([q.copy() for _ in range(count)], grips, stage)

    def settle_before_release(q: np.ndarray, position: np.ndarray, grip: float) -> None:
        trajectory = env.task_config()["trajectory"]
        minimum = int(math.ceil(float(trajectory["place_settle_min_seconds"])*env.FPS))
        maximum = int(math.ceil(float(trajectory["place_settle_max_seconds"])*env.FPS))
        stable_frames = 0
        for index in range(maximum):
            execute([q.copy()], grip, "PLACE_SETTLE")
            tcp_error = float(np.linalg.norm(data.site_xpos[ids.tcp] - position))
            joint_speed = float(np.max(np.abs(data.qvel[ids.arm_dof])))
            stable = (
                tcp_error <= float(trajectory["release_tcp_tolerance"])
                and joint_speed <= float(trajectory["release_joint_speed"])
            )
            stable_frames = stable_frames + 1 if stable else 0
            if index + 1 >= minimum and stable_frames >= 3:
                return
        raise RuntimeError(
            f"放置位姿未稳定，拒绝松爪: tcp_error={tcp_error:.4f}, joint_speed={joint_speed:.4f}"
        )

    retry_count = 0
    first_attempt = True
    success = False
    failure_reason = "unknown"
    try:
        while retry_count <= max_retries:
            diagnostic = None
            if diagnostic_dir is not None:
                diagnostic = diagnostic_dir/f"seed_{spec.seed}_retry_{retry_count}.png"
            sensed_xy = env.locate_apple(model, data, ids, spec, teacher_source, diagnostic)
            if retry_count:
                profile = env.target_profile(spec)
                rest_z = env.base.TABLE_TOP_Z + float(profile["support_height"])
                target_xyz = data.xpos[ids.apple_body].copy()
                if target_xyz[2] < rest_z - 0.004:
                    target_xyz[2] = rest_z + 0.002
                    env.set_apple_pose(model, data, ids, target_xyz)
            planned_xy = sensed_xy.copy()
            if first_attempt and spec.recovery_type == "approach_offset":
                planned_xy += np.array([0.060, 0.0])
                events.append({"type": "approach_offset", "frame": len(frames), "offset": [0.060, 0.0]})
            elif retry_count:
                # A rolling sphere can stop a few millimetres away from the first
                # contact point. Re-detect it, then vary the grasp centre so a
                # retry is not an identical copy of the failed grasp.
                retry_distance = min(0.007, float(env.target_profile(spec)["horizontal_radius"])*0.10)
                retry_y = (retry_distance, -retry_distance)[(retry_count - 1) % 2]
                planned_xy += np.array([0.0, retry_y])
                events.append({"type": "retry_grasp_offset", "frame": len(frames), "offset": [0.0, retry_y]})

            current_q = data.qpos[ids.arm_qpos].copy()
            rotation = data.site_xmat[ids.tcp].reshape(3, 3).copy()
            profile = env.target_profile(spec)
            support_height = float(profile["support_height"])
            object_center_z = float(data.xpos[ids.apple_body, 2])
            grasp_z = object_center_z + 0.015 - min(retry_count, 2)*0.003
            pre_position = np.array([*planned_xy, max(0.50, grasp_z+0.11)])
            grasp_position = np.array([*planned_xy, grasp_z])
            lift_z = env.minimum_clearance(spec) + float(env.task_config()["trajectory"]["pretransfer_lift_margin"])
            lift_position = np.array([*planned_xy, max(grasp_z+0.080, lift_z)])
            try:
                q_pre = env.base.solve_ik(
                    model, ids.tcp, ids.arm_joint, ids.arm_qpos, ids.arm_dof,
                    current_q, pre_position, rotation, max_iterations=600,
                )
                q_grasp = env.base.solve_ik(
                    model, ids.tcp, ids.arm_joint, ids.arm_qpos, ids.arm_dof,
                    q_pre, grasp_position, rotation, max_iterations=600,
                )
                q_lift = env.base.solve_ik(
                    model, ids.tcp, ids.arm_joint, ids.arm_qpos, ids.arm_dof,
                    q_grasp, lift_position, rotation, max_iterations=600,
                )
            except RuntimeError as error:
                failure_reason = f"approach_ik: {error}"
                retry_count += 1
                first_attempt = False
                continue

            trajectory = env.task_config()["trajectory"]
            execute(env.joint_segment(current_q, q_pre, float(trajectory["approach_seconds"])), gripper_open, "PREGRASP")
            execute(env.joint_segment(q_pre, q_grasp, float(trajectory["descend_seconds"])), gripper_open, "DESCEND")
            close_target = 0.84 if first_attempt and spec.recovery_type == "weak_grasp" else max(0.06, gripper_hold-0.02*retry_count)
            if first_attempt and spec.recovery_type == "weak_grasp":
                events.append({"type": "weak_grasp", "frame": len(frames), "gripper": close_target})
            hold(q_grasp, gripper_open, close_target, float(trajectory["gripper_seconds"]), "CLOSE")
            execute(
                env.joint_segment(q_grasp, q_lift, float(trajectory["lift_seconds"]),
                                  max_step=float(trajectory["lift_max_joint_step"])),
                close_target, "VERIFY_GRASP_LIFT",
            )
            apple_height = float(data.xpos[ids.apple_body, 2])
            grasp_offset_z = float(data.site_xpos[ids.tcp, 2] - data.xpos[ids.apple_body, 2])
            grasped = apple_height > env.base.TABLE_TOP_Z + support_height + 0.025
            if not grasped:
                failure_reason = "grasp_verification_failed"
                hold(q_lift, close_target, gripper_open, 0.6, "RECOVER_OPEN")
                execute(env.joint_segment(q_lift, q_pre, 1.0), gripper_open, "RECOVER_RETREAT")
                for _ in range(300):
                    mujoco.mj_step(model, data)
                retry_count += 1
                first_attempt = False
                continue

            if first_attempt and spec.recovery_type == "object_slip":
                current_xy = data.xpos[ids.apple_body, :2].copy()
                distractors = [np.asarray(position) for position in spec.positions[1:]]
                directions = [
                    np.array([1.0, 0.0]), np.array([-1.0, 0.0]),
                    np.array([0.0, 1.0]), np.array([0.0, -1.0]),
                    np.array([0.707, 0.707]), np.array([0.707, -0.707]),
                    np.array([-0.707, 0.707]), np.array([-0.707, -0.707]),
                ]
                candidates = []
                for direction in directions:
                    candidate = current_xy + 0.040*direction
                    if not (0.45 <= candidate[0] <= 0.66 and -0.22 <= candidate[1] <= 0.11):
                        continue
                    obstacle_clearance = min(np.linalg.norm(candidate-other) for other in distractors)
                    tray_clearance = np.linalg.norm(candidate-tray)
                    if obstacle_clearance >= 0.085 and tray_clearance >= 0.12:
                        candidates.append((min(obstacle_clearance, tray_clearance), candidate))
                slip_xy = max(candidates, key=lambda item: item[0])[1] if candidates else current_xy + np.array([0.0, -0.035])
                slip_xyz = np.array([*slip_xy, env.base.TABLE_TOP_Z+support_height+0.002])
                env.set_apple_pose(model, data, ids, slip_xyz)
                events.append({"type": "object_slip", "frame": len(frames), "apple_xyz": slip_xyz.tolist()})
                hold(q_lift, close_target, gripper_open, 1.0, "RECOVER_SLIP_OPEN")
                execute(env.joint_segment(q_lift, q_pre, 1.0), gripper_open, "RECOVER_SLIP_RETREAT")
                for _ in range(300):
                    mujoco.mj_step(model, data)
                retry_count += 1
                first_attempt = False
                continue

            destination = tray.copy()
            if first_attempt and spec.recovery_type == "place_offset":
                destination += np.array([0.0, -0.180])
                events.append({"type": "place_offset", "frame": len(frames), "offset": [0.0, -0.180]})
            floor_top_z = env.base.TABLE_TOP_Z + 0.008
            floor_preload = float(trajectory["release_floor_preload"])
            placed_center_z = floor_top_z + support_height - floor_preload
            place_z = placed_center_z + grasp_offset_z
            place_position = np.array([*destination, place_z])
            box_hover_z = env.base.TABLE_TOP_Z + float(env.task_config()["tray"]["wall_height"])
            box_hover_z += float(trajectory["box_hover_margin"])
            hover_position = np.array([*destination, max(box_hover_z, env.minimum_clearance(spec))])
            plan = env.plan_transfer(
                model, ids, q_lift, data.site_xpos[ids.tcp].copy(),
                hover_position, rotation, spec,
            )
            plans.append({
                "cost": plan.cost,
                "clearance": plan.clearance,
                "candidate_offset": plan.candidate_offset,
                "collision_count": float(plan.collision_count),
            })
            execute(env.densify_path(plan.q_path, max_step=0.035), close_target, "TRANSFER_CURVE")
            q_hover = plan.q_path[-1]
            q_place = env.base.solve_ik(
                model, ids.tcp, ids.arm_joint, ids.arm_qpos, ids.arm_dof,
                q_hover, place_position, rotation, max_iterations=600,
            )
            execute(
                env.joint_segment(q_hover, q_place, float(trajectory["descend_seconds"]), max_step=0.025),
                close_target, "BOX_DESCEND",
            )
            settle_before_release(q_place, place_position, close_target)
            hold(q_place, close_target, gripper_open, 1.0, "OPEN")
            settle_count = int(math.ceil(float(trajectory["settle_seconds"])*env.FPS))
            execute([q_place.copy() for _ in range(settle_count)], gripper_open, "VERIFY_PLACE")
            success = env.apple_in_tray(data, ids, tray)

            execute(
                env.joint_segment(q_place, q_hover, 1.2, max_step=0.025),
                gripper_open, "RETREAT" if success else "RECOVER_RETREAT",
            )
            if success:
                failure_reason = "none"
                break
            failure_reason = "place_verification_failed"
            retry_count += 1
            first_attempt = False
            for _ in range(150):
                mujoco.mj_step(model, data)
    except (RuntimeError, TimeoutError) as error:
        failure_reason = f"{type(error).__name__}: {error}"
        success = False
    finally:
        if renderer is not None:
            renderer.close()

    metadata: dict[str, Any] = {
        "scene_seed": spec.seed,
        "split": spec.split,
        "target_type": spec.target_type,
        "success": success,
        "recovery_type": spec.recovery_type,
        "retry_count": retry_count,
        "teacher_source": teacher_source,
        "teacher_version": env.TEACHER_VERSION,
        "camera_config_version": env.CAMERA_VERSION,
        "tray_xy": spec.tray_xy,
        "positions": spec.positions,
        "events": events,
        "stage_counts": {name: stages.count(name) for name in sorted(set(stages))},
        "failure_reason": failure_reason,
        "final_apple_xyz": data.xpos[ids.apple_body].astype(float).tolist(),
        "plans": plans,
    }
    metadata.update(episode_metrics(frames, plans))
    return EpisodeResult(success=success, frames=frames, metadata=metadata)
