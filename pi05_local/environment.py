from __future__ import annotations

import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent.parent
ARM_LAB = HERE.parent / "mujoco_arm_lab"
if str(ARM_LAB) not in sys.path:
    sys.path.insert(0, str(ARM_LAB))

import panda_grasp as base
import panda_vision_grasp as vision
from .scene_overrides import TARGET_PROFILES, configure_target, stabilize_scene

FPS = 10
DT = 1.0 / FPS
EXTERNAL_CAMERA = vision.CAMERA_NAMES[0]
WRIST_CAMERA = "腕部相机"
TEACHER_VERSION = "panda_multi_object_teacher_v2"
CAMERA_VERSION = "camera_v1"
RECOVERY_TYPES = ("approach_offset", "weak_grasp", "object_slip", "place_offset")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def task_config() -> dict[str, Any]:
    return load_yaml(HERE / "configs" / "panda_apple_task.yaml")


def camera_config() -> dict[str, Any]:
    return load_yaml(HERE / "configs" / "camera_v1.yaml")


@dataclass
class TaskSpec:
    seed: int
    positions: list[list[float]]
    tray_xy: list[float]
    target_type: str = "apple"
    recovery_type: str = "none"
    split: str = "train"


@dataclass
class RobotIds:
    arm_joint: np.ndarray
    arm_qpos: np.ndarray
    arm_dof: np.ndarray
    finger_joint: np.ndarray
    finger_qpos: np.ndarray
    arm_actuator: np.ndarray
    gripper_actuator: int
    home_key: int
    tcp: int
    apple_body: int
    apple_joint: int
    apple_qpos: int
    apple_dof: int
    robot_bodies: set[int]


@dataclass
class PlanResult:
    q_path: list[np.ndarray]
    tcp_path: list[np.ndarray]
    cost: float
    clearance: float
    collision_count: int
    candidate_offset: float


class RenderPair:
    def __init__(self, model: mujoco.MjModel, size: int = 256):
        self.option = mujoco.MjvOption()
        self.option.geomgroup[5] = 0
        self.external = mujoco.Renderer(model, size, size)
        self.wrist = mujoco.Renderer(model, size, size)

    def render(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        self.external.update_scene(data, camera=EXTERNAL_CAMERA, scene_option=self.option)
        external = self.external.render().copy().astype(np.uint8)
        self.wrist.update_scene(data, camera=WRIST_CAMERA, scene_option=self.option)
        wrist = self.wrist.render().copy().astype(np.uint8)
        return external, wrist

    def close(self) -> None:
        self.external.close()
        self.wrist.close()


def make_assets_with_wrist_camera() -> dict[str, bytes]:
    assets = base.add_tcp_to_panda_assets()
    panda_xml = assets["panda.xml"]
    anchor = b'<body name="left_finger" pos="0 0 0.0584">'
    cfg = camera_config()["wrist_camera_in_hand"]
    pos = " ".join(str(v) for v in cfg["pos"])
    quat = " ".join(str(v) for v in cfg["quat"])
    camera = (
        f'<camera name="{WRIST_CAMERA}" pos="{pos}" quat="{quat}" '
        f'fovy="{cfg["fovy"]}"/>\n                      '
    ).encode("utf-8")
    if panda_xml.count(anchor) != 1:
        raise RuntimeError("Panda XML 结构变化，无法插入腕部相机")
    assets["panda.xml"] = panda_xml.replace(anchor, camera + anchor, 1)
    assets.update(vision.realistic_asset_files())
    return assets


def tray_xml(tray_xy: np.ndarray) -> str:
    cfg = task_config()["tray"]
    hx, hy = [float(v) for v in cfg["inner_half_size"]]
    wall = float(cfg["wall_thickness"])
    wall_h = float(cfg["wall_height"])
    z0 = base.TABLE_TOP_Z
    return f'''
    <body name="绿色托盘" pos="{tray_xy[0]:.7f} {tray_xy[1]:.7f} {z0 + 0.004:.7f}">
      <geom name="托盘底" type="box" size="{hx + wall:.4f} {hy + wall:.4f} .004"
            rgba=".08 .62 .20 1" friction="1.1 .015 .001"/>
      <geom name="托盘左壁" type="box" pos="{-hx-wall/2:.4f} 0 {wall_h/2:.4f}"
            size="{wall/2:.4f} {hy+wall:.4f} {wall_h/2:.4f}" rgba=".06 .48 .15 1"/>
      <geom name="托盘右壁" type="box" pos="{hx+wall/2:.4f} 0 {wall_h/2:.4f}"
            size="{wall/2:.4f} {hy+wall:.4f} {wall_h/2:.4f}" rgba=".06 .48 .15 1"/>
      <geom name="托盘前壁" type="box" pos="0 {-hy-wall/2:.4f} {wall_h/2:.4f}"
            size="{hx:.4f} {wall/2:.4f} {wall_h/2:.4f}" rgba=".06 .48 .15 1"/>
      <geom name="托盘后壁" type="box" pos="0 {hy+wall/2:.4f} {wall_h/2:.4f}"
            size="{hx:.4f} {wall/2:.4f} {wall_h/2:.4f}" rgba=".06 .48 .15 1"/>
    </body>
'''


def make_task_spec(seed: int, recovery_type: str = "none", split: str = "train") -> TaskSpec:
    if recovery_type not in ("none", *RECOVERY_TYPES):
        raise ValueError(f"未知 recovery_type: {recovery_type}")
    cfg = task_config()
    rng = np.random.default_rng(seed)
    count = int(cfg["distractor_count"]) + 1
    positions = vision.sample_positions(rng, count, realistic=True)
    tray_cfg = cfg["tray"]
    tray = np.asarray(tray_cfg["center_xy"], dtype=float)
    jitter = np.asarray(tray_cfg["random_xy"], dtype=float)
    tray += rng.uniform(-jitter, jitter)
    target_types = tuple(cfg["target_objects"])
    target_type = target_types[int(seed) % len(target_types)]
    return TaskSpec(
        seed=int(seed),
        positions=[p.astype(float).tolist() for p in positions],
        tray_xy=tray.tolist(),
        target_type=target_type,
        recovery_type=recovery_type,
        split=split,
    )


def resolve_ids(model: mujoco.MjModel) -> RobotIds:
    arm_joint = np.asarray([
        base.object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in base.ARM_JOINT_NAMES
    ])
    finger_joint = np.asarray([
        base.object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in base.FINGER_JOINT_NAMES
    ])
    arm_actuator = np.asarray([
        base.object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        for name in base.ARM_ACTUATOR_NAMES
    ])
    apple_joint = base.object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
    names = {f"link{i}" for i in range(1, 8)} | {"hand", "left_finger", "right_finger"}
    robot_bodies = {
        body_id for name in names
        if (body_id := mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)) >= 0
    }
    return RobotIds(
        arm_joint=arm_joint,
        arm_qpos=model.jnt_qposadr[arm_joint].copy(),
        arm_dof=model.jnt_dofadr[arm_joint].copy(),
        finger_joint=finger_joint,
        finger_qpos=model.jnt_qposadr[finger_joint].copy(),
        arm_actuator=arm_actuator,
        gripper_actuator=base.object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8"),
        home_key=base.object_id(model, mujoco.mjtObj.mjOBJ_KEY, "home"),
        tcp=base.object_id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp"),
        apple_body=base.object_id(model, mujoco.mjtObj.mjOBJ_BODY, "cube"),
        apple_joint=apple_joint,
        apple_qpos=int(model.jnt_qposadr[apple_joint]),
        apple_dof=int(model.jnt_dofadr[apple_joint]),
        robot_bodies=robot_bodies,
    )


def reset_data(model: mujoco.MjModel, data: mujoco.MjData, ids: RobotIds) -> None:
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    data.qpos[ids.arm_qpos] = model.key_qpos[ids.home_key, ids.arm_qpos]
    data.qpos[ids.finger_qpos] = model.key_qpos[ids.home_key, ids.finger_qpos]
    data.ctrl[ids.arm_actuator] = data.qpos[ids.arm_qpos]
    data.ctrl[ids.gripper_actuator] = model.actuator_ctrlrange[ids.gripper_actuator, 1]
    mujoco.mj_forward(model, data)
    for _ in range(250):
        mujoco.mj_step(model, data)


def build_task_model(spec: TaskSpec) -> tuple[mujoco.MjModel, mujoco.MjData, RobotIds]:
    positions = [np.asarray(position, dtype=float) for position in spec.positions]
    scene = vision.build_random_scene(positions, realistic=True)
    scene = stabilize_scene(scene)
    profile = target_profile(spec)
    scene = configure_target(
        scene, spec.target_type, base.TABLE_TOP_Z + float(profile["support_height"])
    )
    scene = re.sub(r'\s*<geom name="放置区\d+".*?/>', "", scene, flags=re.DOTALL)
    scene = scene.replace(
        "  </worldbody>", tray_xml(np.asarray(spec.tray_xy, dtype=float)) + "  </worldbody>", 1
    )
    model = mujoco.MjModel.from_xml_string(scene, assets=make_assets_with_wrist_camera())
    data = mujoco.MjData(model)
    ids = resolve_ids(model)
    reset_data(model, data, ids)
    return model, data, ids


def normalized_gripper_state(data: mujoco.MjData, ids: RobotIds) -> float:
    return float(np.clip(np.mean(data.qpos[ids.finger_qpos]) / 0.040, 0.0, 1.0))


def actuator_gripper(model: mujoco.MjModel, ids: RobotIds, normalized: float) -> float:
    low, high = model.actuator_ctrlrange[ids.gripper_actuator]
    return float(low + np.clip(normalized, 0.0, 1.0) * (high - low))


def minimum_clearance(spec: TaskSpec) -> float:
    positions = [np.asarray(position) for position in spec.positions[1:]]
    shapes = vision.REALISTIC_OBJECT_SHAPES[1 : len(positions) + 1]
    start = np.asarray(spec.positions[0])
    goal = np.asarray(spec.tray_xy)
    direction = goal - start
    norm2 = float(direction @ direction)
    highest = base.TABLE_TOP_Z + 0.09
    for position, shape in zip(positions, shapes, strict=True):
        t = float(np.clip(((position - start) @ direction) / max(norm2, 1e-8), 0.0, 1.0))
        nearest = start + t * direction
        if np.linalg.norm(position - nearest) < 0.13:
            highest = max(highest, base.TABLE_TOP_Z + 2.0 * float(shape[3]))
    margin = float(task_config()["trajectory"]["obstacle_margin"])
    wall_height = float(task_config()["tray"]["wall_height"])
    radius = float(target_profile(spec)["horizontal_radius"])
    return max(base.TABLE_TOP_Z + wall_height + 0.075, highest + radius + margin)


def cubic_bezier(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, t: float) -> np.ndarray:
    u = 1.0 - t
    return u**3*p0 + 3*u*u*t*p1 + 3*u*t*t*p2 + t**3*p3


def collision_count(model: mujoco.MjModel, ids: RobotIds, q: np.ndarray, gripper_norm: float = 0.35) -> int:
    check = mujoco.MjData(model)
    check.qpos[:] = model.qpos0
    check.qpos[ids.arm_qpos] = q
    check.qpos[ids.finger_qpos] = 0.040 * gripper_norm
    mujoco.mj_forward(model, check)
    count = 0
    for index in range(check.ncon):
        contact = check.contact[index]
        body1 = int(model.geom_bodyid[contact.geom1])
        body2 = int(model.geom_bodyid[contact.geom2])
        if ids.apple_body in {body1, body2}:
            continue
        if body1 in ids.robot_bodies or body2 in ids.robot_bodies:
            count += 1
    return count


def path_cost(qs: list[np.ndarray], points: list[np.ndarray], clearance: float) -> float:
    q = np.asarray(qs)
    points_array = np.asarray(points)
    joint_length = float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum())
    tcp_length = float(np.linalg.norm(np.diff(points_array, axis=0), axis=1).sum())
    jerk = float(np.linalg.norm(np.diff(q, n=3, axis=0), axis=1).sum()) if len(q) >= 4 else 0.0
    return 0.8*tcp_length + 0.12*joint_length + 0.02*jerk + 0.005/max(clearance-base.TABLE_TOP_Z, 1e-3)


def plan_transfer(
    model: mujoco.MjModel,
    ids: RobotIds,
    start_q: np.ndarray,
    start_pos: np.ndarray,
    goal_pos: np.ndarray,
    rotation: np.ndarray,
    spec: TaskSpec,
) -> PlanResult:
    cfg = task_config()["trajectory"]
    samples = int(cfg["curve_samples"])
    clearance = minimum_clearance(spec)
    direction = goal_pos[:2] - start_pos[:2]
    distance = float(np.linalg.norm(direction))
    normal = np.array([-direction[1], direction[0]]) / max(distance, 1e-8)
    candidates: list[PlanResult] = []
    errors: list[str] = []
    for raw_offset in cfg["lateral_offsets"]:
        offset = float(raw_offset)
        control1 = start_pos.copy()
        control2 = goal_pos.copy()
        control1[:2] += direction/3.0 + offset*normal
        control2[:2] -= direction/3.0 - offset*normal
        control1[2] = clearance
        control2[2] = clearance
        points = [
            cubic_bezier(start_pos, control1, control2, goal_pos, value)
            for value in np.linspace(0.0, 1.0, samples)
        ]
        qs = [start_q.copy()]
        collisions = 0
        try:
            for point in points[1:]:
                q = base.solve_ik(
                    model, ids.tcp, ids.arm_joint, ids.arm_qpos, ids.arm_dof,
                    qs[-1], point, rotation, max_iterations=500,
                )
                collisions += collision_count(model, ids, q)
                qs.append(q)
        except RuntimeError as error:
            errors.append(f"offset={offset}: {error}")
            continue
        if collisions:
            errors.append(f"offset={offset}: {collisions} collision samples")
            continue
        candidates.append(PlanResult(
            q_path=qs,
            tcp_path=points,
            cost=path_cost(qs, points, clearance),
            clearance=clearance,
            collision_count=collisions,
            candidate_offset=offset,
        ))
    if not candidates:
        raise RuntimeError("没有安全可行的搬运路径: " + "; ".join(errors))
    return min(candidates, key=lambda result: result.cost)


def joint_segment(q0: np.ndarray, q1: np.ndarray, seconds: float, max_step: float | None = None) -> list[np.ndarray]:
    configured = float(task_config()["trajectory"]["max_joint_step"])
    step = configured if max_step is None else max_step
    by_time = max(2, int(math.ceil(seconds*FPS)))
    # smoothstep has a maximum slope of 1.5; include it so max_step is a hard limit.
    by_step = max(2, int(math.ceil(1.5*np.max(np.abs(q1-q0))/step)))
    count = max(by_time, by_step)
    return [base.interpolate(q0, q1, (index+1)/count) for index in range(count)]


def densify_path(qs: list[np.ndarray], max_step: float | None = None) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    for q0, q1 in zip(qs[:-1], qs[1:], strict=True):
        result.extend(joint_segment(q0, q1, DT, max_step=max_step))
    return result


def target_profile(spec: TaskSpec) -> dict[str, Any]:
    try:
        return TARGET_PROFILES[spec.target_type]
    except KeyError as error:
        raise ValueError(f"未知目标物体: {spec.target_type}") from error


def task_prompt(spec: TaskSpec) -> str:
    return task_config()["task"].format(target=target_profile(spec)["label"])


def locate_apple(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ids: RobotIds,
    spec: TaskSpec,
    source: str,
    diagnostic_path: Path | None = None,
) -> np.ndarray:
    if source == "privileged":
        return data.xpos[ids.apple_body, :2].copy()
    if source != "vision":
        raise ValueError("teacher_source 必须是 privileged 或 vision")
    sense = mujoco.MjData(model)
    sense.qpos[:] = data.qpos
    sense.qvel[:] = data.qvel
    mujoco.mj_forward(model, sense)
    rotation = sense.site_xmat[ids.tcp].reshape(3, 3).copy()
    q_observe = base.solve_ik(
        model, ids.tcp, ids.arm_joint, ids.arm_qpos, ids.arm_dof,
        sense.qpos[ids.arm_qpos].copy(), np.array([0.36, -0.34, 0.64]),
        rotation, max_iterations=600,
    )
    sense.qpos[ids.arm_qpos] = q_observe
    mujoco.mj_forward(model, sense)
    output = diagnostic_path or HERE/"logs"/f"vision_seed_{spec.seed}.png"
    detected, _ = vision.detect_objects(model, sense, len(spec.positions), output, realistic=True)
    return detected[0]


def apple_in_tray(data: mujoco.MjData, ids: RobotIds, tray_xy: np.ndarray) -> bool:
    position = data.xpos[ids.apple_body]
    hx, hy = np.asarray(task_config()["tray"]["inner_half_size"], dtype=float) - 0.022
    inside = abs(float(position[0]-tray_xy[0])) <= hx and abs(float(position[1]-tray_xy[1])) <= hy
    height_ok = base.TABLE_TOP_Z + 0.025 <= position[2] <= base.TABLE_TOP_Z + 0.090
    return bool(inside and height_ok)


def set_apple_pose(model: mujoco.MjModel, data: mujoco.MjData, ids: RobotIds, xyz: np.ndarray) -> None:
    address = ids.apple_qpos
    data.qpos[address:address+3] = xyz
    data.qpos[address+3:address+7] = np.array([1.0, 0.0, 0.0, 0.0])
    data.qvel[ids.apple_dof:ids.apple_dof+6] = 0.0
    mujoco.mj_forward(model, data)
