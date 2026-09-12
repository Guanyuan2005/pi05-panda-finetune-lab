"""π0.5-specific scene simplification and target-object physics."""

from __future__ import annotations

import re

TARGET_PROFILES = {
    "apple": {
        "label": "red apple", "support_height": 0.036,
        "horizontal_radius": 0.036, "mass": 0.065, "gripper_hold": 0.28,
    },
    "orange": {
        "label": "orange", "support_height": 0.033,
        "horizontal_radius": 0.033, "mass": 0.075, "gripper_hold": 0.25,
        "geom": 'type="sphere" size=".033" rgba=".95 .36 .04 1"',
    },
    "can": {
        "label": "blue can", "support_height": 0.035,
        "horizontal_radius": 0.027, "mass": 0.080, "gripper_hold": 0.22,
        "geom": 'type="box" size=".026 .026 .035" rgba="0 0 0 0"',
        "visual_geom": '<geom name="蓝色罐外观" type="cylinder" size=".027 .035" '
                       'rgba=".08 .32 .82 1" mass="0" contype="0" conaffinity="0"/>',
    },
    "box": {
        "label": "yellow box", "support_height": 0.027,
        "horizontal_radius": 0.035, "mass": 0.060, "gripper_hold": 0.24,
        "geom": 'type="box" size=".035 .027 .027" rgba=".94 .72 .08 1"',
    },
}


def stabilize_scene(scene: str) -> str:
    # Do not inherit the extra fixed-cup demo obstacle from mujoco_arm_lab.
    scene = re.sub(
        r'\s*<body name="固定障碍水杯".*?</body>',
        "",
        scene,
        count=1,
        flags=re.DOTALL,
    )
    # The smooth mesh is visual only. A centered sphere supplies predictable
    # mass, inertia and contacts so the apple does not wobble unnaturally.
    scene = re.sub(
        r'<geom name="红色苹果惯量代理".*?/>',
        '<geom name="红色苹果惯量代理" type="sphere" size=".036" '
        'mass=".065" rgba="0 0 0 0" friction="1.2 .02 .01" condim="4"/>',
        scene,
        count=1,
        flags=re.DOTALL,
    )
    scene = re.sub(
        r'<geom name="真实苹果果肉".*?/>',
        '<geom name="真实苹果果肉" type="mesh" mesh="真实苹果果肉网格" '
        'rgba=".812 .090 .078 1" mass="0" contype="0" conaffinity="0"/>',
        scene,
        count=1,
        flags=re.DOTALL,
    )
    scene = scene.replace(
        '<freejoint name="cube_free"/>',
        '<joint name="cube_free" type="free" damping=".08"/>',
    )
    return scene


def configure_target(scene: str, target_type: str, target_z: float) -> str:
    """Turn the first free body into the selected grasp target."""
    if target_type not in TARGET_PROFILES:
        raise ValueError(f"未知目标物体: {target_type}")
    scene = re.sub(
        r'(<body name="cube" pos="[^ ]+ [^ ]+ )[^" ]+(\">)',
        rf'\g<1>{target_z:.7f}\g<2>', scene, count=1,
    )
    if target_type == "apple":
        return scene
    profile = TARGET_PROFILES[target_type]
    scene = re.sub(
        r'<geom name="红色苹果惯量代理".*?/>',
        f'<geom name="目标碰撞体" {profile["geom"]} mass="{profile["mass"]}" '
        'friction="1.8 .02 .01" condim="4"/>' + profile.get("visual_geom", ""),
        scene, count=1, flags=re.DOTALL,
    )
    for name in ("真实苹果果肉", "真实苹果梗", "真实苹果叶"):
        scene = re.sub(rf'\s*<geom name="{name}".*?/>', "", scene, count=1, flags=re.DOTALL)
    return scene
