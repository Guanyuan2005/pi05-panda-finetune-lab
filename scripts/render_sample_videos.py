#!/usr/bin/env python3
"""Render deterministic normal/recovery demonstrations for human inspection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from pi05_local import environment as env
from pi05_local.teacher import run_episode


CONDITIONS = ("none", "approach_offset", "weak_grasp", "object_slip", "place_offset")
TARGETS = ("apple", "orange", "can", "box")
SAMPLES = tuple(
    (f"ep{condition_index*4+target_index:02d}_{target}_{'normal' if condition == 'none' else condition}",
     400 + condition_index*4 + target_index, condition)
    for condition_index, condition in enumerate(CONDITIONS)
    for target_index, target in enumerate(TARGETS)
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render two-camera sample videos.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT/"reports"/"sample_videos")
    parser.add_argument("--teacher-source", choices=("privileged", "vision"), default="privileged")
    parser.add_argument("--case", choices=("all", *(name for name, _, _ in SAMPLES)), default="all")
    return parser.parse_args()


def write_video(path: Path, frames: list[dict], fps: int) -> None:
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width, stream.height, stream.pix_fmt = 1024, 512, "yuv420p"
    stream.options = {"crf": "16", "preset": "medium"}
    try:
        for frame in frames:
            pair = np.concatenate([
                frame["observation.images.image"],
                frame["observation.images.wrist_image"],
            ], axis=1)
            for packet in stream.encode(av.VideoFrame.from_ndarray(pair, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def contact_sheet(rows: list[tuple[str, np.ndarray]], output: Path) -> None:
    tile_width, tile_height = 1024, 542
    canvas = Image.new("RGB", (tile_width, tile_height*len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (name, frame) in enumerate(rows):
        y = index*tile_height
        canvas.paste(Image.fromarray(frame), (0, y+30))
        draw.text((8, y+8), f"{name}: external camera | wrist camera", fill="black")
    canvas.save(output)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = SAMPLES if args.case == "all" else tuple(row for row in SAMPLES if row[0] == args.case)
    results = []
    covers = []
    for name, seed, recovery_type in selected:
        spec = env.make_task_spec(seed, recovery_type=recovery_type)
        episode = run_episode(spec, teacher_source=args.teacher_source, record_images=True, render_size=512)
        output = args.output_dir/f"{name}.mp4"
        write_video(output, episode.frames, env.FPS)
        middle = episode.frames[len(episode.frames)//2]
        covers.append((name, np.concatenate([
            middle["observation.images.image"], middle["observation.images.wrist_image"]
        ], axis=1)))
        results.append({
            "name": name,
            "seed": seed,
            "target_type": spec.target_type,
            "recovery_type": recovery_type,
            "success": episode.success,
            "retry_count": episode.metadata["retry_count"],
            "frames": len(episode.frames),
            "video": output.resolve().as_posix(),
            "events": episode.metadata["events"],
        })
        print(f"{name}: success={episode.success}, frames={len(episode.frames)}, video={output}")
    contact_sheet(covers, args.output_dir/"overview.png")
    (args.output_dir/"index.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0 if all(row["success"] for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
