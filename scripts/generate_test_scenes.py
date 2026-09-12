#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from pi05_local import environment as env


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze held-out deterministic test scenes without actions.")
    parser.add_argument("--seed-start", type=int, default=50000)
    parser.add_argument("--normal", type=int, default=50)
    parser.add_argument("--recovery", type=int, default=50)
    parser.add_argument("--output", type=Path, default=PROJECT/"data"/"test_scenes.json")
    args = parser.parse_args()
    scenes = []
    for index in range(args.normal):
        scenes.append(env.make_task_spec(args.seed_start + index, split="test").__dict__)
    for index in range(args.recovery):
        recovery = env.RECOVERY_TYPES[index % len(env.RECOVERY_TYPES)]
        spec = env.make_task_spec(
            args.seed_start + args.normal + index,
            recovery_type=recovery,
            split="test",
        )
        scenes.append(spec.__dict__)
    payload = {
        "version": "panda_multi_object_box_test_v2",
        "task_template": env.task_config()["task"],
        "normal_scenes": args.normal,
        "recovery_scenes": args.recovery,
        "scenes": scenes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(scenes)} held-out scenes to {args.output}")


if __name__ == "__main__":
    main()
