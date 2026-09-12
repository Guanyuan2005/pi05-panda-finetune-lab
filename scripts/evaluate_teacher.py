#!/usr/bin/env python3
"""Evaluate the deterministic MuJoCo teacher before expensive data collection."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from pi05_local import environment as env
from pi05_local.teacher import run_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--split", default="train")
    parser.add_argument("--mode", choices=("nominal", "recovery", "mixed"), default="nominal")
    parser.add_argument("--teacher-source", choices=("privileged", "vision"), default="privileged")
    parser.add_argument("--min-success-rate", type=float, default=0.90)
    parser.add_argument("--report-json", type=Path, default=Path("reports/teacher_validation.json"))
    parser.add_argument("--report-md", type=Path, default=Path("reports/teacher_validation.md"))
    return parser.parse_args()


def recovery_for(mode: str, index: int) -> str:
    if mode == "nominal":
        return "none"
    types = ("approach_offset", "weak_grasp", "object_slip", "place_offset")
    if mode == "recovery":
        return types[index % len(types)]
    return "none" if index % 2 == 0 else types[(index // 2) % len(types)]


def main() -> int:
    args = parse_args()
    rows = []
    for index in range(args.episodes):
        seed = args.seed_start + index
        recovery = recovery_for(args.mode, index)
        spec = env.make_task_spec(seed=seed, split=args.split, recovery_type=recovery)
        result = run_episode(spec, teacher_source=args.teacher_source, record_images=False)
        rows.append({
            "seed": seed,
            "recovery_type": recovery,
            "success": result.success,
            "retry_count": result.metadata["retry_count"],
            "frames": len(result.frames),
            "failure_reason": result.metadata["failure_reason"],
            "max_action_jump": result.metadata["max_action_jump"],
            "minimum_planned_clearance": result.metadata.get("minimum_planned_clearance"),
        })
        print(f"[{index + 1:03d}/{args.episodes:03d}] seed={seed} recovery={recovery} success={result.success}")

    successes = sum(row["success"] for row in rows)
    success_rate = successes / args.episodes if args.episodes else 0.0
    report = {
        "ok": success_rate >= args.min_success_rate,
        "teacher_source": args.teacher_source,
        "mode": args.mode,
        "episodes": args.episodes,
        "successes": successes,
        "success_rate": success_rate,
        "minimum_success_rate": args.min_success_rate,
        "recovery_counts": dict(Counter(row["recovery_type"] for row in rows)),
        "failure_counts": dict(Counter(row["failure_reason"] for row in rows if not row["success"])),
        "max_action_jump": max((row["max_action_jump"] for row in rows), default=0.0),
        "episodes_detail": rows,
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.report_md.write_text(
        "# 教师策略验证报告\n\n"
        f"- 模式：`{args.mode}`\n"
        f"- 教师观测：`{args.teacher_source}`\n"
        f"- 成功：{successes}/{args.episodes}（{report['success_rate']:.1%}）\n"
        f"- 最大相邻动作跳变：{report['max_action_jump']:.4f} rad\n"
        f"- 失败类型：`{json.dumps(report['failure_counts'], ensure_ascii=False)}`\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in report.items() if key != "episodes_detail"}, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
