#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from pi05_local.dataset import collect_shard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect optimized Panda apple-to-tray LeRobot episodes.")
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--episodes", type=int, required=True, help="Total successful episodes to collect.")
    parser.add_argument("--mode", choices=("nominal", "recovery", "mixed"), default="mixed")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--worker-id", type=int, default=0, help="Only used in explicit single-worker mode.")
    parser.add_argument("--dataset-root", type=Path, help="Output for a single worker.")
    parser.add_argument("--staging-root", type=Path, help="Parent containing worker-XX shards.")
    parser.add_argument("--teacher-source", choices=("privileged", "vision"), default="privileged")
    parser.add_argument("--image-storage", choices=("image", "video"), default="image")
    parser.add_argument("--initial-joint-noise", type=float, default=0.0, help="每回合初始关节角均匀扰动幅度(rad)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--headless", action="store_true", help="Accepted for documented CLI compatibility; collection is always headless.")
    return parser.parse_args()


def worker_call(kwargs: dict) -> dict:
    return collect_shard(**kwargs)


def main() -> None:
    args = parse_args()
    if args.episodes <= 0 or args.workers <= 0:
        raise ValueError("--episodes 和 --workers 必须大于 0")
    if args.workers == 1:
        output = args.dataset_root or ((args.staging_root or PROJECT/"data"/"staging"/args.split)/f"worker-{args.worker_id:02d}")
        manifest = collect_shard(
            root=output,
            split=args.split,
            requested_successes=args.episodes,
            seed_start=args.seed_start,
            seed_stride=1,
            worker_id=args.worker_id,
            mode=args.mode,
            teacher_source=args.teacher_source,
            image_storage=args.image_storage,
            overwrite=args.overwrite,
            initial_joint_noise=args.initial_joint_noise,
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return

    if args.dataset_root is not None:
        raise ValueError("多 worker 模式不能使用 --dataset-root；请使用 --staging-root")
    staging = args.staging_root or PROJECT/"data"/"staging"/args.split
    base_count, remainder = divmod(args.episodes, args.workers)
    jobs = []
    for worker_id in range(args.workers):
        count = base_count + int(worker_id < remainder)
        if count == 0:
            continue
        jobs.append({
            "root": staging/f"worker-{worker_id:02d}",
            "split": args.split,
            "requested_successes": count,
            "seed_start": args.seed_start+worker_id,
            "seed_stride": args.workers,
            "worker_id": worker_id,
            "mode": args.mode,
            "teacher_source": args.teacher_source,
            "image_storage": args.image_storage,
            "overwrite": args.overwrite,
            "initial_joint_noise": args.initial_joint_noise,
        })
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(jobs), mp_context=context) as pool:
        manifests = list(pool.map(worker_call, jobs))
    print(json.dumps(manifests, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
