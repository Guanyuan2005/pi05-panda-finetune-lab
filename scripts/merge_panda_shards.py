#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from pi05_local.dataset import (
    append_jsonl,
    create_lerobot_dataset,
    item_to_frame,
    read_jsonl,
    read_manifest,
    safe_remove_tree,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge finalized worker shards episode-by-episode.")
    parser.add_argument("--shards-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def discover(shards_root: Path) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    for manifest_path in sorted(shards_root.rglob("meta/pi05_manifest.json")):
        shard = manifest_path.parent.parent
        manifest = read_manifest(shard)
        result.setdefault(manifest["split"], []).append(shard)
    if not result:
        raise FileNotFoundError(f"没有找到有效分片: {shards_root}")
    return result


def merge_split(split: str, shards: list[Path], output: Path) -> dict:
    records = []
    manifests = []
    for shard in shards:
        manifest = read_manifest(shard)
        manifests.append(manifest)
        for episode in read_jsonl(shard/"meta"/"pi05_episodes.jsonl"):
            records.append((int(episode["scene_seed"]), shard, episode))
    records.sort(key=lambda entry: entry[0])
    if len({seed for seed, _, _ in records}) != len(records):
        raise ValueError(f"{split} 分片中存在重复 scene_seed")
    image_storage = manifests[0]["image_storage"]
    if any(manifest["image_storage"] != image_storage for manifest in manifests):
        raise ValueError("不能合并 image_storage 不同的分片")
    dataset = create_lerobot_dataset(output, f"local/panda_multi_object_box_v2_{split}", image_storage)
    cache: dict[Path, LeRobotDataset] = {}
    try:
        for new_index, (_seed, shard, record) in enumerate(records):
            if shard not in cache:
                cache[shard] = LeRobotDataset(
                    read_manifest(shard)["repo_id"], root=shard, video_backend="pyav"
                )
            source = cache[shard]
            old_index = int(record["episode_index"])
            subset = LeRobotDataset(
                source.repo_id, root=shard, episodes=[old_index], video_backend="pyav"
            )
            for item_index in range(len(subset)):
                dataset.add_frame(item_to_frame(subset[item_index]))
            dataset.save_episode(parallel_encoding=False)
            merged = dict(record)
            merged["source_shard"] = shard.as_posix()
            merged["source_episode_index"] = old_index
            merged["episode_index"] = new_index
            append_jsonl(output/"meta"/"pi05_episodes.jsonl", merged)
    finally:
        dataset.finalize()
    manifest = {
        "format": "LeRobotDataset-v3",
        "repo_id": f"local/panda_multi_object_box_v2_{split}",
        "split": split,
        "fps": manifests[0]["fps"],
        "image_storage": image_storage,
        "successful_episodes": len(records),
        "source_shards": [path.as_posix() for path in shards],
    }
    write_json(output/"meta"/"pi05_manifest.json", manifest)
    return manifest


def main() -> None:
    args = parse_args()
    output = args.output_root.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"输出目录已存在: {output}; 使用 --overwrite 明确覆盖")
        safe_remove_tree(output)
    output.mkdir(parents=True)
    result = {}
    for split, shards in discover(args.shards_root.resolve()).items():
        result[split] = merge_split(split, shards, output/split)
    all_seeds = [
        (split, int(record["scene_seed"]))
        for split in result
        for record in read_jsonl(output/split/"meta"/"pi05_episodes.jsonl")
    ]
    seed_values = [seed for _, seed in all_seeds]
    if len(seed_values) != len(set(seed_values)):
        raise ValueError("合并后的不同 split 之间存在重复 scene_seed")
    write_json(output/"manifest.json", {"splits": result, "total_episodes": len(all_seeds)})
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
