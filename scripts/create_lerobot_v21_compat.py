#!/usr/bin/env python3
"""Create a non-destructive LeRobot v2.1 metadata view for OpenPI."""

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    meta_out = output / "meta"
    meta_out.mkdir(exist_ok=True)

    data_out = output / "data" / "chunk-000"
    data_link = output / "data"
    if data_link.is_symlink():
        data_link.unlink()
    data_out.mkdir(parents=True, exist_ok=True)

    info = json.loads((source / "meta/info.json").read_text())
    info["codebase_version"] = "v2.0"
    info["data_path"] = "data/chunk-{episode_chunk:03d}/file-{episode_index:03d}.parquet"
    (meta_out / "info.json").write_text(json.dumps(info, indent=2) + "\n")
    (meta_out / "stats.json").write_bytes((source / "meta/stats.json").read_bytes())

    task_table = pq.read_table(source / "meta/tasks.parquet")
    task_data = task_table.to_pydict()
    task_texts = task_data["__index_level_0__"]
    tasks = [
        {"task_index": int(index), "task": text}
        for index, text in zip(task_data["task_index"], task_texts, strict=True)
    ]
    write_jsonl(meta_out / "tasks.jsonl", tasks)
    task_by_index = {row["task_index"]: row["task"] for row in tasks}

    grouped = {}
    data_files = sorted((source / "data").glob("chunk-*/file-*.parquet"))
    for path in data_files:
        table = pq.read_table(path).replace_schema_metadata(None)
        for image_key in ("observation.images.image", "observation.images.wrist_image"):
            col = table[image_key]
            if pa.types.is_struct(col.type):
                table = table.set_column(table.schema.get_field_index(image_key), image_key, pc.struct_field(col, "bytes"))
        episode_values = table.column("episode_index").to_pylist()
        task_values = table.column("task_index").to_pylist()
        for episode_index, task_index in zip(
            episode_values,
            task_values,
            strict=True,
        ):
            item = grouped.setdefault(int(episode_index), {"task_index": int(task_index), "length": 0})
            item["length"] += 1
            if item["task_index"] != int(task_index):
                raise RuntimeError(f"episode {episode_index} has multiple tasks")
        for episode_index in sorted(set(map(int, episode_values))):
            mask = [int(x) == episode_index for x in episode_values]
            episode_table = table.filter(mask)
            pq.write_table(episode_table, data_out / f"file-{episode_index:03d}.parquet")
    episodes = [
        {
            "episode_index": episode_index,
            "tasks": [task_by_index[item["task_index"]]],
            "length": item["length"],
        }
        for episode_index, item in sorted(grouped.items())
    ]
    write_jsonl(meta_out / "episodes.jsonl", episodes)

    total_frames = sum(row["length"] for row in episodes)
    if len(episodes) != info["total_episodes"] or total_frames != info["total_frames"]:
        raise RuntimeError(
            f"metadata mismatch: episodes={len(episodes)}, frames={total_frames}"
        )
    print(f"created={output} episodes={len(episodes)} frames={total_frames}")


if __name__ == "__main__":
    main()
