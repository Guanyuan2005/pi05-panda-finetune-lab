#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from pi05_local.dataset import directory_checksums, safe_remove_tree, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a checksummed cloud upload bundle.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"输出已存在: {output}; 使用 --overwrite")
        safe_remove_tree(output)
    output.mkdir(parents=True)
    shutil.copytree(args.dataset_root.resolve(), output/"dataset")
    shutil.copytree(PROJECT/"configs", output/"configs")
    shutil.copytree(PROJECT/"pi05_local", output/"pi05_local", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(PROJECT/"scripts", output/"scripts", ignore=shutil.ignore_patterns("__pycache__", ".gitkeep"))
    reports = PROJECT/"reports"
    if reports.exists():
        shutil.copytree(reports, output/"reports", ignore=shutil.ignore_patterns("sample_videos"))
    test_scenes = PROJECT/"data"/"test_scenes.json"
    if test_scenes.exists():
        shutil.copy2(test_scenes, output/"test_scenes.json")
    for name in ("README.md", "requirements-local.txt", "π0.5微调复现与部署指南.md", "算力平台申请前的本地工作清单.md", "本地实现与验证说明.md"):
        source = PROJECT/name
        if source.exists():
            shutil.copy2(source, output/name)
    checksums = directory_checksums(output)
    write_json(output/"SHA256SUMS.json", checksums)
    summary = {"files": len(checksums), "bytes": sum(item["bytes"] for item in checksums), "output": output.as_posix()}
    write_json(output/"bundle_manifest.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
