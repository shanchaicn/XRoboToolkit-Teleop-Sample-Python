#!/usr/bin/env python3
"""
Merge multiple LeRobot v3 datasets under one folder into a single dataset.

Uses the official ``lerobot.datasets.dataset_tools.merge_datasets`` API
(same as ``lerobot-edit-dataset --operation.type merge``).

A subfolder is treated as a dataset when it contains ``meta/info.json``.

Example (merge all TASK folders under tb6r5_rings):
    python scripts/misc/merge_lerobot_datasets_in_folder.py \
        --input-dir data/lerobot/tb6r5_rings \
        --output-root data/lerobot/tb6r5_rings/merged \
        --repo-id local/tb6r5_rings_merged \
        --exclude merged \
        --overwrite

Merge only selected datasets:
    python scripts/misc/merge_lerobot_datasets_in_folder.py \
        --input-dir data/lerobot/tb6r5_rings \
        --output-root data/lerobot/tb6r5_rings/merged \
        --repo-id local/tb6r5_rings_merged \
        --include P02 S01
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def _is_lerobot_dataset_dir(path: Path) -> bool:
    return (path / "meta" / "info.json").is_file()


def _load_info(path: Path) -> dict:
    return json.loads((path / "meta" / "info.json").read_text(encoding="utf-8"))


def discover_datasets(
    input_dir: Path,
    *,
    include: list[str] | None,
    exclude: set[str],
    repo_prefix: str,
) -> list[tuple[str, Path, str]]:
    """Return list of (name, root, repo_id) sorted by name."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input-dir not found: {input_dir}")

    if include:
        candidates = [input_dir / name for name in include]
        missing = [p.name for p in candidates if not p.is_dir()]
        if missing:
            raise FileNotFoundError(f"included folders not found under {input_dir}: {missing}")
    else:
        candidates = sorted(p for p in input_dir.iterdir() if p.is_dir())

    datasets: list[tuple[str, Path, str]] = []
    for path in candidates:
        name = path.name
        if name in exclude:
            continue
        if not _is_lerobot_dataset_dir(path):
            continue
        repo_id = f"{repo_prefix}{name}" if repo_prefix else name
        datasets.append((name, path.resolve(), repo_id))

    return datasets


def print_dataset_summary(datasets: list[tuple[str, Path, str]]) -> None:
    print(f"{'name':<12} {'episodes':>8} {'frames':>8} {'fps':>4}  repo_id")
    print("-" * 60)
    for name, root, repo_id in datasets:
        info = _load_info(root)
        print(
            f"{name:<12} {info.get('total_episodes', '?'):>8} "
            f"{info.get('total_frames', '?'):>8} {info.get('fps', '?'):>4}  {repo_id}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge multiple LeRobot v3 datasets from subfolders into one dataset."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Parent folder containing dataset subfolders (each with meta/info.json).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Output directory for the merged dataset.",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="repo_id for the merged dataset, e.g. local/tb6r5_rings_merged",
    )
    parser.add_argument(
        "--repo-prefix",
        default="local/",
        help="Prefix prepended to each source folder name to form source repo_id (default: local/).",
    )
    parser.add_argument(
        "--include",
        nargs="+",
        default=None,
        help="Only merge these subfolder names. Default: all valid datasets under input-dir.",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Subfolder names to skip (e.g. merged output folder).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output-root if it already exists before merging.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List source datasets only; do not merge.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_root = args.output_root.resolve()

    exclude = set(args.exclude)
    # Always skip output folder if it lives inside input-dir.
    if output_root.parent == input_dir:
        exclude.add(output_root.name)

    datasets = discover_datasets(
        input_dir,
        include=args.include,
        exclude=exclude,
        repo_prefix=args.repo_prefix,
    )

    if len(datasets) < 2:
        print(
            f"Need at least 2 LeRobot datasets under {input_dir}; found {len(datasets)}.",
            file=sys.stderr,
        )
        if datasets:
            print("Found:", ", ".join(d[0] for d in datasets), file=sys.stderr)
        sys.exit(1)

    print(f"Input : {input_dir}")
    print(f"Output: {output_root}")
    print(f"repo_id: {args.repo_id}")
    print(f"Sources ({len(datasets)}):")
    print_dataset_summary(datasets)

    if args.dry_run:
        print("Dry run only; no merge performed.")
        return

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{output_root} already exists. Use --overwrite or choose another --output-root."
            )
        shutil.rmtree(output_root)

    output_root.parent.mkdir(parents=True, exist_ok=True)

    lerobot_datasets = [
        LeRobotDataset(repo_id=repo_id, root=str(root)) for _, root, repo_id in datasets
    ]

    print("Merging...")
    merged = merge_datasets(
        lerobot_datasets,
        output_repo_id=args.repo_id,
        output_dir=str(output_root),
    )

    print("Done.")
    print(f"Merged dataset: {output_root}")
    print(f"  episodes: {merged.meta.total_episodes}")
    print(f"  frames  : {merged.meta.total_frames}")
    print(f"  tasks   : {merged.meta.total_tasks}")


if __name__ == "__main__":
    main()
