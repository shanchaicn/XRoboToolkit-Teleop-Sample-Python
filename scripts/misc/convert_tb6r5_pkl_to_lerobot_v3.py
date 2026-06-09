#!/usr/bin/env python3
"""
Convert TB6R5 teleop pickle logs to a LeRobot 3.x dataset (prefer video storage).

This script is inspired by:
    openpi/finetuning/examples/arx_r5/arx_dual/convert_dual_arm_data_to_lerobot.py

Key differences vs older converters:
- Uses LeRobot "video" features (when supported) to keep dataset size reasonable.
- Robust to missing per-frame camera streams (fills with last valid frame / black frame).
- Can unpickle logs even if `pyrealsense2` is not installed (compatibility stub).
- Streams one pickle episode at a time (avoids loading all logs into RAM).
- Supports --resume to continue a partially converted dataset.

Example (merge all subfolders under logs/ into one dataset):
    python scripts/misc/convert_tb6r5_pkl_to_lerobot_v3.py \
        logs \
        --root ./data/lerobot/tb6r5_merged \
        --repo-id local/tb6r5_merged \
        --fps 50 \
        --overwrite

Resume after interruption (keeps existing episodes, continues from manifest):
    python scripts/misc/convert_tb6r5_pkl_to_lerobot_v3.py \
        logs \
        --root ./data/lerobot/tb6r5_merged \
        --repo-id local/tb6r5_merged \
        --fps 50 \
        --resume
"""

from __future__ import annotations

import argparse
import gc
import json
import pickle
import shutil
import sys
import types
from pathlib import Path

import cv2
import numpy as np

MANIFEST_NAME = ".tb6r5_convert_done.txt"


def _install_pyrealsense2_pickle_stub():
    """Allow unpickling logs on machines/envs without pyrealsense2 installed."""

    class DummyFormat:
        def __new__(cls, *args, **kwargs):
            return object.__new__(cls)

        def __setstate__(self, state):
            self.state = state

    if "pyrealsense2.pyrealsense2" in sys.modules:
        return

    pkg = types.ModuleType("pyrealsense2")
    pkg.__path__ = []
    sub = types.ModuleType("pyrealsense2.pyrealsense2")
    sub.format = DummyFormat
    pkg.pyrealsense2 = sub
    sys.modules.setdefault("pyrealsense2", pkg)
    sys.modules.setdefault("pyrealsense2.pyrealsense2", sub)


def _load_pickle(path: Path) -> list[dict]:
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except ModuleNotFoundError as exc:
        if "pyrealsense2" not in str(exc):
            raise
        _install_pyrealsense2_pickle_stub()
        with path.open("rb") as f:
            return pickle.load(f)


def _decode_jpg(data) -> np.ndarray | None:
    """Decode jpg bytes (or pass-through ndarray) to HWC uint8 RGB."""
    if data is None:
        return None
    if isinstance(data, bytes):
        arr = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    elif isinstance(data, np.ndarray):
        image = data
    else:
        return None

    if image is None:
        return None

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif image.ndim == 3 and image.shape[-1] == 3:
        pass
    else:
        return None

    return image.astype(np.uint8, copy=False)


def _collect_input_files(inputs: list[Path], recursive: bool = True) -> list[Path]:
    """Collect teleop_log_*.pkl from files or directories (optionally recursive)."""
    files: list[Path] = []
    for p in inputs:
        if p.is_dir():
            if recursive:
                found = sorted(p.rglob("teleop_log_*.pkl"))
            else:
                found = sorted(p.glob("teleop_log_*.pkl"))
            files.extend(found)
        elif p.is_file():
            files.append(p)
    return sorted(set(files), key=lambda f: (str(f.parent), f.name))


def _manifest_path(root: Path) -> Path:
    return root / MANIFEST_NAME


def _load_done_manifest(root: Path) -> set[str]:
    path = _manifest_path(root)
    if not path.is_file():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def _append_done_manifest(root: Path, pkl_path: Path) -> None:
    path = _manifest_path(root)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{pkl_path.resolve()}\n")


def _infer_camera_names_from_episode(episode: list[dict]) -> list[str]:
    camera_names = set()
    for entry in episode:
        images = entry.get("image")
        if isinstance(images, dict):
            camera_names.update(images.keys())
    return sorted(camera_names)


def _find_first_color_image(episode: list[dict]) -> np.ndarray:
    for entry in episode:
        images = entry.get("image")
        if not isinstance(images, dict):
            continue
        for camera_data in images.values():
            if not isinstance(camera_data, dict):
                continue
            image = _decode_jpg(camera_data.get("color"))
            if image is not None:
                return image
    raise ValueError("No color image found in episode.")


def _probe_first_valid_episode(files: list[Path], max_probe: int = 5) -> tuple[Path, list[dict]]:
    for path in files[:max_probe]:
        episode = _load_pickle(path)
        if episode:
            return path, episode
    raise ValueError("No non-empty episode found in input logs.")


def _read_existing_episode_count(root: Path) -> int:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        return 0
    with info_path.open(encoding="utf-8") as f:
        info = json.load(f)
    return int(info.get("total_episodes", 0))


def _create_dataset(args, features: dict):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    kwargs = {
        "repo_id": args.repo_id,
        "root": str(args.root),
        "robot_type": args.robot_type,
        "fps": args.fps,
        "features": features,
        "use_videos": not args.no_videos,
    }
    if args.image_writer_processes is not None:
        kwargs["image_writer_processes"] = args.image_writer_processes
    if args.image_writer_threads is not None:
        kwargs["image_writer_threads"] = args.image_writer_threads
    if args.video_backend is not None:
        kwargs["video_backend"] = args.video_backend

    try:
        return LeRobotDataset.create(**kwargs)
    except TypeError:
        for key in ("use_videos", "image_writer_processes", "image_writer_threads", "video_backend"):
            kwargs.pop(key, None)
        return LeRobotDataset.create(**kwargs)


def _open_dataset_for_resume(args):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    kwargs = {"repo_id": args.repo_id, "root": str(args.root)}
    if args.video_backend is not None:
        kwargs["video_backend"] = args.video_backend
    try:
        return LeRobotDataset(**kwargs)
    except TypeError:
        kwargs.pop("video_backend", None)
        return LeRobotDataset(**kwargs)


def _add_frame(dataset, frame: dict, task: str):
    try:
        dataset.add_frame(frame, task=task)
        return
    except TypeError:
        pass
    if hasattr(dataset, "add_episode_frame"):
        frame = dict(frame)
        frame["task"] = task
        dataset.add_episode_frame(frame)
    else:
        frame = dict(frame)
        frame["task"] = task
        dataset.add_frame(frame)


def _export_episode(
    dataset,
    episode: list[dict],
    camera_names: list[str],
    height: int,
    width: int,
    channels: int,
    task: str,
    include_depth: bool,
) -> int:
    black = np.zeros((height, width, channels), dtype=np.uint8)
    last_color: dict[str, np.ndarray] = {cam: black for cam in camera_names}
    last_depth: dict[str, np.ndarray] = {cam: black for cam in camera_names}
    frame_count = 0

    for entry in episode:
        frame = {
            "observation.state": np.asarray(entry["observation"], dtype=np.float32),
            "action": np.asarray(entry["action"], dtype=np.float32),
        }

        images = entry.get("image") or {}
        if isinstance(images, dict):
            for cam in camera_names:
                cam_data = images.get(cam)
                if isinstance(cam_data, dict):
                    color = _decode_jpg(cam_data.get("color"))
                    if color is not None:
                        last_color[cam] = color
                    if include_depth:
                        depth = _decode_jpg(cam_data.get("depth"))
                        if depth is not None:
                            last_depth[cam] = depth

                frame[f"observation.images.{cam}"] = last_color[cam]
                if include_depth:
                    frame[f"observation.images.{cam}.depth"] = last_depth[cam]

        _add_frame(dataset, frame, task=task)
        frame_count += 1

    return frame_count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Pickle files or directories. Directories are scanned recursively for teleop_log_*.pkl "
        "(pass logs/ to merge all session subfolders).",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only read teleop_log_*.pkl in the top level of each input directory (not subfolders).",
    )
    parser.add_argument("--root", type=Path, required=True, help="Output dataset root directory")
    parser.add_argument("--repo-id", required=True, help="LeRobot repo_id (e.g. local/tb6r5-20260602-pnp)")
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--task", default="tb6r5 teleoperation")
    parser.add_argument("--robot-type", default="tb6r5")
    parser.add_argument("--overwrite", action="store_true", help="Delete output root if it already exists")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue converting into an existing dataset (uses manifest + skips done pkls).",
    )

    parser.add_argument("--no-videos", action="store_true", help="Store images as files instead of videos if supported")
    parser.add_argument("--video-backend", default=None, help="Optional LeRobot video_backend value")
    parser.add_argument(
        "--image-writer-processes",
        type=int,
        default=None,
        help="LeRobot image writer processes (omit or use 1 to limit RAM spikes).",
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=None,
        help="LeRobot image writer threads per process.",
    )
    parser.add_argument(
        "--include-depth",
        action="store_true",
        help="Also export depth stream (stored as RGB uint8 after decoding). Can increase size a lot.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Convert at most N episode files (for testing).",
    )
    args = parser.parse_args()

    if args.overwrite and args.resume:
        raise ValueError("Use only one of --overwrite or --resume.")

    files = _collect_input_files(args.inputs, recursive=not args.no_recursive)
    if not files:
        raise FileNotFoundError("No input pickle files found.")

    by_parent: dict[str, int] = {}
    for f in files:
        by_parent[str(f.parent)] = by_parent.get(str(f.parent), 0) + 1
    print(f"Found {len(files)} episode file(s) from {len(by_parent)} folder(s):")
    for parent, count in sorted(by_parent.items()):
        print(f"  {parent}: {count}")

    done_paths = set()
    if args.resume:
        if not args.root.exists():
            raise FileNotFoundError(f"--resume requested but {args.root} does not exist.")
        done_paths = _load_done_manifest(args.root)
        if not done_paths:
            existing = _read_existing_episode_count(args.root)
            if existing > 0:
                print(
                    f"[resume] No manifest yet; skipping first {existing} file(s) by meta/info.json episode count."
                )
                done_paths = {str(f.resolve()) for f in files[:existing]}
        pending = [f for f in files if str(f.resolve()) not in done_paths]
        print(f"[resume] {len(done_paths)} done, {len(pending)} pending.")
        files = pending
        if args.max_episodes is not None:
            files = files[: args.max_episodes]
        if not files:
            print("Nothing left to convert.")
            return 0
    else:
        if args.root.exists():
            if not args.overwrite:
                raise FileExistsError(
                    f"{args.root} already exists. Use --overwrite, --resume, or another --root."
                )
            shutil.rmtree(args.root)
        if args.max_episodes is not None:
            files = files[: args.max_episodes]

    probe_path, probe_episode = _probe_first_valid_episode(files)
    sample_image = _find_first_color_image(probe_episode)
    height, width, channels = sample_image.shape
    first_entry = probe_episode[0]
    state_dim = int(np.asarray(first_entry["observation"]).size)
    action_dim = int(np.asarray(first_entry["action"]).size)
    camera_names = _infer_camera_names_from_episode(probe_episode)
    del probe_episode
    gc.collect()

    print(f"[probe] Using {probe_path} for feature shapes.")
    print(f"  state_dim={state_dim}, action_dim={action_dim}, cameras={camera_names}, image={height}x{width}x{channels}")

    vision_dtype = "image" if args.no_videos else "video"
    features: dict = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": [f"state_{i}" for i in range(state_dim)],
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": [f"action_{i}" for i in range(action_dim)],
        },
    }
    for cam in camera_names:
        features[f"observation.images.{cam}"] = {
            "dtype": vision_dtype,
            "shape": (height, width, channels),
            "names": ["height", "width", "channel"],
        }
        if args.include_depth:
            features[f"observation.images.{cam}.depth"] = {
                "dtype": vision_dtype,
                "shape": (height, width, channels),
                "names": ["height", "width", "channel"],
            }

    if args.resume:
        dataset = _open_dataset_for_resume(args)
        print(f"[resume] Opened existing dataset at {args.root}")
    else:
        dataset = _create_dataset(args, features)
        print(f"Created dataset at {args.root}")

    total_frames = 0
    converted = 0

    for idx, pkl_path in enumerate(files, start=1):
        print(f"[{idx}/{len(files)}] Loading {pkl_path} ...", flush=True)
        episode = _load_pickle(pkl_path)
        if not episode:
            print(f"  skip empty episode")
            continue

        n_frames = _export_episode(
            dataset,
            episode,
            camera_names,
            height,
            width,
            channels,
            args.task,
            args.include_depth,
        )
        del episode
        gc.collect()

        dataset.save_episode()
        _append_done_manifest(args.root, pkl_path)
        converted += 1
        total_frames += n_frames
        print(f"  saved {n_frames} frames (episode {converted})", flush=True)

    if hasattr(dataset, "consolidate"):
        print("Consolidating dataset ...", flush=True)
        dataset.consolidate()

    print(f"Done. Added {converted} episode(s), {total_frames} frame(s) -> {args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
