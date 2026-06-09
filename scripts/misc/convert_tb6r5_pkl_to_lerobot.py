"""Convert TB6R5 teleop pickle logs to a LeRobot dataset.

Example:
    python scripts/misc/convert_tb6r5_pkl_to_lerobot.py \
        logs/tb6r5/teleop_log_20260529_115024_1.pkl \
        --root data/lerobot/tb6r5 \
        --repo-id local/tb6r5 \
        --fps 50
"""

import argparse
import pickle
import shutil
import sys
import types
from pathlib import Path

import cv2
import numpy as np


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
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    return image.astype(np.uint8, copy=False)


def _collect_input_files(paths: list[Path], recursive: bool = True) -> list[Path]:
    files = []
    for path in paths:
        if path.is_dir():
            if recursive:
                files.extend(path.rglob("teleop_log_*.pkl"))
            else:
                files.extend(path.glob("teleop_log_*.pkl"))
        elif path.is_file():
            files.append(path)
    return sorted(set(files), key=lambda f: (str(f.parent), f.name))


def _find_first_image(data_by_episode: list[list[dict]], include_depth: bool):
    streams = ("color", "depth") if include_depth else ("color",)
    for episode in data_by_episode:
        for entry in episode:
            images = entry.get("image")
            if not isinstance(images, dict):
                continue
            for camera_name, camera_data in images.items():
                if not isinstance(camera_data, dict):
                    continue
                for stream in streams:
                    image = _decode_jpg(camera_data.get(stream))
                    if image is not None:
                        return camera_name, stream, image
    raise ValueError("No camera image found in input logs.")


def _build_features(data_by_episode: list[list[dict]], include_depth: bool) -> dict:
    _, _, sample_image = _find_first_image(data_by_episode, include_depth)
    height, width, channels = sample_image.shape
    first_entry = data_by_episode[0][0]
    state_dim = int(np.asarray(first_entry["observation"]).size)
    action_dim = int(np.asarray(first_entry["action"]).size)

    features = {
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

    streams = ("color", "depth") if include_depth else ("color",)
    camera_names = sorted(
        {
            camera_name
            for episode in data_by_episode
            for entry in episode
            for camera_name in (entry.get("image") or {}).keys()
        }
    )
    for camera_name in camera_names:
        for stream in streams:
            key = f"observation.images.{camera_name}" if stream == "color" else f"observation.images.{camera_name}.{stream}"
            features[key] = {
                "dtype": "image",
                "shape": (height, width, channels),
                "names": ["height", "width", "channel"],
            }
    return features


def _build_image_defaults(data_by_episode: list[list[dict]], include_depth: bool) -> dict[str, np.ndarray]:
    """Build fallback images so every frame has every declared image feature."""
    defaults = {}
    streams = ("color", "depth") if include_depth else ("color",)
    for episode in data_by_episode:
        for entry in episode:
            images = entry.get("image")
            if not isinstance(images, dict):
                continue
            for camera_name, camera_data in images.items():
                if not isinstance(camera_data, dict):
                    continue
                for stream in streams:
                    key = (
                        f"observation.images.{camera_name}"
                        if stream == "color"
                        else f"observation.images.{camera_name}.{stream}"
                    )
                    if key in defaults:
                        continue
                    image = _decode_jpg(camera_data.get(stream))
                    if image is not None:
                        defaults[key] = image
    return defaults


def _create_dataset(args, features: dict):
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise ImportError(
            "未安装 lerobot。请先在当前环境安装，例如：pip install lerobot"
        ) from exc

    kwargs = {
        "repo_id": args.repo_id,
        "root": args.root,
        "fps": args.fps,
        "features": features,
        "robot_type": args.robot_type,
        "use_videos": not args.no_videos,
    }
    try:
        return LeRobotDataset.create(**kwargs)
    except TypeError:
        kwargs.pop("use_videos", None)
        return LeRobotDataset.create(**kwargs)


def _add_frame(dataset, frame: dict):
    if hasattr(dataset, "add_frame"):
        dataset.add_frame(frame)
    else:
        dataset.add_episode_frame(frame)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Pickle files or directories (recursive scan for teleop_log_*.pkl; use logs/ to merge subfolders).",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only read teleop_log_*.pkl in the top level of each input directory.",
    )
    parser.add_argument("--root", type=Path, default=Path("data/lerobot/tb6r5"), help="Output dataset root")
    parser.add_argument("--repo-id", default="local/tb6r5", help="LeRobot repo_id")
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--task", default="tb6r5 teleoperation")
    parser.add_argument("--robot-type", default="tb6r5")
    parser.add_argument("--include-depth", action="store_true", help="Also export depth JPGs as image observations")
    parser.add_argument("--no-videos", action="store_true", help="Store image files instead of LeRobot videos if supported")
    parser.add_argument("--overwrite", action="store_true", help="Delete output root if it already exists")
    args = parser.parse_args()

    files = _collect_input_files(args.inputs, recursive=not args.no_recursive)
    if not files:
        raise FileNotFoundError("No input pickle files found.")
    print(f"Found {len(files)} episode file(s) under {len({f.parent for f in files})} folder(s).")

    data_by_episode = [_load_pickle(path) for path in files]
    data_by_episode = [episode for episode in data_by_episode if episode]
    if not data_by_episode:
        raise ValueError("Input logs contain no frames.")

    if args.root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.root} already exists. Use --overwrite or choose another --root path.")
        shutil.rmtree(args.root)

    features = _build_features(data_by_episode, args.include_depth)
    image_defaults = _build_image_defaults(data_by_episode, args.include_depth)
    dataset = _create_dataset(args, features)
    image_feature_keys = [key for key, spec in features.items() if spec.get("dtype") == "image"]

    streams = ("color", "depth") if args.include_depth else ("color",)
    total_frames = 0
    for path, episode in zip(files, data_by_episode):
        for entry in episode:
            frame = {
                "observation.state": np.asarray(entry["observation"], dtype=np.float32),
                "action": np.asarray(entry["action"], dtype=np.float32),
                "task": args.task,
            }

            images = entry.get("image") or {}
            for camera_name, camera_data in images.items():
                if not isinstance(camera_data, dict):
                    continue
                for stream in streams:
                    image = _decode_jpg(camera_data.get(stream))
                    if image is None:
                        continue
                    key = (
                        f"observation.images.{camera_name}"
                        if stream == "color"
                        else f"observation.images.{camera_name}.{stream}"
                    )
                    frame[key] = image

            for key in image_feature_keys:
                if key not in frame:
                    # Some entries miss a camera stream; use the first valid frame from that stream.
                    frame[key] = image_defaults[key]

            _add_frame(dataset, frame)
            total_frames += 1

        dataset.save_episode()
        print(f"Saved episode from {path} ({len(episode)} frames)")

    if hasattr(dataset, "consolidate"):
        dataset.consolidate()
    print(f"Done. Exported {len(data_by_episode)} episodes / {total_frames} frames to {args.root}")


if __name__ == "__main__":
    main()
