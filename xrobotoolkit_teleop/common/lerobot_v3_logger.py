"""
Online LeRobot v3 dataset writer for TB6-R5 teleoperation.

Follows the same pattern as ``lerobot.scripts.lerobot_record``:
  LeRobotDataset.create → add_frame (each timestep) → save_episode (per segment)

Reference: https://github.com/huggingface/lerobot
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from xrobotoolkit_teleop.utils.image_utils import decompress_jpg_to_image


def _rgb_uint8_hwc(image: np.ndarray) -> np.ndarray:
    if image is None:
        return None
    img = np.asarray(image)
    if img.ndim == 2:
        import cv2

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3 and img.shape[-1] == 4:
        import cv2

        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    return img.astype(np.uint8, copy=False)


def _decode_camera_frame(cam_data: dict) -> Optional[np.ndarray]:
    if not isinstance(cam_data, dict):
        return None
    color = cam_data.get("color")
    if color is None:
        return None
    if isinstance(color, bytes):
        return _rgb_uint8_hwc(decompress_jpg_to_image(color))
    return _rgb_uint8_hwc(color)


def build_tb6r5_lerobot_features(
    state_dim: int,
    action_dim: int,
    camera_names: list[str],
    height: int,
    width: int,
    channels: int = 3,
    use_videos: bool = True,
    include_depth: bool = False,
) -> dict:
    """Feature schema aligned with ``lerobot_record`` / ``convert_tb6r5_pkl_to_lerobot_v3``."""
    vision_dtype = "image" if not use_videos else "video"
    state_names = [f"state_{i}" for i in range(state_dim)]
    action_names = [f"action_{i}" for i in range(action_dim)]
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": state_names,
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": action_names,
        },
    }
    for cam in camera_names:
        features[f"observation.images.{cam}"] = {
            "dtype": vision_dtype,
            "shape": (height, width, channels),
            "names": ["height", "width", "channel"],
        }
        if include_depth:
            features[f"observation.images.{cam}.depth"] = {
                "dtype": vision_dtype,
                "shape": (height, width, channels),
                "names": ["height", "width", "channel"],
            }
    return features


class TB6LeRobotV3Logger:
    """Write teleop frames directly into a LeRobot v3 dataset (one episode per B segment)."""

    def __init__(
        self,
        root: str | Path,
        repo_id: str,
        fps: int,
        task: str,
        robot_type: str = "tb6r5",
        camera_names: Optional[list[str]] = None,
        state_dim: int = 7,
        action_dim: int = 7,
        image_height: int = 480,
        image_width: int = 640,
        use_videos: bool = True,
        include_depth: bool = False,
        overwrite: bool = False,
        resume: bool = False,
        streaming_encoding: bool = True,
        vcodec: str = "libsvtav1",
        image_writer_processes: int = 0,
        image_writer_threads: int = 4,
        encoder_threads: Optional[int] = 2,
        encoder_queue_maxsize: int = 30,
    ):
        self.root = Path(root)
        self.repo_id = repo_id
        self.fps = fps
        self.task = task
        self.camera_names = list(camera_names or [])
        self.include_depth = include_depth
        self._last_color: Dict[str, np.ndarray] = {}
        self._last_depth: Dict[str, np.ndarray] = {}
        self._black: Optional[np.ndarray] = None

        if self.root.exists():
            if overwrite:
                shutil.rmtree(self.root)
            elif not resume:
                raise FileExistsError(
                    f"{self.root} exists. Use overwrite=True, resume=True, or another root."
                )

        features = build_tb6r5_lerobot_features(
            state_dim=state_dim,
            action_dim=action_dim,
            camera_names=self.camera_names,
            height=image_height,
            width=image_width,
            use_videos=use_videos,
            include_depth=include_depth,
        )

        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        if resume and self.root.exists():
            self.dataset = LeRobotDataset(
                repo_id=repo_id,
                root=str(self.root),
                streaming_encoding=streaming_encoding,
                vcodec=vcodec,
                encoder_threads=encoder_threads,
                encoder_queue_maxsize=encoder_queue_maxsize,
            )
            print(f"[LeRobot] Resumed dataset at {self.root} ({self.dataset.num_episodes} episodes)")
        else:
            kwargs = {
                "repo_id": repo_id,
                "fps": fps,
                "root": str(self.root),
                "robot_type": robot_type,
                "features": features,
                "use_videos": use_videos,
                "vcodec": vcodec,
                "streaming_encoding": streaming_encoding,
                "encoder_queue_maxsize": encoder_queue_maxsize,
                "encoder_threads": encoder_threads,
            }
            if image_writer_processes or image_writer_threads:
                kwargs["image_writer_processes"] = image_writer_processes
                kwargs["image_writer_threads"] = image_writer_threads
            try:
                self.dataset = LeRobotDataset.create(**kwargs)
            except TypeError:
                for key in (
                    "streaming_encoding",
                    "encoder_queue_maxsize",
                    "encoder_threads",
                    "image_writer_processes",
                    "image_writer_threads",
                ):
                    kwargs.pop(key, None)
                self.dataset = LeRobotDataset.create(**kwargs)
            print(f"[LeRobot] Created dataset at {self.root}")

        if self.camera_names:
            h, w = image_height, image_width
            self._black = np.zeros((h, w, 3), dtype=np.uint8)
            self._last_color = {cam: self._black.copy() for cam in self.camera_names}
            if include_depth:
                self._last_depth = {cam: self._black.copy() for cam in self.camera_names}

        if not streaming_encoding:
            print(
                "[LeRobot] Tip: enable streaming_encoding=True for near-instant save_episode "
                "(see https://huggingface.co/docs/lerobot/streaming_video_encoding)"
            )

    def begin_episode(self) -> None:
        """Clear buffer before a new B-press recording segment."""
        self.dataset.clear_episode_buffer(delete_images=True)

    def add_tb6_entry(self, entry: dict) -> None:
        """Add one teleop timestep (same dict as pkl logging)."""
        frame = {
            "observation.state": np.asarray(entry["observation"], dtype=np.float32),
            "action": np.asarray(entry["action"], dtype=np.float32),
        }

        images = entry.get("image") or {}
        if isinstance(images, dict) and self.camera_names:
            for cam in self.camera_names:
                cam_data = images.get(cam)
                color = _decode_camera_frame(cam_data) if isinstance(cam_data, dict) else None
                if color is not None:
                    self._last_color[cam] = color
                frame[f"observation.images.{cam}"] = self._last_color.get(cam, self._black)

                if self.include_depth:
                    depth_img = None
                    if isinstance(cam_data, dict):
                        depth = cam_data.get("depth")
                        if isinstance(depth, bytes):
                            depth_img = _rgb_uint8_hwc(decompress_jpg_to_image(depth))
                        elif depth is not None:
                            depth_img = _rgb_uint8_hwc(depth)
                    if depth_img is not None:
                        self._last_depth[cam] = depth_img
                    frame[f"observation.images.{cam}.depth"] = self._last_depth.get(cam, self._black)

        frame["task"] = self.task
        self.dataset.add_frame(frame)

    def save_episode(self) -> int:
        """Flush current buffer to disk (parquet + videos). Returns episode length."""
        size = self.dataset.episode_buffer.get("size", 0) if self.dataset.episode_buffer else 0
        if size == 0:
            print("[LeRobot] No frames in buffer; skip save_episode")
            return 0
        self.dataset.save_episode()
        print(f"[LeRobot] Saved episode #{self.dataset.num_episodes - 1} ({size} frames)")
        return int(size)

    def discard_episode(self) -> None:
        self.dataset.clear_episode_buffer(delete_images=True)
        print("[LeRobot] Discarded in-progress episode buffer")

    def finalize(self) -> None:
        if hasattr(self.dataset, "finalize"):
            self.dataset.finalize()
        print(f"[LeRobot] Dataset finalized at {self.root}")
