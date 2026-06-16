"""RealSense camera streaming for TB6-R5 ACT inference."""

from __future__ import annotations

import threading
import time

import numpy as np


def to_rgb_hwc_uint8(color: np.ndarray, height: int, width: int) -> np.ndarray:
    """RealSense color stream is already RGB HWC uint8; resize if needed."""
    arr = np.asarray(color)
    if arr.ndim == 3 and (arr.shape[0] != height or arr.shape[1] != width):
        import cv2

        arr = cv2.resize(arr, (width, height), interpolation=cv2.INTER_AREA)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def show_camera_rgb(images: dict[str, np.ndarray]) -> None:
    """Show RealSense RGB frames in OpenCV windows (RGB -> BGR for imshow)."""
    import cv2

    for name, rgb in images.items():
        if rgb is None:
            continue
        bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
        window = f"ACT RGB - {name}"
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(window, bgr)
    cv2.waitKey(1)


def parse_camera_serials(spec: str | None, default_serial_dict: dict[str, str]) -> dict[str, str]:
    if not spec:
        return dict(default_serial_dict)
    out: dict[str, str] = {}
    for pair in spec.split(","):
        pair = pair.strip()
        if not pair:
            continue
        name, _, serial = pair.partition("=")
        if not name or not serial:
            raise ValueError(f"Invalid --camera-serials entry: '{pair}' (expected name=serial)")
        out[name.strip()] = serial.strip()
    return out


class CameraStream:
    """Owns a RealSenseCameraInterface plus a background polling thread."""

    def __init__(self, serial_dict: dict[str, str], width: int, height: int, fps: int):
        from xrobotoolkit_teleop.hardware.interface.realsense import RealSenseCameraInterface

        self.serial_dict = serial_dict
        self.serial_to_name = {serial: name for name, serial in serial_dict.items()}
        self.width = width
        self.height = height
        self.cam = RealSenseCameraInterface(
            width=width,
            height=height,
            fps=fps,
            serial_numbers=list(serial_dict.values()),
            enable_depth=False,
            enable_compression=False,
        )
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.cam.start()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.cam.update_frames()
            except Exception as exc:
                print(f"[ACT][camera] update_frames error: {exc}")
                time.sleep(0.02)

    def wait_ready(self, timeout_s: float = 10.0) -> None:
        deadline = time.time() + timeout_s
        needed = set(self.serial_dict.values())
        while time.time() < deadline:
            frames = self.cam.get_frames()
            if needed.issubset(set(frames.keys())):
                print("[ACT][camera] all cameras streaming")
                return
            time.sleep(0.1)
        print("[ACT][camera] WARNING: not all cameras produced frames before timeout")

    def get_images(self) -> dict[str, np.ndarray]:
        frames = self.cam.get_frames()
        out: dict[str, np.ndarray] = {}
        for serial, name in self.serial_to_name.items():
            fd = frames.get(serial)
            if fd is not None and fd.get("color") is not None:
                out[name] = to_rgb_hwc_uint8(fd["color"], self.height, self.width)
        return out

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        try:
            self.cam.stop()
        except Exception:
            pass
