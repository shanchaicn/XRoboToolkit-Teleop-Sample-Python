"""Multi-backend camera capture (RealSense / V4L2 / HTTP) shared by teleop and ACT inference."""

from __future__ import annotations

import threading
import time
from typing import Literal

import numpy as np

from xrobotoolkit_teleop.hardware.interface.base_camera import BaseCameraInterface
from xrobotoolkit_teleop.utils.image_utils import compress_image_to_jpg

DEFAULT_REALSENSE_SERIAL_DICT = {
    "realsense_0": "135522071053",
    "realsense_1": "327122073649",
}

CameraBackend = Literal["realsense", "v4l2", "http"]


def parse_name_value_pairs(spec: str, *, option: str, example: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for pair in spec.split(","):
        pair = pair.strip()
        if not pair:
            continue
        name, _, value = pair.partition("=")
        if not name or not value:
            raise ValueError(f"Invalid {option} entry: '{pair}' (expected {example})")
        out[name.strip()] = value.strip()
    if not out:
        raise ValueError(f"{option} must list at least one name=value pair")
    return out


def parse_camera_serials(
    spec: str | None,
    default_serial_dict: dict[str, str] | None = None,
) -> dict[str, str]:
    if not spec:
        return dict(default_serial_dict or DEFAULT_REALSENSE_SERIAL_DICT)
    return parse_name_value_pairs(spec, option="--camera-serials", example="name=serial")


def parse_camera_devices(spec: str | None) -> dict[str, str]:
    if not spec:
        return {}
    return parse_name_value_pairs(
        spec,
        option="--camera-devices",
        example="name=/dev/video0 or name=0",
    )


def parse_camera_urls(spec: str | None) -> dict[str, str]:
    if not spec:
        return {}
    return parse_name_value_pairs(
        spec,
        option="--camera-urls",
        example="name=http://host:8888/RsCameraSensor/0/0/color",
    )


def to_rgb_hwc_uint8(color: np.ndarray, height: int, width: int) -> np.ndarray:
    arr = np.asarray(color)
    if arr.ndim == 3 and (arr.shape[0] != height or arr.shape[1] != width):
        import cv2

        arr = cv2.resize(arr, (width, height), interpolation=cv2.INTER_AREA)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


class RealSenseCameraStream:
    """RealSense via RealSenseCameraInterface + background polling."""

    backend: CameraBackend = "realsense"

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
                print(f"[camera] RealSense update_frames error: {exc}")
                time.sleep(0.02)

    def wait_ready(self, timeout_s: float = 10.0) -> None:
        deadline = time.time() + timeout_s
        needed = set(self.serial_dict.values())
        while time.time() < deadline:
            frames = self.cam.get_frames()
            if needed.issubset(set(frames.keys())):
                print("[camera] all RealSense cameras streaming")
                return
            time.sleep(0.1)
        print("[camera] WARNING: not all RealSense cameras produced frames before timeout")

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


class V4l2CameraStream:
    """Capture RGB via OpenCV VideoCapture (/dev/video* or numeric index)."""

    backend: CameraBackend = "v4l2"

    def __init__(self, device_dict: dict[str, str], width: int, height: int, fps: int):
        self.device_dict = device_dict
        self.width = width
        self.height = height
        self.fps = fps
        self._caps: dict[str, object] = {}
        self._frames: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _open_capture(device: str):
        import cv2

        if device.isdigit():
            return cv2.VideoCapture(int(device), cv2.CAP_V4L2)
        return cv2.VideoCapture(device, cv2.CAP_V4L2)

    def start(self) -> None:
        import cv2

        for name, device in self.device_dict.items():
            cap = self._open_capture(device)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open V4L2 camera '{name}' ({device})")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            self._caps[name] = cap
            print(f"[camera] opened V4L2 '{name}' -> {device}")
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        import cv2

        while not self._stop.is_set():
            for name, cap in self._caps.items():
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                if frame.ndim == 2:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                else:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with self._lock:
                    self._frames[name] = to_rgb_hwc_uint8(rgb, self.height, self.width)
            time.sleep(max(0.0, 1.0 / max(self.fps, 1) * 0.5))

    def wait_ready(self, timeout_s: float = 10.0) -> None:
        deadline = time.time() + timeout_s
        needed = set(self.device_dict.keys())
        while time.time() < deadline:
            with self._lock:
                ready = needed.issubset(set(self._frames.keys()))
            if ready:
                print("[camera] all V4L2 cameras streaming")
                return
            time.sleep(0.1)
        print("[camera] WARNING: not all V4L2 cameras produced frames before timeout")

    def get_images(self) -> dict[str, np.ndarray]:
        with self._lock:
            return {name: frame.copy() for name, frame in self._frames.items()}

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        for name, cap in self._caps.items():
            try:
                cap.release()
            except Exception as exc:
                print(f"[camera] release '{name}' error: {exc}")
        self._caps.clear()


class HttpCameraStream:
    """Poll JPEG/PNG image bytes from HTTP GET endpoints."""

    backend: CameraBackend = "http"

    def __init__(
        self,
        url_dict: dict[str, str],
        width: int,
        height: int,
        fps: int,
        *,
        timeout_s: float = 10.0,
    ):
        self.url_dict = url_dict
        self.width = width
        self.height = height
        self.fps = fps
        self.timeout_s = timeout_s
        self._frames: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    @staticmethod
    def _extract_jpeg_bytes(data: bytes) -> bytes | None:
        if not data:
            return None
        start = data.find(b"\xff\xd8")
        if start < 0:
            return None
        end = data.find(b"\xff\xd9", start)
        if end < 0:
            return None
        return data[start : end + 2]

    @staticmethod
    def _decode_image_bytes(data: bytes) -> np.ndarray | None:
        import cv2

        jpeg = HttpCameraStream._extract_jpeg_bytes(data)
        if jpeg is None:
            return None
        bgr = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _camera_loop(self, name: str, url: str) -> None:
        import urllib.error
        import urllib.request

        while not self._stop.is_set():
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "xrobotoolkit-camera/1.0"})
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    buf = b""
                    while not self._stop.is_set():
                        chunk = resp.read(8192)
                        if not chunk:
                            break
                        buf += chunk
                        while True:
                            jpeg = self._extract_jpeg_bytes(buf)
                            if jpeg is None:
                                break
                            end = buf.find(b"\xff\xd9", buf.find(b"\xff\xd8")) + 2
                            buf = buf[end:]
                            rgb = self._decode_image_bytes(jpeg)
                            if rgb is not None:
                                with self._lock:
                                    self._frames[name] = to_rgb_hwc_uint8(rgb, self.height, self.width)
                        if len(buf) > 2_000_000:
                            buf = buf[-200_000:]
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                if not self._stop.is_set():
                    print(f"[camera] HTTP stream '{name}' disconnected: {exc}")
                time.sleep(0.3)

    def start(self) -> None:
        for name, url in self.url_dict.items():
            print(f"[camera] opened HTTP '{name}' -> {url}")
            thread = threading.Thread(target=self._camera_loop, args=(name, url), daemon=True)
            thread.start()
            self._threads.append(thread)

    def wait_ready(self, timeout_s: float = 10.0) -> None:
        deadline = time.time() + timeout_s
        needed = set(self.url_dict.keys())
        while time.time() < deadline:
            with self._lock:
                ready = needed.issubset(set(self._frames.keys()))
            if ready:
                print("[camera] all HTTP cameras streaming")
                return
            time.sleep(0.1)
        print("[camera] WARNING: not all HTTP cameras produced frames before timeout")

    def get_images(self) -> dict[str, np.ndarray]:
        with self._lock:
            return {name: frame.copy() for name, frame in self._frames.items()}

    def stop(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=1.0)
        self._threads.clear()


def create_camera_stream(
    *,
    camera_urls: str | None = None,
    camera_devices: str | None = None,
    camera_serials: str | None = None,
    camera_serial_dict: dict[str, str] | None = None,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    log_prefix: str = "[camera]",
) -> tuple[RealSenseCameraStream | V4l2CameraStream | HttpCameraStream, list[str], CameraBackend]:
    url_dict = parse_camera_urls(camera_urls)
    if url_dict:
        if camera_serials or camera_devices or camera_serial_dict:
            print(f"{log_prefix} --camera-urls set; ignoring serials/devices")
        names = sorted(url_dict.keys())
        return HttpCameraStream(url_dict, width, height, fps), names, "http"

    device_dict = parse_camera_devices(camera_devices)
    if device_dict:
        if camera_serials or camera_serial_dict:
            print(f"{log_prefix} --camera-devices set; ignoring serials")
        names = sorted(device_dict.keys())
        return V4l2CameraStream(device_dict, width, height, fps), names, "v4l2"

    if camera_serial_dict is not None:
        serial_dict = dict(camera_serial_dict)
    else:
        serial_dict = parse_camera_serials(camera_serials)
    names = sorted(serial_dict.keys())
    return RealSenseCameraStream(serial_dict, width, height, fps), names, "realsense"


class FlexibleCameraInterface(BaseCameraInterface):
    """BaseCameraInterface adapter for RealSense / V4L2 / HTTP streams (no depth on V4L2/HTTP)."""

    def __init__(
        self,
        *,
        camera_serial_dict: dict[str, str] | None = None,
        camera_devices: str | None = None,
        camera_urls: str | None = None,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        enable_compression: bool = True,
        jpg_quality: int = 85,
        log_prefix: str = "[TB6R5]",
    ):
        super().__init__(enable_compression=enable_compression, jpg_quality=jpg_quality)
        self._stream, self._camera_names, self._backend = create_camera_stream(
            camera_urls=camera_urls,
            camera_devices=camera_devices,
            camera_serial_dict=camera_serial_dict,
            width=width,
            height=height,
            fps=fps,
            log_prefix=log_prefix,
        )

    @property
    def camera_names(self) -> list[str]:
        return list(self._camera_names)

    @property
    def backend(self) -> CameraBackend:
        return self._backend

    def start(self) -> None:
        self._stream.start()
        self._stream.wait_ready()

    def stop(self) -> None:
        self._stream.stop()

    def update_frames(self) -> None:
        # RealSense stream polls in its own thread; V4L2/HTTP likewise.
        pass

    def get_frames(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for name, rgb in self._stream.get_images().items():
            out[name] = {"color": rgb}
        return out

    def get_frame(self, identifier: str) -> dict | None:
        return self.get_frames().get(identifier)

    def get_compressed_frames(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for name, fd in self.get_frames().items():
            color = fd.get("color")
            entry: dict = {}
            if color is not None:
                entry["color"] = compress_image_to_jpg(color, self.jpg_quality)
            out[name] = entry
        return out
