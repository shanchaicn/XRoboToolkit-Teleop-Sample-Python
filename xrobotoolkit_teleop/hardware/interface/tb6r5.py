import os
import shutil
import sys
import platform
import random
import subprocess
import threading
import time
import ctypes
from contextlib import contextmanager
from typing import Callable, Literal, Optional, Tuple

import numpy as np

TeleopMode = Literal["placo_ik", "jog_any_c"]
TELEOP_MODES: tuple[TeleopMode, ...] = ("placo_ik", "jog_any_c")
DEFAULT_TELEOP_MODE: TeleopMode = "placo_ik"
DEFAULT_RPC_CMD_RATE_HZ = 50
# Shared by JogAnyC and JogAnyJ (--zone-ratio on CLI).
DEFAULT_ZONE_RATIO = 0.05
DEFAULT_JOG_ZONE_RATIO = DEFAULT_ZONE_RATIO  # back-compat alias
DEFAULT_JOG_ANY_C_ZONE_RATIO = DEFAULT_ZONE_RATIO
DEFAULT_JOG_ANY_J_ZONE_RATIO = DEFAULT_ZONE_RATIO
DEFAULT_JOG_ANY_J_LAST_COUNT = 500
DEFAULT_JOG_ANY_JOINT_VEL = 1.0
DEFAULT_JOG_ANY_JOINT_ACC = 1.0
DEFAULT_JOG_ANY_JOINT_DEC = 1.0
DEFAULT_JOG_ASYNC_TIMEOUT_MS = 5000000
# MoveTwoFingersGripper: distance=0 open, distance=max closed (controller units).
DEFAULT_GRIPPER_MAX_D = 12.0
DEFAULT_TWO_FINGERS_GRIPPER_INTERVAL = 25.0
# VR joystick axes are in [-1, 1]; |axis|==1 is full stick deflection.
JOYSTICK_AXIS_DEFLECTION_MAX = 1.0
# Back-compat alias used by CLI / controller imports
DEFAULT_JOG_ANY_C_ASYNC_TIMEOUT_MS = DEFAULT_JOG_ASYNC_TIMEOUT_MS


def _should_drop_jog_any_j_rpc_log(line: str) -> bool:
    return "JogAnyJ" in line and ("[async] msg:" in line or "[await] msg:" in line)


@contextmanager
def _filter_stdout_lines(should_drop: Callable[[str], bool]):
    """Drop matching stdout lines emitted by rpc.so during a synchronous call."""
    read_fd, write_fd = os.pipe()
    saved_stdout = os.dup(1)
    try:
        os.dup2(write_fd, 1)
        os.close(write_fd)
        yield
    finally:
        os.dup2(saved_stdout, 1)
        captured = b""
        while True:
            chunk = os.read(read_fd, 65536)
            if not chunk:
                break
            captured += chunk
        os.close(read_fd)
        if captured:
            text = captured.decode("utf-8", errors="replace")
            if not text.endswith("\n"):
                text += "\n"
            for line in text.splitlines(keepends=True):
                if should_drop(line):
                    continue
                os.write(saved_stdout, line.encode("utf-8", errors="replace"))
        os.close(saved_stdout)


def _platform_subdir() -> str:
    system = platform.system().lower()
    if system == "linux":
        machine = platform.machine().lower()
        return "arm" if machine in ("aarch64", "arm64") else "x86"
    if system == "windows":
        return "win"
    raise RuntimeError(f"Unsupported OS: {system}")


def _preload_library(path: str) -> bool:
    """Load a shared library globally; return False if the file is missing or wrong arch."""
    if not os.path.isfile(path):
        return False
    try:
        ctypes.CDLL(os.path.abspath(path), mode=ctypes.RTLD_GLOBAL)
        return True
    except OSError:
        return False


def _prepend_ld_library_path(*paths: str):
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    merged = ":".join(path for path in paths if path)
    if existing:
        merged = merged + ":" + existing
    os.environ["LD_LIBRARY_PATH"] = merged


def _topic_lib_dir() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(
        os.path.join(base_dir, "../../../dependencies/get_status_py/topic_all_py/lib", _platform_subdir())
    )


def _rpc_lib_dir() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(
        os.path.join(
            base_dir,
            "../../../dependencies/hello_demo_py/rpc_py_all/lib/linux",
            _platform_subdir(),
        )
    )


def _elf_machine(path: str) -> Optional[str]:
    """Return ELF machine string (e.g. 'x86-64', 'aarch64') or None."""
    try:
        out = subprocess.check_output(["file", "-b", path], text=True, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    if "x86-64" in out:
        return "x86-64"
    if "aarch64" in out or "ARM" in out:
        return "aarch64"
    return None


def _host_elf_machine() -> str:
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        return "x86-64"
    if machine in ("aarch64", "arm64"):
        return "aarch64"
    return machine


def _ensure_topic_protobuf_lib(target_dir: str):
    """Ensure libprotobuf.so.32 matches host arch (git LFS sometimes ships ARM under x86/)."""
    pb = os.path.join(target_dir, "libprotobuf.so")
    pb32 = os.path.join(target_dir, "libprotobuf.so.32")
    host = _host_elf_machine()

    if os.path.isfile(pb) and _elf_machine(pb) == host:
        if not os.path.isfile(pb32) or _elf_machine(pb32) != host:
            shutil.copy2(pb, pb32)
        return pb32

    if os.path.isfile(pb32) and _elf_machine(pb32) == host:
        return pb32

    raise RuntimeError(
        f"No compatible libprotobuf for {host} in {target_dir}. " "Use topic_all_py/lib/x86/libprotobuf.so on x86_64."
    )


def _setup_topic_import():
    topic_root = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../dependencies/get_status_py/topic_all_py")
    )
    target_dir = _topic_lib_dir()

    if target_dir not in sys.path:
        sys.path.insert(0, target_dir)

    pb32 = _ensure_topic_protobuf_lib(target_dir)
    _prepend_ld_library_path(target_dir, topic_root)
    os.environ["LD_PRELOAD"] = pb32
    _preload_library(pb32)


def _setup_rpc_import():
    rpc_root = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../dependencies/hello_demo_py/rpc_py_all")
    )
    target_dir = _rpc_lib_dir()

    if target_dir not in sys.path:
        sys.path.insert(0, target_dir)

    _prepend_ld_library_path(target_dir, rpc_root)


class TB6R5Interface:
    """Low-level interface for the TB6-R5 6-DOF robotic arm.

    Wraps the RPC (control) and Topic (state feedback) SDK modules.
    """

    TELEOP_VAR_NAME = "teleop"

    def __init__(
        self,
        ip: str = "192.168.2.98",
        rpc_port: int = 5868,
        enable_topic: bool = True,
        joint_count: int = 6,
        rpc_cmd_rate_hz: float = DEFAULT_RPC_CMD_RATE_HZ,
        jog_any_c_interrupt: bool = False,
        zone_ratio: float = DEFAULT_ZONE_RATIO,
        jog_any_c_async_timeout_ms: int = DEFAULT_JOG_ASYNC_TIMEOUT_MS,
        cartesian_vel: Optional[float] = None,
        cartesian_acc: Optional[float] = None,
        cartesian_dec: Optional[float] = None,
        joint_vel: float = DEFAULT_JOG_ANY_JOINT_VEL,
        joint_acc: float = DEFAULT_JOG_ANY_JOINT_ACC,
        joint_dec: float = DEFAULT_JOG_ANY_JOINT_DEC,
    ):
        self.ip = ip
        self.rpc_port = rpc_port
        self.enable_topic = enable_topic
        self.rpc_cmd_rate_hz = max(float(rpc_cmd_rate_hz), 1.0)
        self.jog_any_c_interrupt = bool(jog_any_c_interrupt)
        self.zone_ratio = max(float(zone_ratio), 0.0)
        self.jog_async_timeout_ms = max(int(jog_any_c_async_timeout_ms), 100)
        self.jog_any_c_async_timeout_ms = self.jog_async_timeout_ms
        self.cartesian_vel = None if cartesian_vel is None else max(float(cartesian_vel), 0.0)
        self.cartesian_acc = None if cartesian_acc is None else max(float(cartesian_acc), 0.0)
        self.cartesian_dec = None if cartesian_dec is None else max(float(cartesian_dec), 0.0)
        self.joint_vel = max(float(joint_vel), 0.0)
        self.joint_acc = max(float(joint_acc), 0.0)
        self.joint_dec = max(float(joint_dec), 0.0)
        self.client = None
        self._rpc = None
        self._topic = None
        self.joint_count = max(int(joint_count), 1)
        self._last_cmd_time = 0.0
        self._last_cmd_q: Optional[np.ndarray] = None
        self._min_cmd_interval_s = 1.0 / self.rpc_cmd_rate_hz
        self._joint_cmd_eps = 1e-4
        self._cached_q = np.zeros(self.joint_count)
        self._cached_dq = np.zeros(self.joint_count)
        self._cached_robottarget_xyz = np.zeros(3)
        self._cached_robottarget_quat = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z
        self._robottarget_healthy = False
        self._last_cmd_xyz: Optional[np.ndarray] = None
        self._last_cmd_quat: Optional[np.ndarray] = None
        self._pose_cmd_eps = 1e-4
        self._state_lock = threading.Lock()
        self._state_thread: Optional[threading.Thread] = None
        self._state_stop = threading.Event()
        self._topic_healthy = False
        self._topic_warned = False
        self._server_in_error = False
        self._last_rpc_error: Optional[str] = None
        self._cartesian_stream_count = 0
        self._joint_stream_count = 0
        self._jog_async_pending = 0
        self._max_jog_async_pending = 32
        self._rpc_sync_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self):
        if self.enable_topic:
            _setup_topic_import()
            import topic

            self._topic = topic
            topic.start_subscriber(self.ip)
            print("Topic subscriber started.")
            time.sleep(0.5)

        _setup_rpc_import()
        import rpc

        self._rpc = rpc
        print(f"Connecting to TB6-R5 at {self.ip}:{self.rpc_port} ...")
        self.client = rpc.CPPClient(self.ip, self.rpc_port)
        print("RPC connected.")

        if self.enable_topic:
            self._start_state_reader()
        self._send_init_commands()

    def disconnect(self):
        self._state_stop.set()
        if self._state_thread is not None:
            self._state_thread.join(timeout=1.0)
            self._state_thread = None
        if self.client is not None:
            self._send_rpc_sync("{Disable}", timeout_ms=3000, sleep_s=0.1)
            print("TB6-R5 disconnected.")

    def is_topic_healthy(self) -> bool:
        return self._topic_healthy

    def _start_state_reader(self):
        if self._state_thread is not None:
            return

        def _reader_loop():
            while not self._state_stop.is_set():
                q, dq, rt_xyz, rt_quat, rt_ok, ok = self._read_state_from_topic()
                with self._state_lock:
                    if ok:
                        self._cached_q = q
                        self._cached_dq = dq
                        self._topic_healthy = True
                    if rt_ok:
                        self._cached_robottarget_xyz = rt_xyz
                        self._cached_robottarget_quat = rt_quat
                        self._robottarget_healthy = True
                    elif not self._topic_warned:
                        self._topic_warned = True
                        print(
                            "[TB6R5] Topic feedback unavailable (protobuf unpack failed). "
                            "Placo will track commanded joints; update get_status_py if needed."
                        )
                time.sleep(1.0 / self.rpc_cmd_rate_hz)

        self._state_thread = threading.Thread(name="tb6r5_state_reader", target=_reader_loop, daemon=True)
        self._state_thread.start()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _send_init_commands(self):
        init_cmds = [
            "{Clear}",
            "{Disable}",
            "{Mode}",
            "{SetMaxToq}",
            "{Recover}",
            "{SetRate}",
            "{Enable}",
            "{Start}",
            "{Var --clear}",
            "{Recover}",
            "{SetUsingSP --state=on}",
            "{Var --type=jointtarget --name=teleop --value={0,0,0,0,0,0,0,0,0,0}}",
        ]
        for cmd in init_cmds:
            self._send_rpc_sync(cmd, timeout_ms=5000, sleep_s=0.1)

    # ------------------------------------------------------------------
    # RPC helpers
    # ------------------------------------------------------------------

    def _send_rpc_sync(
        self,
        cmd: str,
        timeout_ms: int = 5000,
        sleep_s: float = 0.0,
        ignore_subcmd_errors: bool = False,
    ):
        with self._rpc_sync_lock:
            msg = self._rpc.Msg(cmd)
            msg.setMsgID(10001)
            msg.setMsgSeqID(random.randint(1, 10000))
            status, resp_list = self.client.CallAwait(msg, timeout_ms)
        if status != 0:
            print(f"[TB6R5] RPC sync failed: {cmd} (status={status})")
            return False
        self._last_rpc_error = None
        for r in resp_list or []:
            if r.code < 0 and not ignore_subcmd_errors:
                self._last_rpc_error = r.message
                self._server_in_error = True
                print(f"[TB6R5] RPC error: {cmd} -> {r.message}")
                return False
        self._server_in_error = False
        if sleep_s > 0:
            time.sleep(sleep_s)
        return True

    def _delete_teleop_var(self):
        """Remove teleop jointtarget variable before re-creating it."""
        delete_cmd = "{DeleteVar --name=" + self.TELEOP_VAR_NAME + "}"
        self._send_rpc_sync(delete_cmd, timeout_ms=2000, ignore_subcmd_errors=True)

    @staticmethod
    def _resolve_stream_clear_buffer(stream_count: int, clear_buffer: Optional[int] = None) -> int:
        """First command in a grip segment: clear_buffer=0; all following: clear_buffer=1."""
        if clear_buffer is not None:
            return int(clear_buffer)
        return 0 if stream_count == 0 else 1

    def reset_cartesian_stream(self):
        """Reset JogAnyC stream state (call when grip is first pressed)."""
        self._cartesian_stream_count = 0
        with self._state_lock:
            self._jog_async_pending = 0

    def reset_joint_stream(self):
        """Reset JogAnyJ stream state (call when grip is first pressed in placo_ik)."""
        self._joint_stream_count = 0
        with self._state_lock:
            self._jog_async_pending = 0

    def _wait_jog_async_pending(self, timeout_s: Optional[float] = None):
        if timeout_s is None:
            timeout_s = self.jog_async_timeout_ms / 1000.0 + 0.5
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            with self._state_lock:
                if self._jog_async_pending == 0:
                    return
            time.sleep(0.005)

    def _wait_jog_any_c_async_pending(self, timeout_s: Optional[float] = None):
        self._wait_jog_async_pending(timeout_s)

    def has_fault(self) -> bool:
        with self._state_lock:
            return self._server_in_error

    def ensure_jog_ready(self) -> bool:
        """Clear fault and ensure controller is in Start state before JogAnyC."""
        self._wait_jog_async_pending()
        if self._server_in_error:
            self.clear_error()
        return self._send_rpc_sync("{Start}", timeout_ms=3000, ignore_subcmd_errors=True)

    def clear_error(self) -> bool:
        """Clear controller fault (required after 'server in error')."""
        self._wait_jog_async_pending()
        ok = self._send_rpc_sync("{Clear}", timeout_ms=3000, ignore_subcmd_errors=True)
        if ok:
            self._server_in_error = False
            self._cartesian_stream_count = 0
            self._joint_stream_count = 0
            with self._state_lock:
                self._jog_async_pending = 0
        return ok

    def _format_jog_any_j_cmd(
        self,
        q: np.ndarray,
        clear_buffer: int = 0,
        zone_ratio: Optional[float] = None,
        last_count: int = DEFAULT_JOG_ANY_J_LAST_COUNT,
    ) -> str:
        if zone_ratio is None:
            zone_ratio = self.zone_ratio
        val_str = self._format_jointtarget_value(q)
        cmd = (
            "{JogAnyJ --jointtarget_value="
            + val_str
            + f" --zone_ratio={float(zone_ratio):.4f} --clear_buffer={int(clear_buffer)} --last_count={int(last_count)}"
            + f" --joint_vel={self.joint_vel:.4f} --joint_acc={self.joint_acc:.4f} --joint_dec={self.joint_dec:.4f}"
        )
        return cmd + "}"

    def _send_jog_any_j(
        self,
        q: np.ndarray,
        move_timeout_ms: int = 5000,
        clear_buffer: int = 0,
        zone_ratio: Optional[float] = None,
        last_count: int = DEFAULT_JOG_ANY_J_LAST_COUNT,
    ) -> bool:
        """JogAnyJ with inline jointtarget (manual v1.7.5)."""
        cmd = self._format_jog_any_j_cmd(q, clear_buffer, zone_ratio, last_count)
        return self._send_rpc_sync(cmd, timeout_ms=move_timeout_ms)

    def _send_move_abs_j(self, q: np.ndarray, move_timeout_ms: int = 30000) -> bool:
        val_str = self._format_jointtarget_value(q)
        cmd = "{MoveAbsJ --jointtarget_value=" + val_str + "}"
        return self._send_rpc_sync(cmd, timeout_ms=move_timeout_ms)

    def _format_jog_any_c_cmd(
        self,
        xyz: np.ndarray,
        quat_wxyz: np.ndarray,
        clear_buffer: int,
        zone_ratio: Optional[float] = None,
    ) -> str:
        if zone_ratio is None:
            zone_ratio = self.zone_ratio
        val_str = self._format_robottarget_value(xyz, quat_wxyz)
        cmd = (
            "{JogAnyC --robottarget_value="
            + val_str
            + f" --zone_ratio={float(zone_ratio):.4f} --clear_buffer={int(clear_buffer)} --last_count=500"
        )
        if self.cartesian_vel is not None:
            cmd += f" --cartesian_vel={self.cartesian_vel:.4f}"
        if self.cartesian_acc is not None:
            cmd += f" --cartesian_acc={self.cartesian_acc:.4f}"
        if self.cartesian_dec is not None:
            cmd += f" --cartesian_dec={self.cartesian_dec:.4f}"
        return cmd + "}"

    def _make_jog_async_callback(self, label: str):
        def _on_response(status: int, resp_list):
            with self._state_lock:
                self._jog_async_pending = max(0, self._jog_async_pending - 1)
                if status < 0:
                    self._server_in_error = True
                    self._last_rpc_error = f"{label} async timeout (status={status})"
                    print(f"[TB6R5] {self._last_rpc_error}")
                    return
                for r in resp_list or []:
                    if r.code < 0:
                        self._server_in_error = True
                        self._last_rpc_error = r.message
                        print(f"[TB6R5] {label} async error: {r.message}")
                        return
                self._server_in_error = False
                self._last_rpc_error = None

        return _on_response

    def _send_jog_async(self, cmd: str, label: str, move_timeout_ms: Optional[int] = None) -> bool:
        if move_timeout_ms is None:
            move_timeout_ms = self.jog_async_timeout_ms
        msg = self._rpc.Msg(cmd)
        msg.setMsgID(10001)
        msg.setMsgSeqID(random.randint(1, 10000))
        ok = self.client.CallAsync(msg, move_timeout_ms, self._make_jog_async_callback(label))
        if ok:
            with self._state_lock:
                self._jog_async_pending += 1
        return bool(ok)

    def _send_jog_any_j_async(self, cmd: str, move_timeout_ms: Optional[int] = None) -> bool:
        with _filter_stdout_lines(_should_drop_jog_any_j_rpc_log):
            return self._send_jog_async(cmd, "JogAnyJ", move_timeout_ms)

    def _send_jog_any_c_async(self, cmd: str, move_timeout_ms: Optional[int] = None) -> bool:
        return self._send_jog_async(cmd, "JogAnyC", move_timeout_ms)

    def _send_jog_any_c(
        self,
        xyz: np.ndarray,
        quat_wxyz: np.ndarray,
        move_timeout_ms: int = 5000,
        clear_buffer: int = 0,
        zone_ratio: Optional[float] = None,
    ) -> bool:
        """JogAnyC with inline robottarget {x,y,z,qx,qy,qz,qw} (manual v1.7.5)."""
        cmd = self._format_jog_any_c_cmd(xyz, quat_wxyz, clear_buffer, zone_ratio)
        return self._send_rpc_sync(cmd, timeout_ms=move_timeout_ms)

    # ------------------------------------------------------------------
    # State feedback
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_robottarget_value(rt_value) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Parse topic robottarget {x,y,z,qx,qy,qz,qw} -> xyz, quat (w,x,y,z)."""
        try:
            if rt_value is None:
                return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), False
            if hasattr(rt_value, "__len__"):
                vals = [float(v) for v in rt_value]
            else:
                return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), False
            if len(vals) < 7:
                return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), False
            xyz = np.array(vals[:3], dtype=float)
            qx, qy, qz, qw = vals[3], vals[4], vals[5], vals[6]
            quat = np.array([qw, qx, qy, qz], dtype=float)
            return xyz, quat, True
        except (TypeError, ValueError):
            return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), False

    def _read_state_from_topic(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, bool]:
        if self._topic is None:
            return (
                self._cached_q.copy(),
                self._cached_dq.copy(),
                self._cached_robottarget_xyz.copy(),
                self._cached_robottarget_quat.copy(),
                False,
                False,
            )
        try:
            state = self._topic.get_system_state()
            if not state.has_rt():
                return (
                    self._cached_q.copy(),
                    self._cached_dq.copy(),
                    self._cached_robottarget_xyz.copy(),
                    self._cached_robottarget_quat.copy(),
                    False,
                    False,
                )

            rt = state.get_rt()
            if not rt.models:
                return (
                    self._cached_q.copy(),
                    self._cached_dq.copy(),
                    self._cached_robottarget_xyz.copy(),
                    self._cached_robottarget_quat.copy(),
                    False,
                    False,
                )

            model = rt.models[0]
            start = model.joint_start_idx

            q = np.zeros(self.joint_count)
            dq = np.zeros(self.joint_count)
            for j in range(start, min(start + model.joint_count, start + self.joint_count)):
                joint = rt.models_joints[j]
                q[j - start] = joint.position
                dq[j - start] = joint.velocity

            rt_xyz, rt_quat, rt_ok = self._cached_robottarget_xyz.copy(), self._cached_robottarget_quat.copy(), False
            if rt.models_current_points:
                cur = rt.models_current_points[0]
                rt_xyz, rt_quat, rt_ok = self._parse_robottarget_value(cur.robottarget)

            return q, dq, rt_xyz, rt_quat, rt_ok, True
        except Exception:
            return (
                self._cached_q.copy(),
                self._cached_dq.copy(),
                self._cached_robottarget_xyz.copy(),
                self._cached_robottarget_quat.copy(),
                False,
                False,
            )

    def get_joint_positions(self) -> np.ndarray:
        with self._state_lock:
            return self._cached_q.copy()

    def get_joint_velocities(self) -> np.ndarray:
        with self._state_lock:
            return self._cached_dq.copy()

    def get_last_joint_command(self) -> Optional[np.ndarray]:
        with self._state_lock:
            return None if self._last_cmd_q is None else self._last_cmd_q.copy()

    def get_robottarget(self) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Current TCP pose from Topic RT (xyz meters, quat wxyz)."""
        with self._state_lock:
            if not self._robottarget_healthy:
                return self._cached_robottarget_xyz.copy(), self._cached_robottarget_quat.copy(), False
            return self._cached_robottarget_xyz.copy(), self._cached_robottarget_quat.copy(), True

    def is_robottarget_healthy(self) -> bool:
        with self._state_lock:
            return self._robottarget_healthy

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def _format_robottarget_value(self, xyz: np.ndarray, quat_wxyz: np.ndarray) -> str:
        xyz = np.asarray(xyz, dtype=float).ravel()[:3]
        quat = np.asarray(quat_wxyz, dtype=float).ravel()
        if len(quat) < 4:
            quat = np.array([1.0, 0.0, 0.0, 0.0])
        qw, qx, qy, qz = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
        return "{" + ",".join(f"{v:.6f}" for v in (xyz[0], xyz[1], xyz[2], qx, qy, qz, qw)) + "}"

    def _format_jointtarget_value(self, q: np.ndarray) -> str:
        q = np.asarray(q, dtype=float).ravel()
        n = min(len(q), self.joint_count)
        values = [0.0] * 10
        for i in range(n):
            values[i] = float(q[i])
        return "{" + ",".join(f"{v:.6f}" for v in values) + "}"

    def set_joint_positions(self, q: np.ndarray, force: bool = False, clear_buffer: Optional[int] = None):
        """Send joint targets via JogAnyJ (async RPC, non-blocking)."""
        with self._state_lock:
            if self._server_in_error:
                return

        q = np.asarray(q, dtype=float).ravel()
        n = min(len(q), self.joint_count)
        q_cmd = q[:n].copy()

        now = time.time()
        if self._last_cmd_time > 0.0 and now - self._last_cmd_time < self._min_cmd_interval_s:
            return
        if (
            not force
            and self._last_cmd_q is not None
            and np.max(np.abs(q_cmd - self._last_cmd_q)) < self._joint_cmd_eps
        ):
            return

        clear_buffer = self._resolve_stream_clear_buffer(self._joint_stream_count, clear_buffer)
        cmd = self._format_jog_any_j_cmd(q_cmd, clear_buffer=clear_buffer)
        with self._state_lock:
            if self._jog_async_pending >= self._max_jog_async_pending:
                return
        if not self._send_jog_any_j_async(cmd):
            return

        self._last_cmd_time = now
        self._last_cmd_q = q_cmd
        self._joint_stream_count += 1

    def set_cartesian_target(
        self,
        xyz: np.ndarray,
        quat_wxyz: np.ndarray,
        clear_buffer: Optional[int] = None,
    ) -> bool:
        """Send TCP pose via JogAnyC (Topic robottarget frame)."""
        with self._state_lock:
            if self._server_in_error:
                return True

        xyz = np.asarray(xyz, dtype=float).ravel()[:3].copy()
        quat = np.asarray(quat_wxyz, dtype=float).ravel()[:4].copy()

        now = time.time()
        if self._last_cmd_time > 0.0 and now - self._last_cmd_time < self._min_cmd_interval_s:
            return True
        if self._last_cmd_xyz is not None and self._last_cmd_quat is not None:
            pos_same = np.linalg.norm(xyz - self._last_cmd_xyz) < self._pose_cmd_eps
            quat_same = np.linalg.norm(quat - self._last_cmd_quat) < self._pose_cmd_eps
            if pos_same and quat_same:
                return True

        # clear_buffer=0 on first cmd after reset_cartesian_stream(); then always 1.
        clear_buffer = self._resolve_stream_clear_buffer(self._cartesian_stream_count, clear_buffer)

        cmd = self._format_jog_any_c_cmd(xyz, quat, clear_buffer)
        with self._state_lock:
            if self._jog_async_pending >= self._max_jog_async_pending:
                return True
        ok = self._send_jog_any_c_async(cmd)
        if not ok:
            return True

        self._last_cmd_time = now
        self._last_cmd_xyz = xyz
        self._last_cmd_quat = quat
        self._cartesian_stream_count += 1
        return True

    @staticmethod
    def gripper_distance_from_joystick_axes(
        axis_x: float,
        axis_y: float,
        max_distance: float = DEFAULT_GRIPPER_MAX_D,
    ) -> float:
        """Map VR joystick axes to MoveTwoFingersGripper distance.

        Args:
            axis_x, axis_y: Joystick components in [-1, 1] (0 = centered).
            max_distance: Fully closed distance sent to RPC (default 12).

        Returns:
            distance in [0, max_distance]:
              0            -> fully open
              max_distance -> fully closed

        Mapping (Chebyshev / L-inf norm on |axis|):
            deflection = min(1, max(|axis_x|, |axis_y|))
            distance = max_distance * deflection

        So either axis alone can drive the full open->closed range; diagonal push
        does not require both axes at 1.0 (unlike the old axis_x^2+axis_y^2 map).
        """
        deflection = min(
            1.0,
            max(abs(float(axis_x)), abs(float(axis_y))) / JOYSTICK_AXIS_DEFLECTION_MAX,
        )
        deflection = max(0.0, deflection)
        return float(max_distance) * deflection

    def move_two_fingers_gripper(
        self,
        distance: float,
        interval: Optional[float] = None,
        max_distance: Optional[float] = None,
    ) -> bool:
        """Send MoveTwoFingersGripper (distance 0=open, larger=more closed)."""
        if max_distance is None:
            max_distance = DEFAULT_GRIPPER_MAX_D
        if interval is None:
            interval = DEFAULT_TWO_FINGERS_GRIPPER_INTERVAL
        distance = max(0.0, min(float(distance), float(max_distance)))
        interval = max(0.0, float(interval))
        cmd = f"{{MoveTwoFingersGripper --distance={distance:.4f} --interval={interval:.4f}}}"
        return self._send_rpc_sync(cmd, timeout_ms=5000)

    def stop(self):
        self._wait_jog_async_pending(timeout_s=0.5)
        self._send_rpc_sync("{Stop}", timeout_ms=3000)

    def go_home(self, q: Optional[np.ndarray] = None):
        """Move to home joint pose (rad) via MoveAbsJ. Defaults to all zeros if q is omitted."""
        if q is None:
            q = np.zeros(self.joint_count)
        self.clear_error()
        self._send_move_abs_j(np.asarray(q, dtype=float).ravel(), move_timeout_ms=30000)

    def enable(self):
        self._send_rpc_sync("{Enable}", timeout_ms=5000)

    def disable(self):
        self._send_rpc_sync("{Disable}", timeout_ms=5000)
