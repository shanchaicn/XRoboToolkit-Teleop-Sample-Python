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
DEFAULT_JOG_ANY_JOINT_VEL = 6.0
DEFAULT_JOG_ANY_JOINT_ACC = 3.0
DEFAULT_JOG_ANY_JOINT_DEC = 3.0
# SubLoop1 --immediate: true => abandon the currently-processing exec and run this one now.
DEFAULT_SUBLOOP1_IMMEDIATE = False
DEFAULT_JOG_ASYNC_TIMEOUT_MS = 5000000
# MoveTwoFingersGripper (YS gripper): distance in mm; teleop maps trigger to [min_d, max_d].
DEFAULT_GRIPPER_MAX_D = 70.0
DEFAULT_GRIPPER_MIN_D = 0.0
DEFAULT_TWO_FINGERS_GRIPPER_INTERVAL = 5.0
DEFAULT_GRIPPER_CMD_DELTA_MM = 0.5
DEFAULT_DUAL_MODEL_MOVE_TIMEOUT_MS = 120000
SUBLOOP1_CMD = "SubLoop1"
DEFAULT_SUBLOOP1_EXEC_TIMEOUT_MS = 5000
DEFAULT_SUBLOOP1_EXIT_TIMEOUT_MS = DEFAULT_DUAL_MODEL_MOVE_TIMEOUT_MS
NOT_RUN_EXECUTE = "NotRunExecute"
GRIPPER_YS_STATUS_FORMAT = "<d"  # TwoFingerGripperYSStatus: actual_pos (double, mm)
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


def validate_robot_sdk(*, require_topic: bool = True) -> None:
    """Fail fast when RPC/topic binaries are missing or built for the wrong arch/OS."""
    subdir = _platform_subdir()
    host = _host_elf_machine()
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"

    rpc_dir = _rpc_lib_dir()
    rpc_so = os.path.join(rpc_dir, "rpc.so")
    if not os.path.isfile(rpc_so):
        raise RuntimeError(
            f"Missing RPC library for linux/{subdir} ({host}): {rpc_so}\n"
            "Copy the vendor rpc_py_all/lib/linux/<arm|x86>/ bundle (rpc.so + deps) "
            "from the robot SDK package. tb5r6 and tb6r5 use the same binaries."
        )
    rpc_elf = _elf_machine(rpc_so)
    if rpc_elf and rpc_elf != host:
        raise RuntimeError(
            f"rpc.so architecture mismatch: file is {rpc_elf}, host is {host} ({rpc_so}). "
            f"Use lib/linux/{subdir}/ built for this machine."
        )

    if subdir == "arm" and py_ver != "3.10":
        print(
            f"[TB6R5] WARNING: ARM RPC/topic .so files are built for Python 3.10; "
            f"current interpreter is {py_ver}. Prefer `python3.10` on ARM if `import rpc` fails."
        )

    if not require_topic:
        return

    topic_dir = _topic_lib_dir()
    topic_so = os.path.join(topic_dir, "topic.so")
    if not os.path.isfile(topic_so):
        raise RuntimeError(
            f"Missing topic library for {subdir} ({host}): {topic_so}\n"
            "Copy topic_all_py/lib/<arm|x86>/ (topic.so, libprotobuf.so.32, libzmq.so*) "
            "from the robot SDK package."
        )
    topic_elf = _elf_machine(topic_so)
    if topic_elf and topic_elf != host:
        raise RuntimeError(
            f"topic.so architecture mismatch: file is {topic_elf}, host is {host} ({topic_so})."
        )
    _ensure_topic_protobuf_lib(topic_dir)


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
        subloop1_immediate: bool = DEFAULT_SUBLOOP1_IMMEDIATE,
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
        self.subloop1_immediate = bool(subloop1_immediate)
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
        self._subloop1_stream_pending = 0
        self._max_jog_async_pending = 32
        self._rpc_sync_lock = threading.Lock()
        self._cached_gripper_mm: Optional[float] = None
        self._gripper_feedback_healthy = False
        self._last_gripper_distance_sent: Optional[float] = None
        self._last_gripper_cmd_time = 0.0
        self._gripper_cmd_delta_mm = DEFAULT_GRIPPER_CMD_DELTA_MM
        self._subloop1_active = False
        self._subloop1_exiting = False
        self._rpc_ready = False
        self._topic_all_py_root = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../dependencies/get_status_py/topic_all_py")
        )

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        """True only after RPC init succeeded (and topic is healthy when enabled)."""
        if self.client is None or not self._rpc_ready:
            return False
        if self.enable_topic and not self._topic_healthy:
            return False
        return True

    def wait_for_topic_healthy(self, timeout_s: float = 5.0) -> bool:
        deadline = time.monotonic() + max(float(timeout_s), 0.0)
        while time.monotonic() < deadline:
            if self.is_topic_healthy():
                return True
            time.sleep(0.05)
        return False

    def connect(self, topic_wait_timeout_s: float = 5.0):
        self._rpc_ready = False
        self._state_stop = threading.Event()
        self._topic_healthy = False
        try:
            validate_robot_sdk(require_topic=self.enable_topic)
            if self.enable_topic:
                self._connect_topic_subscriber()

            _setup_rpc_import()
            import rpc

            self._rpc = rpc
            print(f"Connecting to TB6-R5 at {self.ip}:{self.rpc_port} ...")
            self.client = rpc.CPPClient(self.ip, self.rpc_port)

            if self.enable_topic:
                self._start_state_reader()

            if not self._send_init_commands():
                raise ConnectionError(
                    f"TB6-R5 RPC init failed at {self.ip}:{self.rpc_port} "
                    "(robot unreachable or rejected commands)."
                )

            if self.enable_topic and not self.wait_for_topic_healthy(topic_wait_timeout_s):
                raise ConnectionError(
                    f"TB6-R5 topic feedback not available from {self.ip} within {topic_wait_timeout_s:.1f}s."
                )

            self._rpc_ready = True
            print(f"TB6-R5 connected and verified at {self.ip}:{self.rpc_port}.")
        except Exception:
            self.disconnect()
            raise

    def _connect_topic_subscriber(self):
        _setup_topic_import()
        import topic

        self._topic = topic
        topic.start_subscriber(self.ip)
        print("Topic subscriber started.")
        time.sleep(0.5)

    def connect_topic_feedback(self, topic_wait_timeout_s: float = 5.0):
        """Start topic feedback only (no RPC). Used by LeRobot PICO teleop."""
        if not self.enable_topic:
            raise RuntimeError("connect_topic_feedback requires enable_topic=True")
        validate_robot_sdk(require_topic=True)
        self._state_stop = threading.Event()
        self._topic_healthy = False
        try:
            self._connect_topic_subscriber()
            self._start_state_reader()
            if not self.wait_for_topic_healthy(topic_wait_timeout_s):
                raise ConnectionError(
                    f"TB6-R5 topic feedback not available from {self.ip} within {topic_wait_timeout_s:.1f}s."
                )
            print(f"TB6-R5 topic feedback verified at {self.ip}.")
        except Exception:
            self.disconnect_topic_feedback()
            raise

    def disconnect_topic_feedback(self):
        """Stop topic feedback thread without touching RPC."""
        self._state_stop.set()
        if self._state_thread is not None:
            self._state_thread.join(timeout=1.0)
            self._state_thread = None
        self._topic_healthy = False
        print("TB6-R5 topic feedback disconnected.")

    def disconnect(self):
        self._rpc_ready = False
        self._state_stop.set()
        if self._state_thread is not None:
            self._state_thread.join(timeout=1.0)
            self._state_thread = None
        if self.client is not None:
            try:
                self.exit_subloop1_if_active(timeout_ms=3000, blocking_exit=True)
                self._send_rpc_sync("{Disable}", timeout_ms=3000, sleep_s=0.1)
            except Exception:
                pass
            self.client = None
        self._topic_healthy = False
        print("TB6-R5 disconnected.")

    def is_topic_healthy(self) -> bool:
        return self._topic_healthy

    def _start_state_reader(self):
        if self._state_thread is not None:
            return

        def _reader_loop():
            while not self._state_stop.is_set():
                q, dq, rt_xyz, rt_quat, rt_ok, ok = self._read_state_from_topic()
                gripper_mm, gripper_ok = self._read_gripper_mm_from_topic()
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
                    if gripper_ok:
                        self._cached_gripper_mm = gripper_mm
                        self._gripper_feedback_healthy = True
                time.sleep(1.0 / self.rpc_cmd_rate_hz)

        self._state_thread = threading.Thread(name="tb6r5_state_reader", target=_reader_loop, daemon=True)
        self._state_thread.start()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _send_init_commands(self) -> bool:
        init_cmds = [
            "{Clear}",
            "{Disable}",
            "{Mode}",
            "{SetMaxToq}",
            "{Recover}",
            "{SetRate}",
            "{Var --clear}",
            "{Recover}",
            "{SetUsingSP --state=on}",
            "{Var --type=jointtarget --name=teleop --value={0,0,0,0,0,0,0,0,0,0}}",
        ]
        for cmd in init_cmds:
            if not self._send_rpc_sync(cmd, timeout_ms=5000, sleep_s=0.1):
                return False
        # Arm/gripper dual-model: left=arm, right=gripper (NotRunExecute when gripper idle).
        if not self.send_dual_model("Enable", NOT_RUN_EXECUTE, timeout_ms=5000, sleep_s=0.1):
            return False
        if not self.send_dual_model("Start", NOT_RUN_EXECUTE, timeout_ms=5000, sleep_s=0.1, ignore_subcmd_errors=True):
            return False
        return True

    def _ensure_command_channel(self) -> bool:
        if not self.is_connected:
            return False
        with self._state_lock:
            if self._server_in_error:
                return False
        return True

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

    def _wait_motion_settled(
        self,
        timeout_s: float,
        target_q: Optional[np.ndarray] = None,
        vel_eps: float = 0.02,
        pos_tol: float = 0.02,
        settle_count: int = 3,
    ) -> bool:
        """Block until the arm stops (and optionally reaches target_q) using topic feedback.

        Manual note: inside SubLoop the first exec returns only AFTER exit, so the async
        callback cannot be used to detect motion completion. We must rely on joint state.
        Returns True if motion settled, False on timeout.
        """
        if self._topic is None or not self._topic_healthy:
            time.sleep(min(max(timeout_s, 0.0), 0.5))
            return False
        deadline = time.time() + max(timeout_s, 0.0)
        stable = 0
        while time.time() < deadline:
            dq = self.get_joint_velocities()
            moving = bool(np.any(np.abs(dq) > vel_eps))
            reached = True
            if target_q is not None:
                q = self.get_joint_positions()
                reached = bool(np.all(np.abs(q - target_q) < pos_tol))
            if (not moving) and reached:
                stable += 1
                if stable >= settle_count:
                    return True
            else:
                stable = 0
            time.sleep(0.02)
        return False

    def has_fault(self) -> bool:
        with self._state_lock:
            return self._server_in_error

    def ensure_jog_ready(self) -> bool:
        """Clear fault and ensure controller is in Start state before JogAnyC."""
        self._wait_jog_async_pending()
        if self._server_in_error:
            self.clear_error()
        return self.send_dual_model(
            "Start",
            NOT_RUN_EXECUTE,
            timeout_ms=3000,
            ignore_subcmd_errors=True,
        )

    def clear_error(self) -> bool:
        """Clear controller fault (required after 'server in error')."""
        if not self._subloop1_active:
            self._wait_jog_async_pending()
        self.exit_subloop1_if_active(timeout_ms=3000, blocking_exit=True)
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

    def _format_move_abs_j_inner(self, q: np.ndarray) -> str:
        val_str = self._format_jointtarget_value(q)
        return f"MoveAbsJ --jointtarget_value={val_str}"

    def _send_move_abs_j(self, q: np.ndarray, move_timeout_ms: int = 30000) -> bool:
        cmd = "{" + self._format_move_abs_j_inner(q) + "}"
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

    def _read_gripper_mm_from_topic(self) -> Tuple[Optional[float], bool]:
        """Read YS gripper actual_pos (mm) from NRT subsystem data."""
        if self._topic is None:
            return None, False
        try:
            if self._topic_all_py_root not in sys.path:
                sys.path.insert(0, self._topic_all_py_root)
            from system_state_reader import (
                get_subsystem_count,
                get_subsystem_data_size,
                get_subsystem_name,
                has_nrt_data,
                parse_subsystem_data,
            )

            if not has_nrt_data():
                return None, False
            for idx in range(get_subsystem_count()):
                name = get_subsystem_name(idx)
                if "Gripper" not in name and "gripper" not in name:
                    continue
                if get_subsystem_data_size(idx) < 8:
                    continue
                actual_pos = float(parse_subsystem_data(idx, GRIPPER_YS_STATUS_FORMAT)[0])
                return actual_pos, True
            return None, False
        except Exception:
            return None, False

    def get_gripper_distance_mm(self) -> Optional[float]:
        with self._state_lock:
            return None if self._cached_gripper_mm is None else float(self._cached_gripper_mm)

    def is_gripper_feedback_healthy(self) -> bool:
        with self._state_lock:
            return self._gripper_feedback_healthy

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
        if not self._ensure_command_channel():
            return

        q = np.asarray(q, dtype=float).ravel()
        n = min(len(q), self.joint_count)
        q_cmd = q[:n].copy()

        clear_buffer = self._resolve_stream_clear_buffer(self._joint_stream_count, clear_buffer)
        cmd = self._format_jog_any_j_cmd(q_cmd, clear_buffer=clear_buffer)
        with self._state_lock:
            if self._jog_async_pending >= self._max_jog_async_pending:
                return
        if not self._send_jog_any_j_async(cmd):
            return

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

        # clear_buffer=0 on first cmd after reset_cartesian_stream(); then always 1.
        clear_buffer = self._resolve_stream_clear_buffer(self._cartesian_stream_count, clear_buffer)

        cmd = self._format_jog_any_c_cmd(xyz, quat, clear_buffer)
        with self._state_lock:
            if self._jog_async_pending >= self._max_jog_async_pending:
                return True
        ok = self._send_jog_any_c_async(cmd)
        if not ok:
            return True

        self._last_cmd_xyz = xyz
        self._last_cmd_quat = quat
        self._cartesian_stream_count += 1
        return True

    @staticmethod
    def gripper_distance_from_trigger(
        trigger_value: float,
        max_distance: float = DEFAULT_GRIPPER_MAX_D,
        min_distance: float = DEFAULT_GRIPPER_MIN_D,
    ) -> float:
        """Map VR trigger [0,1] linearly to gripper distance in mm (trigger=0 open/max_d, trigger=1 closed/min_d)."""
        trigger = max(0.0, min(1.0, float(trigger_value)))
        lo = float(min_distance)
        hi = float(max_distance)
        return hi - trigger * (hi - lo)

    @staticmethod
    def gripper_distance_from_joystick_axes(
        axis_x: float,
        axis_y: float,
        max_distance: float = DEFAULT_GRIPPER_MAX_D,
        min_distance: float = DEFAULT_GRIPPER_MIN_D,
    ) -> float:
        """Legacy joystick mapping (deprecated; use gripper_distance_from_trigger)."""
        deflection = min(
            1.0,
            max(abs(float(axis_x)), abs(float(axis_y))) / JOYSTICK_AXIS_DEFLECTION_MAX,
        )
        deflection = max(0.0, deflection)
        lo = float(min_distance)
        hi = float(max_distance)
        return hi - deflection * (hi - lo)

    @staticmethod
    def _strip_cmd_braces(cmd: str) -> str:
        cmd = cmd.strip()
        if cmd.startswith("{") and cmd.endswith("}"):
            return cmd[1:-1]
        return cmd

    def _clamp_gripper_distance(
        self,
        distance: float,
        max_distance: Optional[float] = None,
        min_distance: Optional[float] = None,
    ) -> float:
        if max_distance is None:
            max_distance = DEFAULT_GRIPPER_MAX_D
        if min_distance is None:
            min_distance = DEFAULT_GRIPPER_MIN_D
        lo = float(min_distance)
        hi = float(max_distance)
        if lo > hi:
            lo, hi = hi, lo
        return max(lo, min(float(distance), hi))

    def _format_gripper_inner(
        self,
        distance: float,
        interval: float,
        max_distance: Optional[float] = None,
        min_distance: Optional[float] = None,
    ) -> str:
        distance = self._clamp_gripper_distance(distance, max_distance, min_distance)
        interval = max(0.0, float(interval))
        return f"MoveTwoFingersGripper --distance={distance:.4f} --interval={interval:.4f}"

    def format_dual_model_cmd(self, arm_inner: str, grip_inner: str) -> str:
        """Direct combined multi-model RPC: {arm||grip}. Blocks until both models finish."""
        arm_inner = (arm_inner or NOT_RUN_EXECUTE).strip()
        grip_inner = (grip_inner or NOT_RUN_EXECUTE).strip()
        return f"{{{arm_inner}||{grip_inner}}}"

    def send_dual_model(
        self,
        arm_inner: str,
        grip_inner: str,
        timeout_ms: int = 5000,
        sleep_s: float = 0.0,
        ignore_subcmd_errors: bool = False,
    ) -> bool:
        cmd = self.format_dual_model_cmd(arm_inner, grip_inner)
        return self._send_rpc_sync(
            cmd,
            timeout_ms=timeout_ms,
            sleep_s=sleep_s,
            ignore_subcmd_errors=ignore_subcmd_errors,
        )

    def format_subloop1_exec_cmd(
        self,
        arm_inner: str,
        grip_inner: str,
        immediate: bool = False,
    ) -> str:
        arm_inner = (arm_inner or NOT_RUN_EXECUTE).strip()
        grip_inner = (grip_inner or NOT_RUN_EXECUTE).strip()
        immediate_suffix = " --immediate=true" if immediate else ""
        return (
            f"{{{SUBLOOP1_CMD} --exec={{{arm_inner}}}{immediate_suffix}"
            f"||{SUBLOOP1_CMD} --exec={{{grip_inner}}}{immediate_suffix}}}"
        )

    def format_subloop1_cmd(
        self,
        arm_inner: str,
        gripper_distance: float,
        interval: Optional[float] = None,
        max_distance: Optional[float] = None,
        min_distance: Optional[float] = None,
    ) -> str:
        if interval is None:
            interval = DEFAULT_TWO_FINGERS_GRIPPER_INTERVAL
        grip_inner = self._format_gripper_inner(gripper_distance, interval, max_distance, min_distance)
        return self.format_subloop1_exec_cmd(arm_inner, grip_inner)

    def send_subloop1_blocking(
        self,
        arm_inner: str,
        grip_inner: str,
        timeout_ms: int = DEFAULT_SUBLOOP1_EXIT_TIMEOUT_MS,
        immediate: bool = False,
        settle_target_q: Optional[np.ndarray] = None,
        settle_timeout_s: float = 15.0,
    ) -> bool:
        """One-shot SubLoop1 session: first exec (async) -> wait motion done -> exit (sync).

        Per manual, the first exec is async and its result is only returned at exit. So we
        cannot use the async callback to detect when MoveAbsJ finished; we poll joint state
        (settle_target_q) and only then send exit, otherwise exit would abort the motion.
        """
        self.exit_subloop1_if_active(timeout_ms=timeout_ms, blocking_exit=True)
        cmd = self.format_subloop1_exec_cmd(arm_inner, grip_inner, immediate=immediate)
        print(f"[TB6R5] SubLoop1 blocking exec: {cmd}")
        if not self._send_subloop1_first_async(cmd):
            return False
        settled = self._wait_motion_settled(settle_timeout_s, target_q=settle_target_q)
        if not settled:
            print(
                f"[TB6R5] SubLoop1 blocking: motion did not settle within {settle_timeout_s:.1f}s; "
                "sending exit anyway."
            )
        if self.has_fault():
            print(f"[TB6R5] SubLoop1 blocking exec error: {self._last_rpc_error}")
        exit_cmd = self.format_subloop1_exit_cmd()
        print(f"[TB6R5] SubLoop1 blocking exit: {exit_cmd}")
        ok = self.send_subloop1_exit(timeout_ms=timeout_ms, blocking=True)
        if ok:
            print("[TB6R5] SubLoop1 blocking session complete.")
        return ok

    def format_subloop1_exit_cmd(self) -> str:
        """Dual-model SubLoop1 exit; flushes queued exec and returns the first exec result."""
        return f"{{{SUBLOOP1_CMD} --exec={{exit}}||{SUBLOOP1_CMD} --exec={{exit}}}}"

    def _send_subloop1_first_async(self, cmd: str) -> bool:
        """First SubLoop1 exec in a session: async with a long timeout (manual requirement)."""

        def _on_response(status: int, resp_list):
            with self._state_lock:
                self._jog_async_pending = max(0, self._jog_async_pending - 1)
            if status < 0:
                self._server_in_error = True
                self._last_rpc_error = f"SubLoop1 first exec async timeout (status={status})"
                print(f"[TB6R5] {self._last_rpc_error}")
                return
            for r in resp_list or []:
                if r.code < 0:
                    self._server_in_error = True
                    self._last_rpc_error = r.message
                    print(f"[TB6R5] SubLoop1 first exec error: {r.message}")
                    return
            self._server_in_error = False
            self._last_rpc_error = None

        msg = self._rpc.Msg(cmd)
        msg.setMsgID(10001)
        msg.setMsgSeqID(random.randint(1, 10000))
        ok = self.client.CallAsync(msg, self.jog_async_timeout_ms, _on_response)
        if ok:
            with self._state_lock:
                self._jog_async_pending += 1
            self._subloop1_active = True
        return bool(ok)

    def _send_subloop1_stream_async(self, cmd: str) -> bool:
        """Fire subsequent SubLoop1 exec without blocking the control loop (manual allows async)."""

        def _on_response(status: int, resp_list):
            with self._state_lock:
                self._subloop1_stream_pending = max(0, self._subloop1_stream_pending - 1)
            if status < 0:
                print(f"[TB6R5] SubLoop1 stream async failed (status={status}): {cmd[:120]}...")

        msg = self._rpc.Msg(cmd)
        msg.setMsgID(10001)
        msg.setMsgSeqID(random.randint(1, 10000))
        ok = self.client.CallAsync(msg, DEFAULT_SUBLOOP1_EXEC_TIMEOUT_MS, _on_response)
        if ok:
            with self._state_lock:
                self._subloop1_stream_pending += 1
        return bool(ok)

    def _finalize_subloop1_session(self) -> None:
        self._subloop1_active = False
        self._subloop1_exiting = False
        with self._state_lock:
            self._jog_async_pending = 0
            self._subloop1_stream_pending = 0

    def _send_subloop1_exit_async(self, cmd: str, timeout_ms: int) -> bool:
        if not self._subloop1_active:
            return True

        def _on_response(status: int, resp_list):
            with self._state_lock:
                self._jog_async_pending = max(0, self._jog_async_pending - 1)
            self._finalize_subloop1_session()
            if status < 0:
                print(f"[TB6R5] SubLoop1 exit async failed (status={status})")
                return
            for r in resp_list or []:
                if r.code < 0:
                    print(f"[TB6R5] SubLoop1 exit error: {r.message}")
                    return

        self._subloop1_exiting = True
        self._subloop1_active = False
        with self._state_lock:
            self._subloop1_stream_pending = 0

        msg = self._rpc.Msg(cmd)
        msg.setMsgID(10001)
        msg.setMsgSeqID(random.randint(1, 10000))
        ok = self.client.CallAsync(msg, timeout_ms, _on_response)
        if ok:
            with self._state_lock:
                self._jog_async_pending += 1
        else:
            self._subloop1_exiting = False
        return bool(ok)

    def send_subloop1_exit(
        self,
        timeout_ms: int = DEFAULT_SUBLOOP1_EXIT_TIMEOUT_MS,
        blocking: bool = False,
    ) -> bool:
        if not self._subloop1_active and not self._subloop1_exiting:
            return True
        cmd = self.format_subloop1_exit_cmd()
        if blocking:
            if not self._subloop1_active:
                return True
            ok = self._send_rpc_sync(cmd, timeout_ms=timeout_ms)
            self._finalize_subloop1_session()
            return ok
        return self._send_subloop1_exit_async(cmd, timeout_ms=timeout_ms)

    def exit_subloop1_if_active(
        self,
        timeout_ms: Optional[int] = None,
        settle_timeout_s: float = 2.0,
        blocking_exit: bool = False,
    ) -> bool:
        if not self._subloop1_active and not self._subloop1_exiting:
            return True
        if timeout_ms is None:
            timeout_ms = DEFAULT_SUBLOOP1_EXIT_TIMEOUT_MS
        if blocking_exit and self._subloop1_active:
            # Blocking path (homing/shutdown): wait for motion to settle before sync exit.
            self._wait_motion_settled(settle_timeout_s)
        return self.send_subloop1_exit(timeout_ms=timeout_ms, blocking=blocking_exit)

    def send_subloop1(
        self,
        arm_inner: str,
        gripper_distance: Optional[float],
        interval: Optional[float] = None,
        max_distance: Optional[float] = None,
        min_distance: Optional[float] = None,
        timeout_ms: int = DEFAULT_SUBLOOP1_EXEC_TIMEOUT_MS,
        immediate: Optional[bool] = None,
    ) -> bool:
        """Send a SubLoop1 multi-model exec. gripper_distance=None -> gripper sub-model gets
        NotRunExecute (do NOT re-issue MoveTwoFingersGripper every cycle, which floods the
        gripper queue and stalls both models)."""
        if not self._ensure_command_channel():
            return False
        if immediate is None:
            immediate = self.subloop1_immediate
        if interval is None:
            interval = DEFAULT_TWO_FINGERS_GRIPPER_INTERVAL
        if gripper_distance is None:
            grip_inner = NOT_RUN_EXECUTE
        else:
            grip_inner = self._format_gripper_inner(gripper_distance, interval, max_distance, min_distance)
        cmd = self.format_subloop1_exec_cmd(arm_inner, grip_inner, immediate=immediate)
        if self._subloop1_exiting:
            return False
        if not self._subloop1_active:
            ok = self._send_subloop1_first_async(cmd)
        else:
            ok = self._send_subloop1_stream_async(cmd)
        if ok and gripper_distance is not None:
            self._last_gripper_distance_sent = self._clamp_gripper_distance(
                gripper_distance, max_distance, min_distance
            )
            self._last_gripper_cmd_time = time.time()
        return ok

    def _should_send_gripper(
        self,
        gripper_distance: float,
        force: bool = False,
        cmd_delta: Optional[float] = None,
        max_distance: Optional[float] = None,
        min_distance: Optional[float] = None,
    ) -> bool:
        if force:
            return True
        if cmd_delta is None:
            cmd_delta = self._gripper_cmd_delta_mm
        gripper_distance = self._clamp_gripper_distance(gripper_distance, max_distance, min_distance)
        if self._last_gripper_distance_sent is None:
            return True
        return abs(gripper_distance - self._last_gripper_distance_sent) >= cmd_delta

    def send_gripper_only(
        self,
        gripper_distance: float,
        interval: Optional[float] = None,
        max_distance: Optional[float] = None,
        min_distance: Optional[float] = None,
        force: bool = False,
        cmd_delta: Optional[float] = None,
    ) -> bool:
        if not self._should_send_gripper(
            gripper_distance,
            force=force,
            cmd_delta=cmd_delta,
            max_distance=max_distance,
            min_distance=min_distance,
        ):
            return False
        return self.send_subloop1(
            NOT_RUN_EXECUTE,
            gripper_distance,
            interval=interval,
            max_distance=max_distance,
            min_distance=min_distance,
        )

    def set_joint_positions_with_gripper(
        self,
        q: np.ndarray,
        gripper_distance: float,
        force: bool = False,
        clear_buffer: Optional[int] = None,
        interval: Optional[float] = None,
        max_distance: Optional[float] = None,
        min_distance: Optional[float] = None,
        cmd_delta: Optional[float] = None,
    ) -> bool:
        """Send JogAnyJ + MoveTwoFingersGripper atomically via SubLoop1 (stream async RPC)."""
        if not self._ensure_command_channel():
            return False

        q = np.asarray(q, dtype=float).ravel()
        n = min(len(q), self.joint_count)
        q_cmd = q[:n].copy()
        gripper_distance = self._clamp_gripper_distance(gripper_distance, max_distance, min_distance)

        gripper_changed = self._should_send_gripper(
            gripper_distance,
            force=force,
            cmd_delta=cmd_delta,
            max_distance=max_distance,
            min_distance=min_distance,
        )

        clear_buffer = self._resolve_stream_clear_buffer(self._joint_stream_count, clear_buffer)
        arm_inner = self._strip_cmd_braces(self._format_jog_any_j_cmd(q_cmd, clear_buffer=clear_buffer))
        grip_arg = gripper_distance if gripper_changed else None
        ok = self.send_subloop1(
            arm_inner,
            grip_arg,
            interval=interval,
            max_distance=max_distance,
            min_distance=min_distance,
        )
        if not ok:
            return False

        self._last_cmd_q = q_cmd
        self._joint_stream_count += 1
        return True

    def set_cartesian_target_with_gripper(
        self,
        xyz: np.ndarray,
        quat_wxyz: np.ndarray,
        gripper_distance: float,
        clear_buffer: Optional[int] = None,
        interval: Optional[float] = None,
        max_distance: Optional[float] = None,
        min_distance: Optional[float] = None,
        cmd_delta: Optional[float] = None,
        force: bool = False,
    ) -> bool:
        """Send JogAnyC + MoveTwoFingersGripper atomically via SubLoop1 (stream async RPC)."""
        with self._state_lock:
            if self._server_in_error:
                return True

        xyz = np.asarray(xyz, dtype=float).ravel()[:3].copy()
        quat = np.asarray(quat_wxyz, dtype=float).ravel()[:4].copy()
        gripper_distance = self._clamp_gripper_distance(gripper_distance, max_distance, min_distance)

        gripper_changed = self._should_send_gripper(
            gripper_distance,
            force=force,
            cmd_delta=cmd_delta,
            max_distance=max_distance,
            min_distance=min_distance,
        )

        clear_buffer = self._resolve_stream_clear_buffer(self._cartesian_stream_count, clear_buffer)
        arm_inner = self._strip_cmd_braces(self._format_jog_any_c_cmd(xyz, quat, clear_buffer))
        grip_arg = gripper_distance if gripper_changed else None
        ok = self.send_subloop1(
            arm_inner,
            grip_arg,
            interval=interval,
            max_distance=max_distance,
            min_distance=min_distance,
        )
        if not ok:
            return True

        self._last_cmd_xyz = xyz
        self._last_cmd_quat = quat
        self._cartesian_stream_count += 1
        return True

    def move_two_fingers_gripper(
        self,
        distance: float,
        interval: Optional[float] = None,
        max_distance: Optional[float] = None,
        min_distance: Optional[float] = None,
    ) -> bool:
        """Send MoveTwoFingersGripper only (distance clamped to [min_d, max_d], mm)."""
        return self.send_gripper_only(
            distance,
            interval=interval,
            max_distance=max_distance,
            min_distance=min_distance,
        )

    def stop(self):
        if not self._subloop1_active:
            self._wait_jog_async_pending(timeout_s=0.5)
        self.exit_subloop1_if_active(timeout_ms=3000, blocking_exit=True)
        self._send_rpc_sync("{Stop}", timeout_ms=3000)

    def go_home(
        self,
        q: Optional[np.ndarray] = None,
        *,
        gripper_distance: Optional[float] = None,
        interval: Optional[float] = None,
        max_distance: Optional[float] = None,
        min_distance: Optional[float] = None,
        move_timeout_ms: int = DEFAULT_DUAL_MODEL_MOVE_TIMEOUT_MS,
        settle_timeout_s: float = 15.0,
    ) -> bool:
        """Home via SubLoop1: MoveAbsJ + gripper exec (async), then exit (sync wait)."""
        if q is None:
            q = np.zeros(self.joint_count)
        if self.has_fault():
            self.clear_error()
        q = np.asarray(q, dtype=float).ravel()
        arm_inner = self._format_move_abs_j_inner(q)
        if interval is None:
            interval = DEFAULT_TWO_FINGERS_GRIPPER_INTERVAL
        if gripper_distance is None:
            grip_inner = NOT_RUN_EXECUTE
        else:
            grip_inner = self._format_gripper_inner(gripper_distance, interval, max_distance, min_distance)
        ok = self.send_subloop1_blocking(
            arm_inner,
            grip_inner,
            timeout_ms=move_timeout_ms,
            settle_target_q=q[: self.joint_count],
            settle_timeout_s=settle_timeout_s,
        )
        if ok and gripper_distance is not None:
            self._last_gripper_distance_sent = self._clamp_gripper_distance(
                gripper_distance, max_distance, min_distance
            )
            self._last_gripper_cmd_time = time.time()
        return ok

    def enable(self):
        self.send_dual_model("Enable", NOT_RUN_EXECUTE, timeout_ms=5000)

    def disable(self):
        self._send_rpc_sync("{Disable}", timeout_ms=5000)
