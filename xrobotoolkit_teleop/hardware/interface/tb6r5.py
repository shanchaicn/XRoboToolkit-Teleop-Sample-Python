import os
import shutil
import sys
import platform
import random
import subprocess
import threading
import time
import ctypes
from typing import Literal, Optional, Tuple

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
DEFAULT_SUBLOOP1_EXEC_TIMEOUT_MS = 6_000_000
DEFAULT_SUBLOOP1_EXIT_TIMEOUT_MS = DEFAULT_DUAL_MODEL_MOVE_TIMEOUT_MS
NOT_RUN_EXECUTE = "NotRunExecute"
GRIPPER_YS_STATUS_FORMAT = "<d"  # TwoFingerGripperYSStatus: actual_pos (double, mm)
# VR joystick axes are in [-1, 1]; |axis|==1 is full stick deflection.
JOYSTICK_AXIS_DEFLECTION_MAX = 1.0
# Back-compat alias used by CLI / controller imports
DEFAULT_JOG_ANY_C_ASYNC_TIMEOUT_MS = DEFAULT_JOG_ASYNC_TIMEOUT_MS

# JogAnyJ RPC dialect (controller / Codeit version).
# 44: include --zone_ratio and --clear_buffer (current).
# 45: omit --zone_ratio and --clear_buffer.
DEFAULT_CD_VERSION = 44
CD_VERSIONS = (44, 45)

# subloop: 1=wrap JogAnyJ in SubLoop1 --exec={...} (default); 0=direct {arm||grip} dual-model.
DEFAULT_SUBLOOP = 1
SUBLOOP_MODES = (0, 1)

# g_model: 0=arm-only (no || dual-model), 2=MoveTwoFingersGripper, 3=JogAnyJ j1 meters
DEFAULT_G_MODEL = 2
G_MODELS = (0, 2, 3)
# Collect / teleop g_model=3: 0–80mm ↔ j1 0.000–0.080 m; vel/acc/dec=0.5
DEFAULT_GRIPPER_G3_MAX_D = 80.0
DEFAULT_GRIPPER_G3_CMD_DELTA_MM = 0.1
GRIPPER_JOG_MM_FULL_SCALE = 80.0
GRIPPER_JOG_JOINT_MIN = 0.0
GRIPPER_JOG_JOINT_MAX = 0.080
GRIPPER_JOG_JOINT_VEL = 0.5
GRIPPER_JOG_JOINT_ACC = 0.5
GRIPPER_JOG_JOINT_DEC = 0.5


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
        g_model: int = DEFAULT_G_MODEL,
        cd_version: int = DEFAULT_CD_VERSION,
        subloop: int = DEFAULT_SUBLOOP,
        gripper_jog_joint_vel: float = GRIPPER_JOG_JOINT_VEL,
        gripper_jog_joint_acc: float = GRIPPER_JOG_JOINT_ACC,
        gripper_jog_joint_dec: float = GRIPPER_JOG_JOINT_DEC,
        gripper_jog_joint_min: float = GRIPPER_JOG_JOINT_MIN,
        gripper_jog_joint_max: float = GRIPPER_JOG_JOINT_MAX,
        gripper_jog_mm_full_scale: float = GRIPPER_JOG_MM_FULL_SCALE,
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
        self.g_model = int(g_model)
        if self.g_model not in G_MODELS:
            raise ValueError(f"g_model must be one of {G_MODELS}, got {self.g_model}")
        self.cd_version = int(cd_version)
        if self.cd_version not in CD_VERSIONS:
            raise ValueError(f"cd_version must be one of {CD_VERSIONS}, got {self.cd_version}")
        self.subloop = int(subloop)
        if self.subloop not in SUBLOOP_MODES:
            raise ValueError(f"subloop must be one of {SUBLOOP_MODES}, got {self.subloop}")
        self.gripper_jog_joint_vel = max(float(gripper_jog_joint_vel), 0.0)
        self.gripper_jog_joint_acc = max(float(gripper_jog_joint_acc), 0.0)
        self.gripper_jog_joint_dec = max(float(gripper_jog_joint_dec), 0.0)
        self.gripper_jog_joint_min = float(gripper_jog_joint_min)
        self.gripper_jog_joint_max = float(gripper_jog_joint_max)
        self.gripper_jog_mm_full_scale = max(float(gripper_jog_mm_full_scale), 1e-6)
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
        self._rpc_generation = 0
        self._last_command_block_log_time = 0.0
        self._cartesian_stream_count = 0
        self._joint_stream_count = 0
        self._gripper_stream_count = 0
        self._jog_async_pending = 0
        self._subloop1_stream_pending = 0
        self._max_jog_async_pending = 32
        self._rpc_sync_lock = threading.Lock()
        self._cached_gripper_mm: Optional[float] = None
        self._gripper_feedback_healthy = False
        self._last_gripper_distance_sent: Optional[float] = None
        self._last_gripper_cmd_time = 0.0
        self._gripper_cmd_delta_mm = (
            DEFAULT_GRIPPER_G3_CMD_DELTA_MM if self.g_model == 3 else DEFAULT_GRIPPER_CMD_DELTA_MM
        )
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
                self._send_rpc_sync("{Disable}", timeout_ms=3000, sleep_s=0.1, ignore_subcmd_errors=True)
            except Exception:
                pass
            self.client = None
        self._topic = None
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
        # g_model=0: single-model {Enable}/{Start} (no gripper / no ||)
        # g_model=2: Enable||NotRunExecute; g_model=3: both sides Enable/Start
        if self.g_model == 0:
            if not self._send_rpc_sync("{Enable}", timeout_ms=500000, sleep_s=0.1):
                return False
            if not self._send_rpc_sync(
                "{Start}", timeout_ms=500000, sleep_s=0.1, ignore_subcmd_errors=True
            ):
                return False
            return True
        if self.g_model == 3:
            enable_right = "Enable"
            start_right = "Start"
        else:
            enable_right = NOT_RUN_EXECUTE
            start_right = NOT_RUN_EXECUTE
        if not self.send_dual_model("Enable", enable_right, timeout_ms=500000, sleep_s=0.1):
            return False
        if not self.send_dual_model("Start", start_right, timeout_ms=500000, sleep_s=0.1, ignore_subcmd_errors=True):
            return False
        return True

    def _ensure_command_channel(self) -> bool:
        """True when RPC client can accept Jog/Enable commands.

        Intentionally does NOT require topic health: topic drops must not silently
        block JogAnyJ while grip is still active (Clear/Enable sync still works).
        """
        if self.client is None or not self._rpc_ready:
            self._log_command_channel_block("rpc_not_ready")
            return False
        with self._state_lock:
            if self._server_in_error:
                reason = self._last_rpc_error or "server_in_error"
                self._log_command_channel_block(reason)
                return False
        return True

    def _log_command_channel_block(self, reason: str) -> None:
        now = time.time()
        if now - self._last_command_block_log_time < 1.0:
            return
        self._last_command_block_log_time = now
        topic = "ok" if self._topic_healthy else "down"
        print(f"[TB6R5] command channel blocked ({reason}); topic={topic}")

    def begin_command_epoch(self, reason: str = "") -> int:
        """Invalidate in-flight async fault latching and clear local error gate."""
        with self._state_lock:
            self._rpc_generation += 1
            gen = self._rpc_generation
            self._server_in_error = False
            self._last_rpc_error = None
            # Do not force pending to 0 here: stale callbacks still decrement safely.
        suffix = f" ({reason})" if reason else ""
        print(f"[TB6R5] command epoch -> {gen}{suffix}")
        return gen

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
        self._gripper_stream_count = 0
        with self._state_lock:
            self._server_in_error = False
            self._last_rpc_error = None

    def reset_joint_stream(self):
        """Reset JogAnyJ stream state (call when grip is first pressed in placo_ik)."""
        self._joint_stream_count = 0
        self._gripper_stream_count = 0
        with self._state_lock:
            # Keep pending count honest for in-flight callbacks; only clear error gate.
            self._server_in_error = False
            self._last_rpc_error = None

    def prepare_g3_gripper_enable(self) -> bool:
        """g_model=3: before teleop stream, clear/recover then re-enable gripper model.

        Sends ``{Clear}``, ``{Recover}``, then ``{NotRunExecute||Enable}`` so a
        dropped gripper enable does not block subsequent dual-model JogAnyJ.
        """
        if self.g_model != 3:
            return True
        # Bump epoch BEFORE Clear so stale JogAnyJ async errors cannot re-latch
        # _server_in_error after we finish Enable.
        self.begin_command_epoch("g3 pre-teleop Clear/Recover/Enable")
        print("[TB6R5] g3 pre-teleop: {Clear} + {Recover} + {NotRunExecute||Enable}")
        ok_clear = self._send_rpc_sync(
            "{Clear}",
            timeout_ms=3000,
            sleep_s=0.05,
            ignore_subcmd_errors=True,
        )
        with self._state_lock:
            self._server_in_error = False
            self._last_rpc_error = None
        ok_rec = self._send_rpc_sync(
            "{Recover}",
            timeout_ms=5000,
            sleep_s=0.05,
            ignore_subcmd_errors=True,
        )
        with self._state_lock:
            self._server_in_error = False
            self._last_rpc_error = None
        ok_en = self.send_dual_model(
            NOT_RUN_EXECUTE,
            "Enable",
            timeout_ms=5000,
            sleep_s=0.05,
            ignore_subcmd_errors=True,
        )
        with self._state_lock:
            self._server_in_error = False
            self._last_rpc_error = None
        if not (ok_clear and ok_rec and ok_en):
            print("[TB6R5] g3 pre-teleop: Clear/Recover/Enable failed (continuing teleop)")
        return bool(ok_clear and ok_rec and ok_en)

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
        if self.g_model == 0:
            return self._send_rpc_sync("{Start}", timeout_ms=3000, ignore_subcmd_errors=True)
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
        self.begin_command_epoch("clear_error")
        ok = self._send_rpc_sync("{Clear}", timeout_ms=3000, ignore_subcmd_errors=True)
        if ok:
            with self._state_lock:
                self._server_in_error = False
                self._last_rpc_error = None
            self._cartesian_stream_count = 0
            self._joint_stream_count = 0
        return ok

    def clear_and_recover(self, *, log_prefix: str = "clear-recover") -> bool:
        """Clear + Recover so a fresh dual-model MoveAbsJ / Disable can proceed after Jog stream."""
        self._server_in_error = False
        print(f"[TB6R5] {log_prefix}: {{Clear}} + {{Recover}}")
        ok_clear = self._send_rpc_sync("{Clear}", timeout_ms=5000, sleep_s=0.05, ignore_subcmd_errors=True)
        self._server_in_error = False
        ok_rec = self._send_rpc_sync("{Recover}", timeout_ms=5000, sleep_s=0.1, ignore_subcmd_errors=True)
        self._server_in_error = False
        self._cartesian_stream_count = 0
        self._joint_stream_count = 0
        self._gripper_stream_count = 0
        with self._state_lock:
            self._jog_async_pending = 0
            self._subloop1_stream_pending = 0
        return bool(ok_clear and ok_rec)

    def _jog_any_j_zone_clear_suffix(
        self,
        clear_buffer: int = 0,
        zone_ratio: Optional[float] = None,
    ) -> str:
        """cd-version 44: --zone_ratio/--clear_buffer; 45: omit both."""
        if self.cd_version >= 45:
            return ""
        if zone_ratio is None:
            zone_ratio = self.zone_ratio
        return f" --zone_ratio={float(zone_ratio):.4f} --clear_buffer={int(clear_buffer)}"

    def _format_jog_any_j_cmd(
        self,
        q: np.ndarray,
        clear_buffer: int = 0,
        zone_ratio: Optional[float] = None,
        last_count: int = DEFAULT_JOG_ANY_J_LAST_COUNT,
    ) -> str:
        val_str = self._format_jointtarget_value(q)
        cmd = (
            "{JogAnyJ --jointtarget_value="
            + val_str
            + self._jog_any_j_zone_clear_suffix(clear_buffer, zone_ratio)
            + f" --last_count={int(last_count)}"
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
        generation = self._rpc_generation

        def _on_response(status: int, resp_list):
            with self._state_lock:
                self._jog_async_pending = max(0, self._jog_async_pending - 1)
                # Ignore faults from commands issued before the latest epoch
                # (e.g. Clear aborted in-flight JogAnyJ after grip re-press).
                if generation != self._rpc_generation:
                    return
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

    def _read_gripper_mm_from_model1(self) -> Tuple[Optional[float], bool]:
        """g_model=3: state from RT models[1].joints[joint_start].position (meters → mm)."""
        if self._topic is None:
            return None, False
        try:
            state = self._topic.get_system_state()
            if not state.has_rt():
                return None, False
            rt = state.get_rt()
            if not rt.models or len(rt.models) < 2:
                return None, False
            model = rt.models[1]
            start = int(getattr(model, "joint_start_idx", 0) or 0)
            if start < 0 or start >= len(rt.models_joints):
                return None, False
            pos_m = float(rt.models_joints[start].position)
            return self._gripper_joint_to_mm(pos_m), True
        except Exception:
            return None, False

    def _read_gripper_mm_from_topic(self) -> Tuple[Optional[float], bool]:
        """Gripper opening feedback (mm).

        g_model=3: RT ``models[1].joint[0].position`` (meters mapped to mm).
        g_model=2: NRT YS gripper ``actual_pos`` (mm).
        """
        if self.g_model == 3:
            return self._read_gripper_mm_from_model1()
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
        if not self._ensure_command_channel():
            return False

        xyz = np.asarray(xyz, dtype=float).ravel()[:3].copy()
        quat = np.asarray(quat_wxyz, dtype=float).ravel()[:4].copy()

        # clear_buffer=0 on first cmd after reset_cartesian_stream(); then always 1.
        clear_buffer = self._resolve_stream_clear_buffer(self._cartesian_stream_count, clear_buffer)

        cmd = self._format_jog_any_c_cmd(xyz, quat, clear_buffer)
        with self._state_lock:
            if self._jog_async_pending >= self._max_jog_async_pending:
                return False
        ok = self._send_jog_any_c_async(cmd)
        if not ok:
            return False

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
            max_distance = DEFAULT_GRIPPER_G3_MAX_D if self.g_model == 3 else DEFAULT_GRIPPER_MAX_D
        if min_distance is None:
            min_distance = DEFAULT_GRIPPER_MIN_D
        lo = float(min_distance)
        hi = float(max_distance)
        if lo > hi:
            lo, hi = hi, lo
        return max(lo, min(float(distance), hi))

    def _gripper_mm_to_joint(self, distance_mm: float) -> float:
        """Map opening mm → JogAnyJ j1 (meters). 0–full_scale ↔ joint_min–joint_max."""
        scale = self.gripper_jog_mm_full_scale
        mm = max(0.0, min(float(distance_mm), scale))
        j_lo = self.gripper_jog_joint_min
        j_hi = self.gripper_jog_joint_max
        return j_lo + (mm / scale) * (j_hi - j_lo)

    def _gripper_joint_to_mm(self, j1_m: float) -> float:
        """Inverse of ``_gripper_mm_to_joint`` for state[6] from model[1].joint[0].position."""
        j_lo = self.gripper_jog_joint_min
        j_hi = self.gripper_jog_joint_max
        span = max(j_hi - j_lo, 1e-9)
        t = (float(j1_m) - j_lo) / span
        return max(0.0, min(t, 1.0)) * self.gripper_jog_mm_full_scale

    def _format_gripper_jog_any_j_inner(
        self,
        distance: float,
        max_distance: Optional[float] = None,
        min_distance: Optional[float] = None,
        clear_buffer: Optional[int] = None,
        last_count: int = DEFAULT_JOG_ANY_J_LAST_COUNT,
    ) -> str:
        """g_model=3: right SubLoop1 slot = JogAnyJ on j1 only (meters)."""
        distance = self._clamp_gripper_distance(distance, max_distance, min_distance)
        j1 = self._gripper_mm_to_joint(distance)
        if clear_buffer is None:
            clear_buffer = 0 if self._gripper_stream_count == 0 else 1
        val_str = self._format_jointtarget_value(np.array([j1], dtype=float))
        return (
            f"JogAnyJ --jointtarget_value={val_str}"
            f"{self._jog_any_j_zone_clear_suffix(clear_buffer)}"
            f" --last_count={int(last_count)}"
            f" --joint_vel={self.gripper_jog_joint_vel:.4f}"
            f" --joint_acc={self.gripper_jog_joint_acc:.4f}"
            f" --joint_dec={self.gripper_jog_joint_dec:.4f}"
        )

    def _format_gripper_inner(
        self,
        distance: float,
        interval: float,
        max_distance: Optional[float] = None,
        min_distance: Optional[float] = None,
        clear_buffer: Optional[int] = None,
    ) -> str:
        if self.g_model == 3:
            return self._format_gripper_jog_any_j_inner(
                distance,
                max_distance=max_distance,
                min_distance=min_distance,
                clear_buffer=clear_buffer,
            )
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
        if self.g_model == 0:
            return f"{{{SUBLOOP1_CMD} --exec={{{arm_inner}}}{immediate_suffix}}}"
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
        """SubLoop1 exit; dual-model when gripper present, arm-only for g_model=0."""
        if self.g_model == 0:
            return f"{{{SUBLOOP1_CMD} --exec={{exit}}}}"
        return f"{{{SUBLOOP1_CMD} --exec={{exit}}||{SUBLOOP1_CMD} --exec={{exit}}}}"

    def _send_subloop1_first_async(self, cmd: str) -> bool:
        """First SubLoop1 exec in a session: async with a long timeout (manual requirement)."""
        generation = self._rpc_generation

        def _on_response(status: int, resp_list):
            with self._state_lock:
                self._jog_async_pending = max(0, self._jog_async_pending - 1)
                if generation != self._rpc_generation:
                    return
            if status < 0:
                with self._state_lock:
                    if generation != self._rpc_generation:
                        return
                    self._server_in_error = True
                    self._last_rpc_error = f"SubLoop1 first exec async timeout (status={status})"
                print(f"[TB6R5] {self._last_rpc_error}")
                return
            for r in resp_list or []:
                if r.code < 0:
                    with self._state_lock:
                        if generation != self._rpc_generation:
                            return
                        self._server_in_error = True
                        self._last_rpc_error = r.message
                    print(f"[TB6R5] SubLoop1 first exec error: {r.message}")
                    return
            with self._state_lock:
                if generation != self._rpc_generation:
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
        """Fire subsequent SubLoop1 exec without blocking (expect_resp=False).

        Avoid registering per-frame pending responses; otherwise Ctrl+C/exit leaves
        hundreds of CallAsync entries that time out as status=-3 /
        ``No valid pending response for seqID``.
        """

        def _on_response(status: int, resp_list):
            return

        msg = self._rpc.Msg(cmd)
        msg.setMsgID(10001)
        msg.setMsgSeqID(random.randint(1, 10000))
        # Prefer expect_resp=False when the binding supports it.
        try:
            ok = self.client.CallAsync(msg, DEFAULT_SUBLOOP1_EXEC_TIMEOUT_MS, _on_response, False)
        except TypeError:
            ok = self.client.CallAsync(msg, DEFAULT_SUBLOOP1_EXEC_TIMEOUT_MS, _on_response)
            if ok:
                with self._state_lock:
                    self._subloop1_stream_pending += 1
            return bool(ok)
        return bool(ok)

    def _finalize_subloop1_session(self) -> None:
        self._subloop1_active = False
        self._subloop1_exiting = False
        with self._state_lock:
            self._jog_async_pending = 0
            self._subloop1_stream_pending = 0
            self._gripper_stream_count = 0
            self._joint_stream_count = 0

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
        if not self.use_subloop1:
            return True
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
        drain_timeout_s: float = 2.0,
    ) -> bool:
        if not self.use_subloop1:
            return True
        if not self._subloop1_active and not self._subloop1_exiting:
            return True
        if timeout_ms is None:
            timeout_ms = DEFAULT_SUBLOOP1_EXIT_TIMEOUT_MS
        # Drain tracked async before exit to reduce seqID timeout spam.
        deadline = time.time() + max(float(drain_timeout_s), 0.0)
        while time.time() < deadline:
            with self._state_lock:
                pending = self._jog_async_pending + self._subloop1_stream_pending
            if pending == 0:
                break
            time.sleep(0.01)
        if blocking_exit and self._subloop1_active:
            # Blocking path (homing/shutdown): wait for motion to settle before sync exit.
            self._wait_motion_settled(settle_timeout_s)
        return self.send_subloop1_exit(timeout_ms=timeout_ms, blocking=blocking_exit)

    @property
    def use_subloop1(self) -> bool:
        """True when Jog/gripper are wrapped in ``SubLoop1 --exec={...}`` (``subloop=1``)."""
        return self.subloop != 0

    def send_subloop1(
        self,
        arm_inner: str,
        gripper_distance: Optional[float],
        interval: Optional[float] = None,
        max_distance: Optional[float] = None,
        min_distance: Optional[float] = None,
        timeout_ms: int = DEFAULT_SUBLOOP1_EXEC_TIMEOUT_MS,
        immediate: Optional[bool] = None,
        gripper_clear_buffer: Optional[int] = None,
    ) -> bool:
        """Send arm (+ optional gripper) exec.

        ``g_model=0``: arm-only single-model RPC (no ``||``):
          ``subloop=0`` → ``{JogAnyJ/JogAnyC ...}``; ``subloop=1`` → ``{SubLoop1 --exec={arm}}``.
        ``g_model=2/3`` + ``subloop=1``: ``{SubLoop1 --exec={arm}||SubLoop1 --exec={grip}}``.
        ``g_model=2/3`` + ``subloop=0``: direct ``{arm||grip}``.

        ``gripper_distance=None`` -> gripper slot is NotRunExecute (ignored for g_model=0).
        """
        if not self._ensure_command_channel():
            return False
        if immediate is None:
            immediate = self.subloop1_immediate
        if interval is None:
            interval = DEFAULT_TWO_FINGERS_GRIPPER_INTERVAL

        arm_inner = (arm_inner or NOT_RUN_EXECUTE).strip()

        # Arm-only (no gripper hardware): never emit dual-model ||.
        if self.g_model == 0:
            if not arm_inner or arm_inner == NOT_RUN_EXECUTE:
                return False
            if not self.use_subloop1:
                cmd = f"{{{arm_inner}}}"
                with self._state_lock:
                    if self._jog_async_pending >= self._max_jog_async_pending:
                        return False
                if "JogAnyC" in arm_inner:
                    ok = self._send_jog_any_c_async(cmd)
                else:
                    ok = self._send_jog_any_j_async(cmd)
                return bool(ok)
            cmd = self.format_subloop1_exec_cmd(arm_inner, NOT_RUN_EXECUTE, immediate=immediate)
            if self._subloop1_exiting:
                return False
            if not self._subloop1_active:
                return self._send_subloop1_first_async(cmd)
            return self._send_subloop1_stream_async(cmd)

        if gripper_distance is None:
            grip_inner = NOT_RUN_EXECUTE
        else:
            if gripper_clear_buffer is None and self.g_model == 3:
                gripper_clear_buffer = 0 if self._gripper_stream_count == 0 else 1
            grip_inner = self._format_gripper_inner(
                gripper_distance,
                interval,
                max_distance,
                min_distance,
                clear_buffer=gripper_clear_buffer,
            )
        if not self.use_subloop1:
            cmd = self.format_dual_model_cmd(arm_inner, grip_inner)
            ok = self._send_subloop1_stream_async(cmd)
        else:
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
            if self.g_model == 3:
                self._gripper_stream_count += 1
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
        if self.g_model == 0:
            return False
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
        """Send JogAnyJ + gripper (MoveTwoFingers or JogAnyJ j1) via SubLoop1."""
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
        grip_arg = None if self.g_model == 0 else (gripper_distance if gripper_changed else None)
        grip_clear = None
        if grip_arg is not None and self.g_model == 3:
            grip_clear = 0 if self._gripper_stream_count == 0 else 1
        ok = self.send_subloop1(
            arm_inner,
            grip_arg,
            interval=interval,
            max_distance=max_distance,
            min_distance=min_distance,
            gripper_clear_buffer=grip_clear,
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
        """Send JogAnyC + gripper via SubLoop1 (g_model=2 MoveTwoFingers / g_model=3 JogAnyJ j1)."""
        if not self._ensure_command_channel():
            return False

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
        grip_arg = None if self.g_model == 0 else (gripper_distance if gripper_changed else None)
        grip_clear = None
        if grip_arg is not None and self.g_model == 3:
            grip_clear = 0 if self._gripper_stream_count == 0 else 1
        ok = self.send_subloop1(
            arm_inner,
            grip_arg,
            interval=interval,
            max_distance=max_distance,
            min_distance=min_distance,
            gripper_clear_buffer=grip_clear,
        )
        if not ok:
            return False

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
        post_exit_clear: Optional[bool] = None,
    ) -> bool:
        """Home via exit SubLoop1 (if active), then MoveAbsJ.

        ``g_model=0`` (arm-only)::

            {MoveAbsJ ...}

        ``g_model=2/3``::

            {NotRunExecute||gripper open ...}
            {MoveAbsJ ...||NotRunExecute}

        Matches policy_infer / test_subloop1_direct_rpc_new_gripper shutdown path.
        ``post_exit_clear``: None = Clear+Recover only if a SubLoop1 session was open;
        True/False force on/off (teleop Ctrl+C should pass True after a stream session).
        """
        if q is None:
            q = np.zeros(self.joint_count)
        if self.has_fault():
            self.clear_error()
        q = np.asarray(q, dtype=float).ravel()
        if interval is None:
            interval = DEFAULT_TWO_FINGERS_GRIPPER_INTERVAL

        was_streaming = bool(self._subloop1_active or self._subloop1_exiting)
        self.exit_subloop1_if_active(timeout_ms=move_timeout_ms, blocking_exit=True)
        do_clear = was_streaming if post_exit_clear is None else bool(post_exit_clear)
        if do_clear:
            self.clear_and_recover(log_prefix="go_home post-exit")
            time.sleep(0.15)

        if self.g_model == 0:
            arm_inner = self._format_move_abs_j_inner(q)
            print(f"[TB6R5] go_home MoveAbsJ: {{{arm_inner}}}")
            if not self._send_move_abs_j(q, move_timeout_ms=int(move_timeout_ms)):
                print("[TB6R5] go_home: MoveAbsJ FAILED")
                return False
            settled = self._wait_motion_settled(settle_timeout_s, target_q=q[: self.joint_count])
            if not settled:
                print(
                    f"[TB6R5] go_home: MoveAbsJ did not settle within {settle_timeout_s:.1f}s "
                    "(topic missing or target not reached)."
                )
                return False
            return True

        if gripper_distance is not None:
            grip_inner = self._format_gripper_inner(
                gripper_distance, interval, max_distance, min_distance, clear_buffer=0
            )
            print(f"[TB6R5] go_home gripper: {{{NOT_RUN_EXECUTE}||{grip_inner}}}")
            if not self.send_dual_model(
                NOT_RUN_EXECUTE,
                grip_inner,
                timeout_ms=min(int(move_timeout_ms), 60_000),
                sleep_s=0.05,
                ignore_subcmd_errors=True,
            ):
                print("[TB6R5] go_home: gripper open FAILED")
                return False
            self._last_gripper_distance_sent = self._clamp_gripper_distance(
                gripper_distance, max_distance, min_distance
            )
            self._last_gripper_cmd_time = time.time()
            if self.g_model == 3:
                self._gripper_stream_count = 1

        arm_inner = self._format_move_abs_j_inner(q)
        print(f"[TB6R5] go_home MoveAbsJ: {{{arm_inner}||{NOT_RUN_EXECUTE}}}")
        if not self.send_dual_model(
            arm_inner,
            NOT_RUN_EXECUTE,
            timeout_ms=int(move_timeout_ms),
            sleep_s=0.05,
            ignore_subcmd_errors=True,
        ):
            print("[TB6R5] go_home: MoveAbsJ FAILED")
            return False

        settled = self._wait_motion_settled(settle_timeout_s, target_q=q[: self.joint_count])
        if not settled:
            print(
                f"[TB6R5] go_home: MoveAbsJ did not settle within {settle_timeout_s:.1f}s "
                "(topic missing or target not reached)."
            )
            return False
        return True

    def enable(self):
        if self.g_model == 0:
            return self._send_rpc_sync("{Enable}", timeout_ms=5000)
        right = "Enable" if self.g_model == 3 else NOT_RUN_EXECUTE
        self.send_dual_model("Enable", right, timeout_ms=5000)

    def disable(self):
        if self._server_in_error:
            self.clear_and_recover(log_prefix="disable-prep")
        if not self._send_rpc_sync("{Disable}", timeout_ms=5000, ignore_subcmd_errors=True):
            self.clear_and_recover(log_prefix="disable-retry")
            self._send_rpc_sync("{Disable}", timeout_ms=5000, ignore_subcmd_errors=True)
