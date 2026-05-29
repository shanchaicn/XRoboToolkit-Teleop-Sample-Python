import os
import threading
import time
from typing import Dict, Literal, Optional

import meshcat.transformations as tf
import numpy as np
from placo_utils.visualization import frame_viz

from xrobotoolkit_teleop.common.base_hardware_teleop_controller import (
    HardwareTeleopController,
)
from xrobotoolkit_teleop.hardware.interface.tb6r5 import (
    DEFAULT_JOG_ANY_C_ASYNC_TIMEOUT_MS,
    DEFAULT_JOG_ANY_JOINT_ACC,
    DEFAULT_JOG_ANY_JOINT_DEC,
    DEFAULT_JOG_ANY_JOINT_VEL,
    DEFAULT_TELEOP_MODE,
    DEFAULT_ZONE_RATIO,
    TB6R5Interface,
    TeleopMode,
)
from xrobotoolkit_teleop.utils.geometry import (
    R_HEADSET_TO_WORLD,
    quat_diff_as_angle_axis,
    quaternion_to_angle_axis,
)
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH

# Official RevA1 URDF (SolidWorks export, aligned with controller robottarget / base_link).
DEFAULT_URDF_PATH = os.path.join(
    ASSET_PATH,
    "TB6-R5-RevA1-urdf/urdf/7260501-000000-001 TB6-R5-RevA1-urdf.urdf",
)
DEFAULT_TCP_LINK_NAME = "ee_Link"
TB6R5_JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
DEFAULT_SCALE_FACTOR = 1.5
DEFAULT_ROBOT_IP = "192.168.11.11"
DEFAULT_RPC_PORT = 5868
DEFAULT_HOME_JOINT_DEG = (0.0, -90.0, 90.0, -90.0, -90.0, 0.0)
# RevA1 URDF sets velocity="0" on every joint; Placo then caps |dq| to 0 per step.
DEFAULT_PLACO_JOINT_VMAX_RAD_S = 3.14
DEFAULT_LOG_JOINT_COUNT = 7
DEFAULT_GRIPPER_TRIGGER_NAME = "right_trigger"
DEFAULT_GRIPPER_TRIGGER_THRESHOLD = 0.5
DEFAULT_GRIPPER_OPEN_CMD = 0.0
DEFAULT_GRIPPER_CLOSED_CMD = 1.0
DEFAULT_GRIPPER_OBSERVATION = 0.0
DEFAULT_REALSENSE_SERIAL_DICT = {
    "realsense_0": "135522071053",
    "realsense_1": "327122073649",
}

# Per control cycle: limit each JogAnyC command step (larger = faster catch-up to PICO target)
DEFAULT_CARTESIAN_MAX_STEP_POS_M = 0.03
DEFAULT_CARTESIAN_MAX_STEP_ROT_RAD = 0.1
DEFAULT_JOG_ANY_C_POSITION_ONLY = True
DEFAULT_JOG_ANY_C_ORIENTATION_ONLY = False
JogAnyCInterruptMode = Literal["on", "off"]
DEFAULT_JOG_ANY_C_INTERRUPT: JogAnyCInterruptMode = "off"
CARTESIAN_WARMUP_FRAMES = 0
CARTESIAN_MIN_START_DELTA_M = 0.001
CARTESIAN_MIN_START_DELTA_ROT_RAD = 0.01
HOME_SETTLE_TIME_S = 3.0
GRIP_ACTIVE_THRESHOLD = 0.5
# World-frame direction for "tool Z points down" in jog_any_c position-only mode
WORLD_DOWN = np.array([0.0, 0.0, -1.0])

DEFAULT_TB6R5_MANIPULATOR_CONFIG = {
    "right_hand": {
        "link_name": DEFAULT_TCP_LINK_NAME,
        "pose_source": "right_controller",
        "control_trigger": "right_grip",
    },
}


def _is_sim_only(robot_ip: Optional[str]) -> bool:
    if robot_ip is None:
        return True
    return str(robot_ip).strip().lower() in ("none", "sim", "offline", "")


class TB6R5TeleopController(HardwareTeleopController):
    """Teleoperation controller for the TB6-R5 6-DOF robotic arm.

    teleop_mode:
      - placo_ik (default): PICO delta -> Placo IK -> JogAnyJ
      - jog_any_c: Topic robottarget + PICO delta -> JogAnyC (no Placo IK).
        URDF ee_Link / base_link are aligned with controller robottarget (RevA1 export).
        Default is position-only with tool Z locked to world -Z; set
        jog_any_c_position_only=False for full 6-DOF pose tracking.
        jog_any_c_orientation_only=True for rotation-only (TCP position latched at grip).
        Each grip segment: first JogAnyC/JogAnyJ uses clear_buffer=0, then clear_buffer=1 (async RPC).
        Target = latched TCP at grip + cumulative PICO delta (position and rotation).
    Pass robot_ip='none' for Placo visualization only (placo_ik mode).
    """

    def __init__(
        self,
        robot_urdf_path: str = DEFAULT_URDF_PATH,
        manipulator_config: dict = DEFAULT_TB6R5_MANIPULATOR_CONFIG,
        robot_ip: str = DEFAULT_ROBOT_IP,
        rpc_port: int = DEFAULT_RPC_PORT,
        R_headset_world: np.ndarray = R_HEADSET_TO_WORLD,
        scale_factor: float = DEFAULT_SCALE_FACTOR,
        visualize_placo: bool = False,
        control_rate_hz: int = 50,
        enable_log_data: bool = True,
        log_dir: str = "logs/tb6r5",
        log_freq: float = 50,
        enable_camera: bool = False,
        camera_fps: int = 30,
        camera_serial_dict: Optional[Dict[str, str]] = None,
        camera_width: int = 640,
        camera_height: int = 480,
        enable_camera_depth: bool = True,
        enable_camera_compression: bool = True,
        camera_jpg_quality: int = 85,
        log_joint_count: int = DEFAULT_LOG_JOINT_COUNT,
        gripper_trigger_name: str = DEFAULT_GRIPPER_TRIGGER_NAME,
        gripper_trigger_threshold: float = DEFAULT_GRIPPER_TRIGGER_THRESHOLD,
        gripper_open_cmd: float = DEFAULT_GRIPPER_OPEN_CMD,
        gripper_closed_cmd: float = DEFAULT_GRIPPER_CLOSED_CMD,
        gripper_observation_default: float = DEFAULT_GRIPPER_OBSERVATION,
        teleop_mode: TeleopMode = DEFAULT_TELEOP_MODE,
        cartesian_max_step_pos_m: float = DEFAULT_CARTESIAN_MAX_STEP_POS_M,
        cartesian_max_step_rot_rad: float = DEFAULT_CARTESIAN_MAX_STEP_ROT_RAD,
        jog_any_c_position_only: bool = DEFAULT_JOG_ANY_C_POSITION_ONLY,
        jog_any_c_orientation_only: bool = DEFAULT_JOG_ANY_C_ORIENTATION_ONLY,
        jog_any_c_interrupt: JogAnyCInterruptMode = DEFAULT_JOG_ANY_C_INTERRUPT,
        zone_ratio: float = DEFAULT_ZONE_RATIO,
        jog_any_c_async_timeout_ms: int = DEFAULT_JOG_ANY_C_ASYNC_TIMEOUT_MS,
        cartesian_vel: Optional[float] = None,
        cartesian_acc: Optional[float] = None,
        cartesian_dec: Optional[float] = None,
        joint_vel: float = DEFAULT_JOG_ANY_JOINT_VEL,
        joint_acc: float = DEFAULT_JOG_ANY_JOINT_ACC,
        joint_dec: float = DEFAULT_JOG_ANY_JOINT_DEC,
        jog_any_c_preview_only: bool = False,
    ):
        self.robot_ip = robot_ip
        self.rpc_port = rpc_port
        self.teleop_mode = teleop_mode
        self.jog_any_c_preview_only = bool(jog_any_c_preview_only)
        self.cartesian_max_step_pos_m = float(cartesian_max_step_pos_m)
        self.cartesian_max_step_rot_rad = float(cartesian_max_step_rot_rad)
        if jog_any_c_orientation_only:
            jog_any_c_position_only = False
        if jog_any_c_orientation_only and jog_any_c_position_only:
            raise ValueError("jog_any_c_orientation_only and jog_any_c_position_only are mutually exclusive")
        self.jog_any_c_position_only = bool(jog_any_c_position_only)
        self.jog_any_c_orientation_only = bool(jog_any_c_orientation_only)
        self.jog_any_c_interrupt = jog_any_c_interrupt == "on"
        self.zone_ratio = max(float(zone_ratio), 0.0)
        self.jog_any_c_async_timeout_ms = max(int(jog_any_c_async_timeout_ms), 100)
        self.cartesian_vel = cartesian_vel
        self.cartesian_acc = cartesian_acc
        self.cartesian_dec = cartesian_dec
        self.joint_vel = float(joint_vel)
        self.joint_acc = float(joint_acc)
        self.joint_dec = float(joint_dec)
        self.enable_camera = enable_camera
        self.control_rate_hz = control_rate_hz
        self.sim_only = _is_sim_only(robot_ip)
        self.arm: Optional[TB6R5Interface] = None
        self.log_joint_count = max(int(log_joint_count), len(TB6R5_JOINT_NAMES))
        self.gripper_trigger_name = gripper_trigger_name
        self.gripper_trigger_threshold = float(gripper_trigger_threshold)
        self.gripper_open_cmd = float(gripper_open_cmd)
        self.gripper_closed_cmd = float(gripper_closed_cmd)
        self.gripper_observation_default = float(gripper_observation_default)
        self._last_gripper_action_cmd = self.gripper_open_cmd
        self.camera_serial_dict = camera_serial_dict or DEFAULT_REALSENSE_SERIAL_DICT
        self.camera_serial_to_name = {serial: name for name, serial in self.camera_serial_dict.items()}
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_fps = camera_fps
        self.enable_camera_depth = enable_camera_depth
        self.enable_camera_compression = enable_camera_compression
        self.camera_jpg_quality = camera_jpg_quality
        self._prev_a_button_state = False
        self._home_q = np.deg2rad(DEFAULT_HOME_JOINT_DEG)
        self._cartesian_target: Dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._cartesian_last_target: Dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._cartesian_warmup: Dict[str, int] = {}
        self._cartesian_started: Dict[str, bool] = {}
        self._cartesian_blocked_until_release: Dict[str, bool] = {}
        self._cartesian_debug_frames: Dict[
            str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ] = {}
        self._cartesian_grip_values: Dict[str, float] = {}
        self._locked_down_quat: Dict[str, np.ndarray] = {}
        self._viz_controller_origin_xyz: Dict[str, np.ndarray] = {}
        self._viz_controller_origin_quat: Dict[str, np.ndarray] = {}
        self._viz_placo_anchor_xyz: Dict[str, np.ndarray] = {}
        self._viz_placo_anchor_quat: Dict[str, np.ndarray] = {}
        self._jog_rot_ref_controller_quat: Dict[str, np.ndarray] = {}
        self._jog_rot_ref_tcp_quat: Dict[str, np.ndarray] = {}
        self._last_home_time = 0.0
        self._last_cartesian_wait_print = 0.0
        self._preview_stream_count: Dict[str, int] = {}
        self._last_preview_cmd_print = 0.0
        self._last_jog_any_j_preview_print = 0.0
        self._last_jog_any_j_preview_q: Optional[np.ndarray] = None
        self._joint_preview_stream_count: Dict[str, int] = {}
        self._jog_any_c_cmd_formatter: Optional[TB6R5Interface] = None
        if self.jog_any_c_preview_only:
            if teleop_mode != "jog_any_c":
                raise ValueError("jog_any_c_preview_only requires teleop_mode='jog_any_c'")
            self._jog_any_c_cmd_formatter = TB6R5Interface(
                enable_topic=False,
                joint_count=self.log_joint_count,
                jog_any_c_interrupt=self.jog_any_c_interrupt,
                zone_ratio=self.zone_ratio,
                cartesian_vel=self.cartesian_vel,
                cartesian_acc=self.cartesian_acc,
                cartesian_dec=self.cartesian_dec,
                joint_vel=self.joint_vel,
                joint_acc=self.joint_acc,
                joint_dec=self.joint_dec,
            )
            print(
                "[TB6R5] jog_any_c preview: PICO teleop + Placo/Meshcat only. "
                "JogAnyC robottarget is computed and visualized; no RPC is sent."
            )
        elif teleop_mode == "jog_any_c" and _is_sim_only(robot_ip):
            print(
                "[TB6R5] jog_any_c requires hardware Topic (robottarget). "
                "Use --jog-any-c-preview for PICO-only verification, "
                "--teleop-mode placo_ik for sim-only, or connect a real robot."
            )
        super().__init__(
            robot_urdf_path=robot_urdf_path,
            manipulator_config=manipulator_config,
            R_headset_world=R_headset_world,
            floating_base=False,
            scale_factor=scale_factor,
            q_init=np.deg2rad(DEFAULT_HOME_JOINT_DEG),
            visualize_placo=visualize_placo,
            control_rate_hz=control_rate_hz,
            enable_log_data=enable_log_data,
            log_dir=log_dir,
            log_freq=log_freq,
            enable_camera=enable_camera,
            camera_fps=camera_fps,
        )

        # Ensure step limits are physically safe to prevent sudden jumps and violent rotations.
        # Safe speed caps: Max position speed = 0.5 m/s, Max angular velocity = 1.5 rad/s (approx. 86 deg/s).
        max_safe_step_pos = 0.5 / self.control_rate_hz
        max_safe_step_rot = 1.5 / self.control_rate_hz

        if self.cartesian_max_step_pos_m > max_safe_step_pos:
            old_val = self.cartesian_max_step_pos_m
            self.cartesian_max_step_pos_m = max_safe_step_pos
            print(
                f"[TB6R5] Warning: Provided cartesian_max_step_pos_m ({old_val:.4f} m) "
                f"exceeds safe limit. Clamped to {self.cartesian_max_step_pos_m:.4f} m "
                f"(maximum Cartesian velocity cap: 0.5 m/s)."
            )

        if self.cartesian_max_step_rot_rad > max_safe_step_rot:
            old_val = self.cartesian_max_step_rot_rad
            self.cartesian_max_step_rot_rad = max_safe_step_rot
            print(
                f"[TB6R5] Warning: Provided cartesian_max_step_rot_rad ({old_val:.4f} rad) "
                f"exceeds safe limit. Clamped to {self.cartesian_max_step_rot_rad:.4f} rad "
                f"(maximum end-effector angular velocity cap: 1.5 rad/s ~ 85.9 deg/s)."
            )

        # Set consistent, robust trigger hysteresis for both modes.
        self.grip_threshold = GRIP_ACTIVE_THRESHOLD
        self.grip_release_threshold = 0.2
        print(
            f"[TB6R5] Grip trigger hysteresis: activate > {self.grip_threshold:.2f}, "
            f"release <= {self.grip_release_threshold:.2f}."
        )

    # ------------------------------------------------------------------
    # Placo setup
    # ------------------------------------------------------------------

    def _placo_setup(self):
        super()._placo_setup()

        if self.solver is not None:
            self.solver.enable_joint_limits(True)
            # URDF velocity=0 makes IK freeze near the feedback pose; override before enabling.
            self.placo_robot.set_velocity_limits(DEFAULT_PLACO_JOINT_VMAX_RAD_S)
            self.solver.dt = self.dt
            self.solver.enable_velocity_limits(True)
            print(
                "[TB6R5] Placo joint limits enabled; velocity limits enabled with "
                f"vmax={DEFAULT_PLACO_JOINT_VMAX_RAD_S:.2f} rad/s (URDF velocity=0 overridden)."
            )

        joint_names = list(TB6R5_JOINT_NAMES)
        for link_name in {cfg["link_name"] for cfg in self.manipulator_config.values()}:
            try:
                self.placo_robot.get_T_world_frame(link_name)
            except RuntimeError as exc:
                raise ValueError(f"URDF link '{link_name}' not found in Placo model: {exc}") from exc
        self.joint_slice = slice(
            self.placo_robot.get_joint_offset(joint_names[0]),
            self.placo_robot.get_joint_offset(joint_names[-1]) + 1,
        )
        if self.arm is not None:
            q = self.arm.get_joint_positions()
            self.placo_robot.state.q[self.joint_slice] = q[: len(TB6R5_JOINT_NAMES)]
            self.placo_robot.update_kinematics()
            self.sync_end_effector_poses_to_placo_tasks()

    # ------------------------------------------------------------------
    # Robot lifecycle
    # ------------------------------------------------------------------

    def _robot_setup(self):
        if self.arm is not None:
            return
        if self.sim_only:
            print("TB6-R5 sim-only mode: skipping hardware connection (Placo viz only).")
            return
        print(f"Setting up TB6-R5 at {self.robot_ip}:{self.rpc_port} (mode={self.teleop_mode}) ...")
        self.arm = TB6R5Interface(
            ip=self.robot_ip,
            rpc_port=self.rpc_port,
            joint_count=self.log_joint_count,
            rpc_cmd_rate_hz=self.control_rate_hz,
            jog_any_c_interrupt=self.jog_any_c_interrupt,
            zone_ratio=self.zone_ratio,
            jog_any_c_async_timeout_ms=self.jog_any_c_async_timeout_ms,
            cartesian_vel=self.cartesian_vel,
            cartesian_acc=self.cartesian_acc,
            cartesian_dec=self.cartesian_dec,
            joint_vel=self.joint_vel,
            joint_acc=self.joint_acc,
            joint_dec=self.joint_dec,
        )
        self.arm.connect()
        if self.teleop_mode == "jog_any_c" and not self.arm.is_robottarget_healthy():
            print("[TB6R5] Warning: Topic robottarget not available yet; wait for RT feedback before teleop.")
        print(f"TB6-R5 ready (control/RPC rate: {self.control_rate_hz} Hz).")

    def _shutdown_robot(self):
        if self.arm is None:
            return
        print("Shutting down TB6-R5 ...")
        self.arm.go_home(self._home_q)
        time.sleep(1.0)
        self.arm.disable()
        print("TB6-R5 shut down.")

    # ------------------------------------------------------------------
    # Control loop hooks
    # ------------------------------------------------------------------

    def _pre_ik_update(self):
        self._check_home_button()

    def _check_home_button(self):
        """Press A once to move the arm to DEFAULT_HOME_JOINT_DEG."""
        a_pressed = self.xr_client.get_button_state_by_name("A")
        if a_pressed and not self._prev_a_button_state:
            self._go_to_home_pose()
        self._prev_a_button_state = a_pressed

    def _go_to_home_pose(self):
        print(f"[TB6R5] A button: homing to {DEFAULT_HOME_JOINT_DEG} deg")
        self.placo_robot.state.q[self.joint_slice] = self._home_q.copy()
        self.placo_robot.update_kinematics()
        self.sync_end_effector_poses_to_placo_tasks()

        for name in self.manipulator_config:
            self.ref_ee_xyz[name] = None
            self.ref_ee_quat[name] = None
            self.ref_controller_xyz[name] = None
            self.ref_controller_quat[name] = None
            self.active[name] = False
            self._locked_down_quat.pop(name, None)
        self._cartesian_target.clear()
        self._cartesian_warmup.clear()
        self._cartesian_last_target.clear()
        self._cartesian_started.clear()
        self._cartesian_blocked_until_release.clear()
        self._cartesian_debug_frames.clear()
        self._locked_down_quat.clear()
        self._reset_viz_controller_session_anchors()
        self._jog_rot_ref_controller_quat.clear()
        self._jog_rot_ref_tcp_quat.clear()
        self._preview_stream_count.clear()

        if self.arm is not None:
            self.arm.go_home(self._home_q)
        self._last_home_time = time.time()

    def _limit_cartesian_step(
        self,
        previous_xyz: np.ndarray,
        previous_quat: np.ndarray,
        target_xyz: np.ndarray,
        target_quat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        step_xyz = np.asarray(target_xyz, dtype=float).ravel()[:3] - np.asarray(previous_xyz, dtype=float).ravel()[:3]
        step_norm = np.linalg.norm(step_xyz)
        max_step = self.cartesian_max_step_pos_m
        if step_norm > max_step:
            target_xyz = previous_xyz + step_xyz * (max_step / step_norm)

        previous_quat = np.asarray(previous_quat, dtype=float).ravel()[:4]
        target_quat = np.asarray(target_quat, dtype=float).ravel()[:4]
        if np.linalg.norm(previous_quat) > 1e-9:
            previous_quat = previous_quat / np.linalg.norm(previous_quat)
        if np.linalg.norm(target_quat) > 1e-9:
            target_quat = target_quat / np.linalg.norm(target_quat)

        # Limit orientation by moving a bounded angle along the shortest quaternion path.
        if np.dot(previous_quat, target_quat) < 0.0:
            target_quat = -target_quat
        delta_rot = quat_diff_as_angle_axis(previous_quat, target_quat)
        angle = np.linalg.norm(delta_rot)
        max_rot_step = self.cartesian_max_step_rot_rad
        if angle > max_rot_step:
            axis = delta_rot / angle
            limited_delta = tf.quaternion_about_axis(max_rot_step, axis)
            target_quat = tf.quaternion_multiply(limited_delta, previous_quat)
            target_quat = target_quat / np.linalg.norm(target_quat)

        return target_xyz, target_quat

    def _xr_pose_to_controller_world(self, xr_pose) -> tuple[np.ndarray, np.ndarray]:
        """Map raw PICO controller pose to robot/world frame (same as teleop input)."""
        controller_xyz = np.array([xr_pose[0], xr_pose[1], xr_pose[2]], dtype=float)
        controller_quat = np.array(
            [xr_pose[6], xr_pose[3], xr_pose[4], xr_pose[5]],
            dtype=float,
        )
        controller_xyz = self.R_headset_world @ controller_xyz

        R_transform = np.eye(4)
        R_transform[:3, :3] = self.R_headset_world
        R_quat = tf.quaternion_from_matrix(R_transform)
        # Match R @ p: orientation is R_headset_to_world * R_controller (not similarity conjugation).
        controller_quat = tf.quaternion_multiply(R_quat, controller_quat)
        quat_norm = np.linalg.norm(controller_quat)
        if quat_norm > 1e-9:
            controller_quat = controller_quat / quat_norm
        return controller_xyz, controller_quat

    @staticmethod
    def _scale_quaternion_angle(quat_wxyz: np.ndarray, scale: float) -> np.ndarray:
        quat_wxyz = np.asarray(quat_wxyz, dtype=float).ravel()[:4]
        if abs(float(scale) - 1.0) < 1e-9:
            return quat_wxyz
        angle_axis = quaternion_to_angle_axis(quat_wxyz)
        angle = np.linalg.norm(angle_axis)
        if angle < 1e-9:
            return np.array([1.0, 0.0, 0.0, 0.0])
        return tf.quaternion_about_axis(float(angle * scale), angle_axis / angle)

    def _compute_grip_relative_target_quat(self, src_name: str, ctrl_quat: np.ndarray) -> np.ndarray:
        """Map grip-relative controller rotation to robottarget TCP orientation."""
        ref_ctrl = self._jog_rot_ref_controller_quat[src_name].copy()
        ref_tcp = self._jog_rot_ref_tcp_quat[src_name].copy()

        # Shortest-path alignment in controller hemisphere to prevent transient flip
        ctrl_quat = np.asarray(ctrl_quat, dtype=float).ravel()[:4].copy()
        if np.dot(ctrl_quat, ref_ctrl) < 0.0:
            ctrl_quat = -ctrl_quat

        q_rel = tf.quaternion_multiply(ctrl_quat, tf.quaternion_inverse(ref_ctrl))
        q_rel = self._scale_quaternion_angle(q_rel, self.scale_factor)
        target_quat = tf.quaternion_multiply(q_rel, ref_tcp)

        # Shortest-path alignment in TCP hemisphere
        if np.dot(target_quat, ref_tcp) < 0.0:
            target_quat = -target_quat

        norm = np.linalg.norm(target_quat)
        if norm < 1e-9:
            return ref_tcp
        return target_quat / norm

    def _process_xr_pose(self, xr_pose, src_name):
        """Cumulative PICO delta from grip start (absolute tracking mode)."""
        controller_xyz, controller_quat = self._xr_pose_to_controller_world(xr_pose)
        if self.ref_controller_xyz[src_name] is None:
            self.ref_controller_xyz[src_name] = controller_xyz.copy()
            self.ref_controller_quat[src_name] = controller_quat.copy()
            return np.zeros(3), np.zeros(3)

        delta_xyz = (controller_xyz - self.ref_controller_xyz[src_name]) * self.scale_factor
        delta_rot = quat_diff_as_angle_axis(self.ref_controller_quat[src_name], controller_quat)
        return delta_xyz, delta_rot

    def _reset_viz_controller_session_anchors(self):
        self._viz_controller_origin_xyz.clear()
        self._viz_controller_origin_quat.clear()
        self._viz_placo_anchor_xyz.clear()
        self._viz_placo_anchor_quat.clear()

    def _ensure_viz_controller_session_anchor(self, src_name: str, xr_pose):
        """Latch controller + Placo anchor once per session; not reset on grip."""
        if src_name in self._viz_controller_origin_xyz:
            return

        ctrl_xyz, ctrl_quat = self._xr_pose_to_controller_world(xr_pose)
        link_name = self.manipulator_config[src_name]["link_name"]
        anchor_xyz, anchor_quat = self._get_link_pose(link_name)

        self._viz_controller_origin_xyz[src_name] = ctrl_xyz.copy()
        self._viz_controller_origin_quat[src_name] = ctrl_quat.copy()
        self._viz_placo_anchor_xyz[src_name] = np.asarray(anchor_xyz, dtype=float).copy()
        self._viz_placo_anchor_quat[src_name] = np.asarray(anchor_quat, dtype=float).copy()

    def _compute_scaled_controller_viz_pose(
        self,
        src_name: str,
        xr_pose,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Scaled controller frame in Placo world; session anchor, independent of grip.

        Position: anchor + scale_factor * (controller - session_origin).
        Orientation: absolute controller pose in world frame (R_HEADSET_TO_WORLD), so
        wrist rotation is visible even when position delta is small.
        """
        self._ensure_viz_controller_session_anchor(src_name, xr_pose)
        ctrl_xyz, ctrl_quat = self._xr_pose_to_controller_world(xr_pose)
        origin_xyz = self._viz_controller_origin_xyz[src_name]
        anchor_xyz = self._viz_placo_anchor_xyz[src_name]

        delta_xyz = (ctrl_xyz - origin_xyz) * self.scale_factor
        viz_xyz = anchor_xyz + delta_xyz
        return viz_xyz, ctrl_quat

    def _update_scaled_controller_viz(self):
        for src_name, config in self.manipulator_config.items():
            xr_pose = self.xr_client.get_pose_by_name(config["pose_source"])
            viz_xyz, viz_quat = self._compute_scaled_controller_viz_pose(src_name, xr_pose)
            frame_viz(
                f"viz_controller_scaled/{src_name}",
                self._pose_matrix(viz_xyz, viz_quat),
            )

    def _set_placo_effector_target(self, src_name: str, xyz: np.ndarray, quat_wxyz: np.ndarray):
        """Placo viz target in base_link world (same frame as URDF ee_Link / robottarget)."""
        xyz = np.asarray(xyz, dtype=float).ravel()[:3]
        quat_wxyz = np.asarray(quat_wxyz, dtype=float).ravel()[:4]
        if self.effector_control_mode[src_name] == "position":
            self.effector_task[src_name].target_world = xyz
            return
        target_pose = tf.quaternion_matrix(quat_wxyz)
        target_pose[:3, 3] = xyz
        self.effector_task[src_name].T_world_frame = target_pose

    def _is_target_safe(self, src_name: str, xyz: np.ndarray, quat_wxyz: np.ndarray) -> bool:
        """Dry-run IK solver in Placo to verify if target is within URDF joint limits."""
        if self.placo_robot is None or self.solver is None:
            return True

        # 1. Save current joint state
        q_save = self.placo_robot.state.q.copy()

        # 2. Set the dry-run target
        self._set_placo_effector_target(src_name, xyz, quat_wxyz)

        # 3. Solve IK locally for a few iterations (steps are small, 5 is plenty)
        for _ in range(5):
            self.solver.solve(True)
            self.placo_robot.update_kinematics()

        # 4. Check if the solved joint angles are within limits
        safe = True
        margin = np.deg2rad(1.0)  # Use a conservative 1.0 degree buffer margin
        for joint_name in TB6R5_JOINT_NAMES:
            try:
                q_val = self.placo_robot.get_joint(joint_name)
                q_min, q_max = self.placo_robot.get_joint_limits(joint_name)
                if q_val < (q_min + margin) or q_val > (q_max - margin):
                    safe = False
                    break
            except Exception:
                pass

        # 5. Restore saved state
        self.placo_robot.state.q = q_save
        self.placo_robot.update_kinematics()

        return safe

    @staticmethod
    def _pose_matrix(xyz: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
        pose = tf.quaternion_matrix(quat_wxyz)
        pose[:3, 3] = np.asarray(xyz, dtype=float).ravel()[:3]
        return pose

    @staticmethod
    def _quat_angle(q1: np.ndarray, q2: np.ndarray) -> float:
        return float(np.linalg.norm(quat_diff_as_angle_axis(np.asarray(q1), np.asarray(q2))))

    @staticmethod
    def _quat_tool_z_down_from_current(quat_wxyz: np.ndarray, world_down: np.ndarray = WORLD_DOWN) -> np.ndarray:
        """Rotate current TCP orientation so tool +Z aligns with world_down (default: world -Z)."""
        quat_wxyz = np.asarray(quat_wxyz, dtype=float).ravel()
        if np.linalg.norm(quat_wxyz) < 1e-9:
            return np.array([1.0, 0.0, 0.0, 0.0])
        R = tf.quaternion_matrix(quat_wxyz)[:3, :3]
        z_tool = R[:, 2]
        z_target = np.asarray(world_down, dtype=float)
        z_target = z_target / (np.linalg.norm(z_target) + 1e-12)
        dot = float(np.clip(np.dot(z_tool, z_target), -1.0, 1.0))
        if dot > 0.999:
            return quat_wxyz / np.linalg.norm(quat_wxyz)
        axis = np.cross(z_tool, z_target)
        norm_v = np.linalg.norm(axis)
        if norm_v < 1e-9:
            axis = np.array([1.0, 0.0, 0.0])
            if abs(z_tool[0]) < 0.5:
                axis = np.array([0.0, 1.0, 0.0])
            angle = np.pi
        else:
            axis = axis / norm_v
            angle = np.arccos(dot)
        delta = tf.quaternion_about_axis(angle, axis)
        q_down = tf.quaternion_multiply(delta, quat_wxyz)
        return q_down / np.linalg.norm(q_down)

    def _format_preview_jog_any_c_cmd(
        self,
        xyz: np.ndarray,
        quat_wxyz: np.ndarray,
        stream_count: int,
    ) -> str:
        if self._jog_any_c_cmd_formatter is None:
            raise RuntimeError("preview JogAnyC formatter is not initialized")
        clear_buffer = 0 if stream_count == 0 else 1
        return self._jog_any_c_cmd_formatter._format_jog_any_c_cmd(xyz, quat_wxyz, clear_buffer)

    def _format_preview_jog_any_j_cmd(self, q: np.ndarray, stream_count: int = 0) -> str:
        formatter = self.arm
        if formatter is None:
            formatter = self._jog_any_c_cmd_formatter
        if formatter is None:
            formatter = TB6R5Interface(
                enable_topic=False,
                joint_vel=self.joint_vel,
                joint_acc=self.joint_acc,
                joint_dec=self.joint_dec,
                jog_any_c_interrupt=self.jog_any_c_interrupt,
            )
        clear_buffer = 0 if stream_count == 0 else 1
        return formatter._format_jog_any_j_cmd(q, clear_buffer=clear_buffer)

    def _preview_or_apply_sim_joint_target(self, q_des: np.ndarray, src_name: str):
        """Sim-only placo_ik path: apply solved q to Placo/Meshcat and print JogAnyJ payload."""
        now = time.time()
        q_des = np.asarray(q_des, dtype=float).ravel()[:6].copy()
        n = len(TB6R5_JOINT_NAMES)
        self.placo_robot.state.q[self.joint_slice] = self._pad_log_vector(q_des, n)[:n]
        self.placo_robot.update_kinematics()

        if now - self._last_jog_any_j_preview_print < 0.05:
            return

        stream_count = self._joint_preview_stream_count.get(src_name, 0)
        print(f">>>sim [preview] msg: {self._format_preview_jog_any_j_cmd(q_des, stream_count)}")
        self._joint_preview_stream_count[src_name] = stream_count + 1
        self._last_jog_any_j_preview_print = now
        self._last_jog_any_j_preview_q = q_des

    def _get_debug_topic_tcp_pose(self, src_name: str) -> tuple[np.ndarray, np.ndarray]:
        if self.arm is not None and self.arm.is_robottarget_healthy() and not self.jog_any_c_preview_only:
            topic_xyz, topic_quat, _ = self.arm.get_robottarget()
            return topic_xyz, topic_quat
        link_name = self.manipulator_config[src_name]["link_name"]
        return self._get_link_pose(link_name)

    def _record_cartesian_debug(
        self,
        src_name: str,
        ref_xyz: np.ndarray,
        ref_quat: np.ndarray,
        target_xyz: np.ndarray,
        target_quat: np.ndarray,
    ):
        topic_xyz, topic_quat = self._get_debug_topic_tcp_pose(src_name)
        self._cartesian_debug_frames[src_name] = (
            topic_xyz,
            topic_quat,
            ref_xyz,
            ref_quat,
            target_xyz,
            target_quat,
        )

    def _update_placo_viz(self):
        super()._update_placo_viz()
        self._update_scaled_controller_viz()
        if self.teleop_mode != "jog_any_c":
            return
        for name, frames in self._cartesian_debug_frames.items():
            topic_xyz, topic_quat, ref_xyz, ref_quat, target_xyz, target_quat = frames
            frame_viz(f"jog_any_c/{name}/01_topic_tcp", self._pose_matrix(topic_xyz, topic_quat))
            frame_viz(f"jog_any_c/{name}/02_current_tcp", self._pose_matrix(ref_xyz, ref_quat))
            frame_viz(f"jog_any_c/{name}/03_target_to_send", self._pose_matrix(target_xyz, target_quat))
            locked_quat = self._locked_down_quat.get(name)
            if locked_quat is not None:
                frame_viz(f"jog_any_c/{name}/04_locked_z_down", self._pose_matrix(ref_xyz, locked_quat))

    def _get_teleop_tcp_pose(self, src_name: str) -> tuple[np.ndarray, np.ndarray, bool]:
        """TCP in base_link / robottarget frame (URDF ee_Link aligned with controller)."""
        if self.arm is not None and self.arm.is_robottarget_healthy():
            return self.arm.get_robottarget()
        link_name = self.manipulator_config[src_name]["link_name"]
        ee_xyz, ee_quat = self._get_link_pose(link_name)
        return ee_xyz, ee_quat, True

    def _can_start_cartesian_teleop(self) -> bool:
        if self.jog_any_c_preview_only:
            return True
        if self.arm is None:
            return False
        if time.time() - self._last_home_time < HOME_SETTLE_TIME_S:
            return False
        return self.arm.is_topic_healthy()

    def _warn_cartesian_not_ready(self, src_name: str):
        now = time.time()
        if now - self._last_cartesian_wait_print < 2.0:
            return
        self._last_cartesian_wait_print = now
        reasons = []
        if self.arm is None:
            reasons.append("hardware not connected (sim-only; jog_any_c needs a real robot)")
        else:
            if now - self._last_home_time < HOME_SETTLE_TIME_S:
                reasons.append("home settling")
            if not self.arm.is_topic_healthy():
                reasons.append("topic feedback unavailable")
        detail = ", ".join(reasons) if reasons else "unknown"
        print(f"[TB6R5] {src_name}: waiting for JogAnyC ({detail}).")

    def _update_cartesian_teleop(self):
        """Topic robottarget + PICO delta -> JogAnyC (no Placo IK).

        jog_any_c_position_only=True: position-only, tool Z locked to world -Z.
        jog_any_c_orientation_only=True: rotation-only, TCP position latched at grip.
        Both False: full 6-DOF pose from PICO position + rotation.
        """
        self.placo_robot.update_kinematics()

        for src_name, config in self.manipulator_config.items():
            xr_grip_val = self.xr_client.get_key_value_by_name(config["control_trigger"])
            self._cartesian_grip_values[src_name] = xr_grip_val
            grip_active = xr_grip_val > GRIP_ACTIVE_THRESHOLD
            if not grip_active:
                self._cartesian_blocked_until_release[src_name] = False
            self.active[src_name] = grip_active and not self._cartesian_blocked_until_release.get(src_name, False)

            if self.active[src_name]:
                if self.ref_ee_xyz[src_name] is None:
                    if not self._can_start_cartesian_teleop():
                        self._warn_cartesian_not_ready(src_name)
                        self.active[src_name] = False
                        self._cartesian_target.pop(src_name, None)
                        continue
                    xyz, quat, _ = self._get_teleop_tcp_pose(src_name)
                    interrupt_note = ", clear_buffer=1 after 1st cmd"
                    if self.jog_any_c_position_only:
                        ref_quat = self._quat_tool_z_down_from_current(quat)
                        self._locked_down_quat[src_name] = ref_quat.copy()
                        print(
                            f"{src_name} is activated (JogAnyC, position-only, tool Z locked to world -Z"
                            f"{interrupt_note})."
                        )
                    elif self.jog_any_c_orientation_only:
                        ref_quat = quat.copy()
                        if np.linalg.norm(ref_quat) > 1e-9:
                            ref_quat = ref_quat / np.linalg.norm(ref_quat)
                        print(
                            f"{src_name} is activated (JogAnyC, orientation-only, TCP position latched"
                            f"{interrupt_note})."
                        )
                    else:
                        ref_quat = quat.copy()
                        if np.linalg.norm(ref_quat) > 1e-9:
                            ref_quat = ref_quat / np.linalg.norm(ref_quat)
                        print(f"{src_name} is activated (JogAnyC, full 6-DOF pose{interrupt_note}).")
                    self.ref_ee_xyz[src_name] = xyz.copy()
                    self.ref_ee_quat[src_name] = ref_quat.copy()
                    self.ref_controller_xyz[src_name] = None
                    self.ref_controller_quat[src_name] = None
                    xr_pose_act = self.xr_client.get_pose_by_name(config["pose_source"])
                    _, ctrl_quat_act = self._xr_pose_to_controller_world(xr_pose_act)
                    self._jog_rot_ref_controller_quat[src_name] = ctrl_quat_act.copy()
                    self._jog_rot_ref_tcp_quat[src_name] = ref_quat.copy()
                    self._process_xr_pose(xr_pose_act, src_name)
                    self._cartesian_warmup[src_name] = CARTESIAN_WARMUP_FRAMES
                    self._cartesian_last_target[src_name] = (xyz.copy(), ref_quat.copy())
                    self._cartesian_started[src_name] = False
                    if self.arm is not None:
                        self.arm.reset_cartesian_stream()
                    if self.jog_any_c_preview_only:
                        self._preview_stream_count[src_name] = 0
                    self._record_cartesian_debug(src_name, xyz, ref_quat, xyz, ref_quat)

                warmup = self._cartesian_warmup.get(src_name, 0)
                if warmup > 0:
                    self._cartesian_warmup[src_name] = warmup - 1
                    self._cartesian_target.pop(src_name, None)

                    # On the very last frame of warmup, re-capture stable reference poses
                    if self._cartesian_warmup[src_name] == 0:
                        xyz, quat, _ = self._get_teleop_tcp_pose(src_name)
                        if self.jog_any_c_position_only:
                            ref_quat = self._quat_tool_z_down_from_current(quat)
                            self._locked_down_quat[src_name] = ref_quat.copy()
                        else:
                            ref_quat = quat.copy()
                            if np.linalg.norm(ref_quat) > 1e-9:
                                ref_quat = ref_quat / np.linalg.norm(ref_quat)
                        self.ref_ee_xyz[src_name] = xyz.copy()
                        self.ref_ee_quat[src_name] = ref_quat.copy()
                        self.ref_controller_xyz[src_name] = None
                        self.ref_controller_quat[src_name] = None

                        xr_pose_act = self.xr_client.get_pose_by_name(config["pose_source"])
                        _, ctrl_quat_act = self._xr_pose_to_controller_world(xr_pose_act)
                        self._jog_rot_ref_controller_quat[src_name] = ctrl_quat_act.copy()
                        self._jog_rot_ref_tcp_quat[src_name] = ref_quat.copy()
                        self._process_xr_pose(xr_pose_act, src_name)
                        self._cartesian_last_target[src_name] = (xyz.copy(), ref_quat.copy())

                    self._set_placo_effector_target(src_name, self.ref_ee_xyz[src_name], self.ref_ee_quat[src_name])
                    self._record_cartesian_debug(
                        src_name,
                        self.ref_ee_xyz[src_name],
                        self.ref_ee_quat[src_name],
                        self.ref_ee_xyz[src_name],
                        self.ref_ee_quat[src_name],
                    )
                    continue
                else:
                    xr_pose = self.xr_client.get_pose_by_name(config["pose_source"])
                    delta_xyz, delta_rot = self._process_xr_pose(xr_pose, src_name)
                    if self.jog_any_c_position_only:
                        target_xyz = self.ref_ee_xyz[src_name] + delta_xyz
                        target_quat = self._locked_down_quat[src_name].copy()
                    elif self.jog_any_c_orientation_only:
                        target_xyz = self.ref_ee_xyz[src_name].copy()
                        _, ctrl_quat = self._xr_pose_to_controller_world(xr_pose)
                        target_quat = self._compute_grip_relative_target_quat(src_name, ctrl_quat)
                    else:
                        target_xyz = self.ref_ee_xyz[src_name] + delta_xyz
                        _, ctrl_quat = self._xr_pose_to_controller_world(xr_pose)
                        target_quat = self._compute_grip_relative_target_quat(src_name, ctrl_quat)
                    ref_xyz = self.ref_ee_xyz[src_name]
                    ref_quat = self.ref_ee_quat[src_name]

                if not self._cartesian_started.get(src_name, False):
                    pos_delta = np.linalg.norm(target_xyz - self.ref_ee_xyz[src_name])
                    rot_delta = self._quat_angle(self.ref_ee_quat[src_name], target_quat)
                    if self.jog_any_c_orientation_only:
                        below_deadband = rot_delta < CARTESIAN_MIN_START_DELTA_ROT_RAD
                    elif self.jog_any_c_position_only:
                        below_deadband = pos_delta < CARTESIAN_MIN_START_DELTA_M
                    else:
                        below_deadband = pos_delta < CARTESIAN_MIN_START_DELTA_M
                        below_deadband = below_deadband and rot_delta < CARTESIAN_MIN_START_DELTA_ROT_RAD
                    if below_deadband:
                        self._cartesian_target.pop(src_name, None)
                        self._set_placo_effector_target(src_name, ref_xyz, ref_quat)
                        self._record_cartesian_debug(
                            src_name,
                            ref_xyz,
                            ref_quat,
                            target_xyz,
                            target_quat,
                        )
                        continue
                    if self.jog_any_c_position_only:
                        print(f"[TB6R5] {src_name}: JogAnyC started after PICO position delta exceeded deadband.")
                    elif self.jog_any_c_orientation_only:
                        print(
                            f"[TB6R5] {src_name}: JogAnyC started after PICO rotation delta exceeded deadband "
                            f"(rot={rot_delta:.4f} rad)."
                        )
                    else:
                        print(
                            f"[TB6R5] {src_name}: JogAnyC started after PICO pose delta exceeded deadband "
                            f"(pos={pos_delta:.4f} m, rot={rot_delta:.4f} rad)."
                        )
                    self._cartesian_started[src_name] = True

                previous = self._cartesian_last_target.get(src_name)
                if previous is not None:
                    target_xyz, target_quat = self._limit_cartesian_step(
                        previous[0],
                        previous[1],
                        target_xyz,
                        target_quat,
                    )
                if self.jog_any_c_orientation_only:
                    target_xyz = self.ref_ee_xyz[src_name].copy()

                # Joint limits dry-run protection
                if previous is not None:
                    if not self._is_target_safe(src_name, target_xyz, target_quat):
                        target_xyz = previous[0].copy()
                        target_quat = previous[1].copy()
                        now = time.time()
                        if not hasattr(self, "_last_limit_warn_time"):
                            self._last_limit_warn_time = {}
                        if now - self._last_limit_warn_time.get(src_name, 0.0) >= 1.0:
                            print(
                                f"[TB6R5] Warning: {src_name} target blocked/clamped to prevent exceeding robot joint limits."
                            )
                            self._last_limit_warn_time[src_name] = now

                self._cartesian_last_target[src_name] = (target_xyz.copy(), target_quat.copy())
                self._cartesian_target[src_name] = (target_xyz, target_quat)
                self._set_placo_effector_target(src_name, target_xyz, target_quat)
                self._record_cartesian_debug(
                    src_name,
                    ref_xyz,
                    ref_quat,
                    target_xyz,
                    target_quat,
                )
            else:
                if self.ref_ee_xyz[src_name] is not None:
                    print(f"{src_name} is deactivated.")
                    self.ref_ee_xyz[src_name] = None
                    self.ref_ee_quat[src_name] = None
                    self.ref_controller_xyz[src_name] = None
                    self.ref_controller_quat[src_name] = None
                self._cartesian_target.pop(src_name, None)
                self._cartesian_last_target.pop(src_name, None)
                self._cartesian_warmup.pop(src_name, None)
                self._cartesian_started.pop(src_name, None)
                self._locked_down_quat.pop(src_name, None)
                self._jog_rot_ref_controller_quat.pop(src_name, None)
                self._jog_rot_ref_tcp_quat.pop(src_name, None)

                if self.arm is not None:
                    xyz, quat, _ = self._get_teleop_tcp_pose(src_name)
                    self._set_placo_effector_target(src_name, xyz, quat)
                    self._record_cartesian_debug(src_name, xyz, quat, xyz, quat)

    def _update_ik(self):
        if self.teleop_mode == "jog_any_c":
            self._update_cartesian_teleop()
            self._send_command()
            return

        prev_active = {k: self.active.get(k, False) for k in self.manipulator_config}
        super()._update_ik()
        for src_name in self.manipulator_config:
            if self.active.get(src_name, False) and not prev_active.get(src_name, False):
                self._joint_preview_stream_count[src_name] = 0
                if self.arm is not None:
                    self.arm.reset_joint_stream()
        # Match UR5 behavior: keep IK target frame aligned with actual EE when inactive
        for src_name, config in self.manipulator_config.items():
            if self.active.get(src_name, False):
                continue
            ee_xyz, ee_quat = self._get_link_pose(config["link_name"])
            self._set_placo_effector_target(src_name, ee_xyz, ee_quat)
        self._send_command()

    def _update_robot_state(self):
        if self.arm is None:
            return
        if self.arm.is_topic_healthy():
            # Topic joint angles match URDF joint axes (RevA1 export aligned with controller).
            self.placo_robot.state.q[self.joint_slice] = self.arm.get_joint_positions()[: len(TB6R5_JOINT_NAMES)]

    def _send_command(self):
        if self.teleop_mode == "jog_any_c":
            if threading.current_thread().name != "_ik_thread":
                return
            for name in self.manipulator_config:
                if not self.active.get(name, False):
                    continue
                target = self._cartesian_target.get(name)
                if target is None:
                    continue
                xyz, quat = target
                if self.jog_any_c_preview_only:
                    stream_count = self._preview_stream_count.get(name, 0)
                    cmd = self._format_preview_jog_any_c_cmd(xyz, quat, stream_count)
                    self._preview_stream_count[name] = stream_count + 1
                    now = time.time()
                    if stream_count == 0 or now - self._last_preview_cmd_print >= 0.5:
                        print(f"[preview] {name}: {cmd}")
                        self._last_preview_cmd_print = now
                    continue
                if self.arm is None:
                    continue
                self.arm.set_cartesian_target(xyz, quat)
                if self.arm.has_fault():
                    print(f"[TB6R5] {name}: JogAnyC fault; clearing and ending this grip segment.")
                    self.arm.clear_error()
                    self.active[name] = False
                    self._cartesian_blocked_until_release[name] = True
                    self.ref_ee_xyz[name] = None
                    self.ref_ee_quat[name] = None
                    self.ref_controller_xyz[name] = None
                    self.ref_controller_quat[name] = None
                    self._cartesian_target.pop(name, None)
                    self._cartesian_last_target.pop(name, None)
                    self._cartesian_warmup.pop(name, None)
                    self._cartesian_started.pop(name, None)
            return

        if self.arm is None:
            if not self.sim_only:
                return
        if threading.current_thread().name != "_ik_thread":
            return

        for name in self.manipulator_config:
            if self.active.get(name, False):
                q_des = self.placo_robot.state.q[self.joint_slice].copy()
                if self.sim_only:
                    self._preview_or_apply_sim_joint_target(q_des, name)
                elif self.arm is not None:
                    self.arm.set_joint_positions(q_des, force=True)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _pad_log_vector(self, values, length: int, fill_value: float = 0.0) -> np.ndarray:
        result = np.full(length, fill_value, dtype=float)
        if values is None:
            return result
        values = np.asarray(values, dtype=float).ravel()
        n = min(len(values), length)
        if n > 0:
            result[:n] = values[:n]
        return result

    def _get_gripper_action_for_logging(self) -> float:
        try:
            trigger_value = self.xr_client.get_key_value_by_name(self.gripper_trigger_name)
        except Exception:
            trigger_value = 0.0
        self._last_gripper_action_cmd = (
            self.gripper_closed_cmd if trigger_value > self.gripper_trigger_threshold else self.gripper_open_cmd
        )
        return self._last_gripper_action_cmd

    def _get_robot_state_for_logging(self) -> Dict:
        gripper_action = self._get_gripper_action_for_logging()
        gripper_observation = self.gripper_observation_default

        if self.arm is None:
            q = self._pad_log_vector(self.placo_robot.state.q[self.joint_slice].copy(), self.log_joint_count)
            qvel = np.zeros(self.log_joint_count)
            action_joints = q.copy()
            topic_healthy = False
            last_jog_any_j_cmd = None
        else:
            q = self._pad_log_vector(self.arm.get_joint_positions(), self.log_joint_count)
            qvel = self._pad_log_vector(self.arm.get_joint_velocities(), self.log_joint_count)
            last_jog_any_j_cmd = self.arm.get_last_joint_command()
            if last_jog_any_j_cmd is None:
                last_jog_any_j_cmd = self.placo_robot.state.q[self.joint_slice].copy()
            action_joints = self._pad_log_vector(last_jog_any_j_cmd, self.log_joint_count)
            topic_healthy = self.arm.is_topic_healthy()

        observation = np.concatenate([q, np.array([gripper_observation], dtype=float)])
        action = np.concatenate([action_joints, np.array([gripper_action], dtype=float)])

        state = {
            # 8-D data requested for downstream collection:
            # first 7 values are arm joints, last value is gripper open/close.
            "observation": observation,
            "action": action,
            "qpos": observation.copy(),
            "qvel": np.concatenate([qvel, np.array([0.0], dtype=float)]),
            "qpos_des": action.copy(),
            "joint_observation": q,
            "joint_action": action_joints,
            "gripper_qpos": gripper_observation,
            "gripper_action": gripper_action,
            "topic_healthy": topic_healthy,
        }

        if last_jog_any_j_cmd is not None:
            state["jog_any_j_cmd"] = action_joints.copy()

        if self.arm is None:
            return {
                **state,
                "topic_joint_qpos": q.copy(),
            }

        state["topic_joint_qpos"] = q.copy()
        if self.teleop_mode == "jog_any_c":
            xyz, quat, ok = self.arm.get_robottarget()
            state["tcp_xyz"] = xyz
            state["tcp_quat"] = quat
            state["tcp_ok"] = ok
            for name, (t_xyz, t_quat) in self._cartesian_target.items():
                state[f"{name}_tcp_target_xyz"] = t_xyz
                state[f"{name}_tcp_target_quat"] = t_quat
        return state

    def _get_camera_frame_for_logging(self) -> Dict:
        if not self.camera_interface:
            return {}

        if self.camera_interface.enable_compression:
            frames_by_serial = self.camera_interface.get_compressed_frames()
        else:
            frames_by_serial = self.camera_interface.get_frames()

        if not frames_by_serial:
            return {}

        frames_by_name = {}
        for serial, frames in frames_by_serial.items():
            camera_name = self.camera_serial_to_name.get(serial, serial)
            frames_by_name[camera_name] = frames
        return frames_by_name

    # ------------------------------------------------------------------
    # Camera (optional)
    # ------------------------------------------------------------------

    def _initialize_camera(self):
        if self.enable_camera:
            print("[TB6R5] Initializing RealSense cameras...")
            try:
                from xrobotoolkit_teleop.hardware.interface.realsense import RealSenseCameraInterface

                self.camera_interface = RealSenseCameraInterface(
                    width=self.camera_width,
                    height=self.camera_height,
                    fps=self.camera_fps,
                    serial_numbers=list(self.camera_serial_dict.values()),
                    enable_depth=self.enable_camera_depth,
                    enable_compression=self.enable_camera_compression,
                    jpg_quality=self.camera_jpg_quality,
                )
                self.camera_interface.start()
                print(f"[TB6R5] RealSense cameras ready: {list(self.camera_serial_dict.keys())}")
            except Exception as e:
                print(f"[TB6R5] Error initializing RealSense cameras: {e}")
                self.camera_interface = None
