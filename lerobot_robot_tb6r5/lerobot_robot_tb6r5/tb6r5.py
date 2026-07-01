"""LeRobot Robot implementation for TB6-R5."""

from __future__ import annotations

import logging
import time
from functools import cached_property
import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
try:
    from lerobot.processor import RobotAction, RobotObservation
except ImportError:  # lerobot >= 0.5
    from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError
from lerobot.robots.robot import Robot
from xrobotoolkit_teleop.hardware.interface.tb6r5 import (
    DEFAULT_GRIPPER_MAX_D,
    DEFAULT_GRIPPER_MIN_D,
    DEFAULT_TWO_FINGERS_GRIPPER_INTERVAL,
    TB6R5Interface,
)
from xrobotoolkit_teleop.hardware.tb6r5_command_gate import is_grip_session_active, reset_command_gate

from .config_tb6r5 import TB6R5Config

logger = logging.getLogger(__name__)

STATE_DIM = 7
JOINT_COUNT = 6
STATE_KEYS = tuple(f"state_{i}" for i in range(STATE_DIM))


class TB6R5(Robot):
    """TB6-R5 6-DOF arm with YS two-finger gripper.

    Proprioception layout matches existing TB6R5 LeRobot datasets:
      - state_0..state_5: joint angles (rad)
      - state_6: gripper opening distance (mm)
    """

    config_class = TB6R5Config
    name = "tb6r5"

    def __init__(self, config: TB6R5Config):
        super().__init__(config)
        self.config = config
        self.arm: TB6R5Interface | None = None
        self.cameras = make_cameras_from_configs(config.cameras)
        self._last_sent_action: dict[str, float] | None = None
        self._last_arm_rpc_time = 0.0
        self._last_gripper_rpc_time = 0.0
        self._last_gripper_input_mm: float | None = None
        self._was_streaming = False

    @cached_property
    def _state_ft(self) -> dict[str, type]:
        return dict.fromkeys(STATE_KEYS, float)

    @cached_property
    def _cameras_ft(self) -> dict[str, tuple[int, int, int]]:
        return {
            cam_name: (cfg.height, cfg.width, 3) for cam_name, cfg in self.config.cameras.items()
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple[int, int, int]]:
        return {**self._state_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._state_ft

    @property
    def is_connected(self) -> bool:
        if self.arm is None:
            arm_ok = False
        elif self.config.passive_mode:
            arm_ok = self.arm.is_topic_healthy()
        else:
            arm_ok = self.arm.is_connected
        cams_ok = all(cam.is_connected for cam in self.cameras.values()) if self.cameras else True
        return arm_ok and cams_ok

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        self.arm = TB6R5Interface(
            ip=self.config.robot_ip,
            rpc_port=self.config.rpc_port,
            enable_topic=self.config.enable_topic,
            zone_ratio=self.config.zone_ratio,
            joint_vel=self.config.joint_vel,
            joint_acc=self.config.joint_acc,
            joint_dec=self.config.joint_dec,
            subloop1_immediate=self.config.subloop1_immediate,
        )
        try:
            if self.config.passive_mode:
                self.arm.connect_topic_feedback(topic_wait_timeout_s=self.config.topic_wait_timeout_s)
            else:
                self.arm.connect(topic_wait_timeout_s=self.config.topic_wait_timeout_s)
                self.arm.reset_joint_stream()
                self.configure()

            for cam in self.cameras.values():
                cam.connect()
        except Exception:
            self.arm = None
            raise

        logger.info("%s connected (ip=%s).", self, self.config.robot_ip)

    def configure(self) -> None:
        if self.arm is None:
            return
        self.arm._gripper_cmd_delta_mm = max(float(self.config.gripper_cmd_delta), 0.0)

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        if self.arm is None:
            raise DeviceNotConnectedError(f"{self} arm is not initialized.")

        q = np.asarray(self.arm.get_joint_positions(), dtype=np.float64).ravel()
        q = q[:JOINT_COUNT]
        if q.size < JOINT_COUNT:
            q = np.pad(q, (0, JOINT_COUNT - q.size))

        gripper_mm = self.arm.get_gripper_distance_mm()
        if gripper_mm is None:
            gripper_mm = 0.0

        obs: RobotObservation = {key: 0.0 for key in STATE_KEYS}
        for i in range(JOINT_COUNT):
            obs[f"state_{i}"] = float(q[i])
        obs["state_6"] = float(gripper_mm)

        for cam_key, cam in self.cameras.items():
            frame = cam.read_latest()
            if frame is None:
                raise RuntimeError(f"Camera '{cam_key}' returned no frame.")
            obs[cam_key] = frame

        return obs

    def _arm_rpc_period_s(self) -> float:
        return 1.0 / max(float(self.config.arm_rpc_rate_hz), 0.1)

    def _gripper_rpc_period_s(self) -> float:
        return 1.0 / max(float(self.config.gripper_rpc_rate_hz), 0.1)

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        if self.config.passive_mode:
            return dict(action)

        if self.arm is None or not self.arm.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected to the robot.")

        target_q = np.array([float(action[f"state_{i}"]) for i in range(JOINT_COUNT)], dtype=np.float64)
        target_gripper = float(action["state_6"])

        present_q = np.asarray(self.arm.get_joint_positions(), dtype=np.float64).ravel()[:JOINT_COUNT]
        if present_q.size < JOINT_COUNT:
            present_q = np.pad(present_q, (0, JOINT_COUNT - present_q.size))

        if self.config.max_joint_step_rad is not None:
            max_step = float(self.config.max_joint_step_rad)
            delta = np.clip(target_q - present_q, -max_step, max_step)
            target_q = present_q + delta

        gripper_max_d = self.config.gripper_max_d or DEFAULT_GRIPPER_MAX_D
        gripper_min_d = self.config.gripper_min_d if self.config.gripper_min_d is not None else DEFAULT_GRIPPER_MIN_D
        target_gripper = float(np.clip(target_gripper, gripper_min_d, gripper_max_d))
        present_gripper = self.arm.get_gripper_distance_mm()
        if present_gripper is None:
            present_gripper = target_gripper
        present_gripper = float(present_gripper)

        if not is_grip_session_active():
            if self._was_streaming:
                self.arm.exit_subloop1_if_active(timeout_ms=5000, blocking_exit=False)
                self._was_streaming = False
            self._last_gripper_input_mm = None
            held = {f"state_{i}": float(present_q[i]) for i in range(JOINT_COUNT)}
            held["state_6"] = present_gripper
            self._last_sent_action = held
            return held

        sent = {f"state_{i}": float(target_q[i]) for i in range(JOINT_COUNT)}
        sent["state_6"] = target_gripper
        self._last_sent_action = sent

        arm_delta = float(np.max(np.abs(target_q - present_q)))
        hold_arm = arm_delta < float(self.config.arm_cmd_eps_rad)
        grip_input_delta = (
            abs(target_gripper - self._last_gripper_input_mm)
            if self._last_gripper_input_mm is not None
            else float("inf")
        )
        hold_grip = grip_input_delta < float(self.config.gripper_cmd_delta)

        if hold_arm and hold_grip:
            if self._was_streaming:
                self.arm.exit_subloop1_if_active(timeout_ms=5000, blocking_exit=False)
                self._was_streaming = False
            return sent

        now = time.monotonic()
        arm_due = (now - self._last_arm_rpc_time) >= self._arm_rpc_period_s()
        grip_due = (now - self._last_gripper_rpc_time) >= self._gripper_rpc_period_s()
        gripper_interval = self.config.gripper_interval or DEFAULT_TWO_FINGERS_GRIPPER_INTERVAL
        # 2 Hz while right_grip held: close on trigger press, open on trigger release.
        send_gripper = grip_due and not hold_grip

        attempted = False
        sent_ok = False
        if not hold_arm and arm_due:
            attempted = True
            grip_cmd_delta = self.config.gripper_cmd_delta if send_gripper else float("inf")
            sent_ok = self.arm.set_joint_positions_with_gripper(
                target_q,
                target_gripper,
                force=False,
                interval=gripper_interval,
                max_distance=gripper_max_d,
                min_distance=gripper_min_d,
                cmd_delta=grip_cmd_delta,
            )
            if sent_ok:
                self._last_arm_rpc_time = now
                if send_gripper:
                    self._last_gripper_rpc_time = now
                    self._last_gripper_input_mm = target_gripper
                self._was_streaming = True
        elif send_gripper:
            attempted = True
            sent_ok = self.arm.send_gripper_only(
                target_gripper,
                interval=gripper_interval,
                max_distance=gripper_max_d,
                min_distance=gripper_min_d,
                force=False,
                cmd_delta=self.config.gripper_cmd_delta,
            )
            if sent_ok:
                self._last_gripper_rpc_time = now
                self._last_gripper_input_mm = target_gripper

        if attempted and not sent_ok:
            raise DeviceNotConnectedError(
                f"{self} refused to send action (robot not connected or RPC channel unavailable)."
            )

        return sent

    @check_if_not_connected
    def disconnect(self) -> None:
        if self.arm is not None:
            if not self.config.passive_mode:
                try:
                    self.arm.exit_subloop1_if_active(timeout_ms=5000, blocking_exit=False)
                except Exception:
                    logger.exception("SubLoop1 exit failed during disconnect.")
                self.arm.disconnect()
            else:
                self.arm.disconnect_topic_feedback()
            self.arm = None

        for cam in self.cameras.values():
            cam.disconnect()

        reset_command_gate()
        logger.info("%s disconnected.", self)
