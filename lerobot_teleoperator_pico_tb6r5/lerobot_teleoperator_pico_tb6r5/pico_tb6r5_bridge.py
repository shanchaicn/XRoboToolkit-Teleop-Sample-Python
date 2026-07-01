"""Bridge TB6R5 PICO Placo IK logic to LeRobot teleop action dicts."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from xrobotoolkit_teleop.hardware.interface.tb6r5 import TB6R5Interface
from xrobotoolkit_teleop.hardware.tb6r5_command_gate import (
    set_grip_session_active,
    set_gripper_trigger_active,
)
from xrobotoolkit_teleop.hardware.tb6r5_teleop_controller import (
    DEFAULT_HOME_JOINT_DEG,
    DEFAULT_SCALE_FACTOR,
    DEFAULT_TB6R5_MANIPULATOR_CONFIG,
    DEFAULT_URDF_PATH,
    TB6R5_JOINT_NAMES,
    TB6R5TeleopController,
)

STATE_DIM = 7
JOINT_COUNT = 6


def _vector_to_action(q_rad: np.ndarray, gripper_mm: float) -> Dict[str, float]:
    q = np.asarray(q_rad, dtype=np.float64).ravel()
    if q.size < JOINT_COUNT:
        q = np.pad(q, (0, JOINT_COUNT - q.size))
    return {f"state_{i}": float(q[i]) for i in range(JOINT_COUNT)} | {"state_6": float(gripper_mm)}


class PicoTB6R5LeRobotBridge(TB6R5TeleopController):
    """PICO + Placo IK without hardware RPC (robot is driven by LeRobot)."""

    def __init__(
        self,
        robot_ip: str,
        robot_urdf_path: str = DEFAULT_URDF_PATH,
        scale_factor: float = DEFAULT_SCALE_FACTOR,
        control_rate_hz: int = 20,
        require_joystick_arm: bool = False,
        topic_wait_timeout_s: float = 5.0,
        **kwargs,
    ):
        self._feedback_ip = robot_ip
        self._feedback: Optional[TB6R5Interface] = None
        self._topic_wait_timeout_s = float(topic_wait_timeout_s)
        self._gripper_trigger_active = False
        self._gripper_trigger_activate = 0.5
        self._gripper_trigger_release = 0.2
        super().__init__(
            robot_urdf_path=robot_urdf_path,
            manipulator_config=DEFAULT_TB6R5_MANIPULATOR_CONFIG,
            robot_ip="none",
            scale_factor=scale_factor,
            visualize_placo=kwargs.pop("visualize_placo", False),
            control_rate_hz=control_rate_hz,
            enable_log_data=False,
            enable_camera=False,
            require_joystick_arm=require_joystick_arm,
            **kwargs,
        )

    def connect_feedback(self) -> None:
        self._feedback = TB6R5Interface(ip=self._feedback_ip, enable_topic=True)
        self._feedback.connect_topic_feedback(topic_wait_timeout_s=self._topic_wait_timeout_s)

    def _feedback_ready(self) -> bool:
        return self._feedback is not None and self._feedback.is_topic_healthy()

    def disconnect_feedback(self) -> None:
        if self._feedback is not None:
            self._feedback.disconnect_topic_feedback()
            self._feedback = None

    def _send_command(self) -> None:
        pass

    def _shutdown_robot(self) -> None:
        pass

    def _on_grip_session_end(self, src_name: str) -> None:
        del src_name

    def _update_robot_state(self) -> None:
        if self._feedback is not None and self._feedback.is_topic_healthy():
            q = self._feedback.get_joint_positions()
            self.placo_robot.state.q[self.joint_slice] = q[: len(TB6R5_JOINT_NAMES)]
            return
        if self.arm is not None:
            super()._update_robot_state()

    def _present_joint_positions(self) -> np.ndarray:
        if self._feedback is not None:
            return np.asarray(self._feedback.get_joint_positions(), dtype=np.float64).ravel()[:JOINT_COUNT]
        return self.placo_robot.state.q[self.joint_slice].copy()

    def _present_gripper_mm(self) -> float:
        if self._feedback is not None:
            grip = self._feedback.get_gripper_distance_mm()
            if grip is not None:
                return float(grip)
        return float(self._target_gripper_distance_mm)

    def _update_gripper_target_from_trigger(self):
        """right_trigger: released=open (max mm), pressed=closed (0 mm)."""
        if self.disable_gripper:
            return
        try:
            trigger = self.xr_client.get_key_value_by_name(self.gripper_trigger_name)
        except Exception:
            return
        if self._gripper_trigger_active:
            self._gripper_trigger_active = trigger > self._gripper_trigger_release
        else:
            self._gripper_trigger_active = trigger > self._gripper_trigger_activate
        # Always follow trigger value so releasing trigger targets open gripper.
        self._target_gripper_distance_mm = TB6R5Interface.gripper_distance_from_trigger(
            trigger,
            self.gripper_max_d,
            self.gripper_min_d,
        )

    def _is_gripper_trigger_active(self) -> bool:
        return self._gripper_trigger_active

    def get_lerobot_action(self) -> Dict[str, float]:
        feedback_ok = self._feedback_ready()
        if feedback_ok:
            self._update_robot_state()

        self._update_gripper_target()
        self._pre_ik_update()

        if feedback_ok:
            self._update_ik()
            if self.visualize_placo:
                self._update_placo_viz()

        active = any(self.active.get(name, False) for name in self.manipulator_config)
        set_grip_session_active(active)
        set_gripper_trigger_active(self._is_gripper_trigger_active())
        if feedback_ok and self._is_teleop_armed() and active:
            q_cmd = self.placo_robot.state.q[self.joint_slice].copy()
            grip_cmd = float(self._target_gripper_distance_mm)
        else:
            q_cmd = self._present_joint_positions()
            grip_cmd = self._present_gripper_mm()

        return _vector_to_action(q_cmd, grip_cmd)

    def go_home_placo(self) -> Dict[str, float]:
        home_q = np.deg2rad(DEFAULT_HOME_JOINT_DEG)
        self.placo_robot.state.q[self.joint_slice] = home_q
        self.placo_robot.update_kinematics()
        self.sync_end_effector_poses_to_placo_tasks()
        self._clear_teleop_session_state()
        return _vector_to_action(home_q, float(self.gripper_max_d))
