"""LeRobot teleoperator: PICO VR + TB6R5TeleopController direct hardware control."""

from __future__ import annotations

import logging
from functools import cached_property
from typing import Any

try:
    from lerobot.processor import RobotAction
except ImportError:  # lerobot >= 0.5
    from lerobot.types import RobotAction

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from xrobotoolkit_teleop.hardware.tb6r5_teleop_controller import DEFAULT_URDF_PATH

from .config_pico_ctl_tb6r5 import PicoCtlTB6R5Config
from .pico_ctl_tb6r5_controller import LeRobotPicoCtlTB6R5Controller

logger = logging.getLogger(__name__)

STATE_DIM = 7
STATE_KEYS = tuple(f"state_{i}" for i in range(STATE_DIM))


class PicoCtlTB6R5(Teleoperator):
    """TB6-R5 teleop using TB6R5TeleopController RPC path; episode control via PICO A/B."""

    config_class = PicoCtlTB6R5Config
    name = "pico_ctl_tb6r5"

    def __init__(self, config: PicoCtlTB6R5Config):
        super().__init__(config)
        self.config = config
        self._controller: LeRobotPicoCtlTB6R5Controller | None = None

    def _ensure_controller(self) -> LeRobotPicoCtlTB6R5Controller:
        if self._controller is None:
            urdf_path = self.config.robot_urdf_path or DEFAULT_URDF_PATH
            self._controller = LeRobotPicoCtlTB6R5Controller(
                robot_urdf_path=urdf_path,
                robot_ip=self.config.robot_ip,
                rpc_port=self.config.rpc_port,
                teleop_mode=self.config.teleop_mode,
                scale_factor=self.config.scale_factor,
                control_rate_hz=self.config.control_rate_hz,
                visualize_placo=self.config.visualize_placo,
                require_grip_to_send_commands=self.config.require_grip_to_send_commands,
                require_joystick_arm=self.config.require_joystick_arm,
                gripper_max_d=self.config.gripper_max_d,
                two_fingers_gripper_interval=self.config.two_fingers_gripper_interval,
                two_fingers_gripper_cmd_delta=self.config.two_fingers_gripper_cmd_delta,
                safe_tcp_z_min_m=self.config.safe_tcp_z_min_m,
                safe_tcp_z_max_m=self.config.safe_tcp_z_max_m,
                zone_ratio=self.config.zone_ratio,
                joint_vel=self.config.joint_vel,
                joint_acc=self.config.joint_acc,
                joint_dec=self.config.joint_dec,
                subloop1_immediate=self.config.subloop1_immediate,
            )
        return self._controller

    @cached_property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(STATE_KEYS, float)

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        ctrl = self._controller
        return ctrl is not None and ctrl.arm is not None and ctrl.arm.is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        self._ensure_controller().start_background_loop()
        logger.info("%s connected (TB6R5TeleopController @ %s).", self, self.config.robot_ip)

    def configure(self) -> None:
        pass

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        return self._ensure_controller().get_action_dict()

    def poll_record_events(self, events: dict) -> None:
        """Merge PICO A/B episode events into lerobot-record events dict."""
        if self._controller is None:
            return
        pending = self._controller.consume_record_events()
        if pending["exit_early"]:
            events["exit_early"] = True
        if pending["rerecord_episode"]:
            events["rerecord_episode"] = True
        if pending["stop_recording"]:
            events["stop_recording"] = True

    def get_teleop_events(self) -> dict[str, Any]:
        if self._controller is None:
            return {
                "terminate_episode": False,
                "rerecord_episode": False,
                "success": False,
                "is_intervention": False,
            }
        return self._controller.get_teleop_events()

    def send_feedback(self, feedback: dict) -> None:
        del feedback

    @check_if_not_connected
    def disconnect(self) -> None:
        if self._controller is not None:
            self._controller.stop_background_loop()
            self._controller = None
        logger.info("%s disconnected.", self)
