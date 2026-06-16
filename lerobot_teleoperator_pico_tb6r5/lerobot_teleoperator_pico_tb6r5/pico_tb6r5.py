"""LeRobot teleoperator: PICO VR -> TB6-R5 joint targets."""

from __future__ import annotations

import logging
from functools import cached_property

try:
    from lerobot.processor import RobotAction
except ImportError:  # lerobot >= 0.5
    from lerobot.types import RobotAction

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError

from xrobotoolkit_teleop.hardware.tb6r5_command_gate import reset_command_gate
from xrobotoolkit_teleop.hardware.tb6r5_teleop_controller import DEFAULT_URDF_PATH

from .config_pico_tb6r5 import PicoTB6R5Config
from .pico_tb6r5_bridge import STATE_DIM, PicoTB6R5LeRobotBridge

logger = logging.getLogger(__name__)

STATE_KEYS = tuple(f"state_{i}" for i in range(STATE_DIM))


class PicoTB6R5(Teleoperator):
    config_class = PicoTB6R5Config
    name = "pico_tb6r5"

    def __init__(self, config: PicoTB6R5Config):
        super().__init__(config)
        self.config = config
        self._bridge: PicoTB6R5LeRobotBridge | None = None
        self._connected = False

    def _ensure_bridge(self) -> PicoTB6R5LeRobotBridge:
        if self._bridge is None:
            urdf_path = self.config.robot_urdf_path or DEFAULT_URDF_PATH
            self._bridge = PicoTB6R5LeRobotBridge(
                robot_ip=self.config.robot_ip,
                robot_urdf_path=urdf_path,
                scale_factor=self.config.scale_factor,
                control_rate_hz=self.config.control_rate_hz,
                require_joystick_arm=self.config.require_joystick_arm,
                gripper_max_d=self.config.gripper_max_d,
                safe_tcp_z_min_m=self.config.safe_tcp_z_min_m,
                safe_tcp_z_max_m=self.config.safe_tcp_z_max_m,
                topic_wait_timeout_s=self.config.topic_wait_timeout_s,
            )
        return self._bridge

    @cached_property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(STATE_KEYS, float)

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        if not self._connected or self._bridge is None:
            return False
        return self._bridge._feedback_ready()

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        bridge = self._ensure_bridge()
        try:
            bridge.connect_feedback()
        except ConnectionError as exc:
            raise DeviceNotConnectedError(str(exc)) from exc
        self._connected = True
        logger.info("%s connected (PICO + topic feedback @ %s).", self, self.config.robot_ip)

    def configure(self) -> None:
        pass

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        return self._ensure_bridge().get_lerobot_action()

    def send_feedback(self, feedback: dict) -> None:
        del feedback

    @check_if_not_connected
    def disconnect(self) -> None:
        if self._bridge is not None:
            self._bridge.disconnect_feedback()
        reset_command_gate()
        self._connected = False
        logger.info("%s disconnected.", self)
