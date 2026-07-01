from dataclasses import dataclass
from typing import Optional

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("pico_ctl_tb6r5")
@dataclass
class PicoCtlTB6R5Config(TeleoperatorConfig):
    """PICO VR teleop for TB6-R5 via TB6R5TeleopController (direct RPC, VR episode buttons)."""

    robot_ip: str = "192.168.11.11"
    rpc_port: int = 5868
    robot_urdf_path: Optional[str] = None
    scale_factor: float = 1.5
    control_rate_hz: int = 50
    teleop_mode: str = "placo_ik"
    require_grip_to_send_commands: bool = True
    require_joystick_arm: bool = False
    gripper_max_d: float = 70.0
    gripper_min_d: float = 0.0
    two_fingers_gripper_interval: float = 25.0
    two_fingers_gripper_cmd_delta: float = 5.0
    safe_tcp_z_min_m: Optional[float] = 0.05
    safe_tcp_z_max_m: Optional[float] = 0.65
    zone_ratio: float = 0.0
    joint_vel: float = 6.0
    joint_acc: float = 3.0
    joint_dec: float = 3.0
    subloop1_immediate: bool = False
    visualize_placo: bool = False
