from dataclasses import dataclass
from typing import Optional

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("pico_tb6r5")
@dataclass
class PicoTB6R5Config(TeleoperatorConfig):
    """PICO VR teleop for TB6-R5 (Placo IK + YS gripper trigger)."""

    robot_ip: str = "192.168.11.11"
    topic_wait_timeout_s: float = 5.0
    robot_urdf_path: Optional[str] = None
    scale_factor: float = 1.5
    # Align with --dataset.fps; 50 Hz matches teleop_tb6r5_hardware.py and reduces latency.
    control_rate_hz: int = 50
    require_joystick_arm: bool = False
    gripper_max_d: float = 70.0
    gripper_min_d: float = 0.0
    safe_tcp_z_min_m: Optional[float] = 0.05
    safe_tcp_z_max_m: Optional[float] = 0.65
