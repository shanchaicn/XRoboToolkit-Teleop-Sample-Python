from dataclasses import dataclass, field
from typing import Optional

from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("tb6r5")
@RobotConfig.register_subclass("tb5r6")
@dataclass
class TB6R5Config(RobotConfig):
    """Configuration for the TB6-R5 arm + YS gripper."""

    robot_ip: str = "192.168.11.11"
    rpc_port: int = 5868
    enable_topic: bool = True
    topic_wait_timeout_s: float = 5.0
    # When True, skip RPC; teleoperator (e.g. pico_ctl_tb6r5) drives the arm. Robot provides obs/cameras only.
    passive_mode: bool = False

    zone_ratio: float = 0.00
    joint_vel: float = 6.0
    joint_acc: float = 3.0
    joint_dec: float = 3.0
    subloop1_immediate: bool = False

    gripper_max_d: float = 70.0
    gripper_interval: float = 25.0
    # Min change in gripper command input (mm) vs last sent target before issuing RPC.
    gripper_cmd_delta: float = 5.0

    # RPC rate limits (match teleop_tb6r5_hardware.py defaults).
    arm_rpc_rate_hz: float = 50.0
    gripper_rpc_rate_hz: float = 2.0
    # Skip arm RPC when max joint delta vs feedback is below this (rad).
    arm_cmd_eps_rad: float = 1e-3

    # Per-step joint delta clamp (rad); None disables clamping.
    max_joint_step_rad: Optional[float] = 0.03

    cameras: dict[str, CameraConfig] = field(default_factory=dict)
