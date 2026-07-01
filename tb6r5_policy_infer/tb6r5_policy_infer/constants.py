"""Default configuration values for TB6-R5 ACT inference."""

from xrobotoolkit_teleop.hardware.interface.tb6r5 import (
    DEFAULT_GRIPPER_MAX_D,
    DEFAULT_TWO_FINGERS_GRIPPER_INTERVAL,
)
from xrobotoolkit_teleop.hardware.tb6r5_teleop_controller import (
    DEFAULT_GRIPPER_RPC_RATE_HZ,
    DEFAULT_HOME_JOINT_DEG,
    DEFAULT_REALSENSE_SERIAL_DICT,
    DEFAULT_TWO_FINGERS_GRIPPER_CMD_DELTA,
)

# Match scripts/hardware/record_tb6r5_lerobot.sh (GRIPPER_MIN_D=30).
DEFAULT_GRIPPER_MIN_D = 30.0

DEFAULT_GRIPPER_OBSERVATION_MM = 0.0
DEFAULT_HOME_SETTLE_TIME_S = 3.0
# Match record_tb6r5_lerobot.sh CONTROL_RATE_HZ / LOG_FREQ.
DEFAULT_CONTROL_FPS = 30.0
# Arm/gripper SubLoop1 RPC rates for deployment (30 Hz arm, 2 Hz gripper).
DEFAULT_ARM_RPC_RATE_HZ = 30.0

# ANSI colors for gripper debug: green=open, red=close.
GREEN = "\033[32m"
RED = "\033[31m"
BOLD_GREEN = "\033[1;32m"
BOLD_RED = "\033[1;31m"
RESET = "\033[0m"

__all__ = [
    "DEFAULT_ARM_RPC_RATE_HZ",
    "DEFAULT_CONTROL_FPS",
    "DEFAULT_GRIPPER_MAX_D",
    "DEFAULT_GRIPPER_MIN_D",
    "DEFAULT_GRIPPER_OBSERVATION_MM",
    "DEFAULT_GRIPPER_RPC_RATE_HZ",
    "DEFAULT_HOME_JOINT_DEG",
    "DEFAULT_HOME_SETTLE_TIME_S",
    "DEFAULT_REALSENSE_SERIAL_DICT",
    "DEFAULT_TWO_FINGERS_GRIPPER_CMD_DELTA",
    "DEFAULT_TWO_FINGERS_GRIPPER_INTERVAL",
    "GREEN",
    "RED",
    "BOLD_GREEN",
    "BOLD_RED",
    "RESET",
]
