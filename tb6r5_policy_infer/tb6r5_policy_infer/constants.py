"""Default configuration values for TB6-R5 ACT inference."""

from xrobotoolkit_teleop.hardware.interface.tb6r5 import DEFAULT_GRIPPER_MAX_D

# Source of truth lives in tb6r5_teleop_controller.DEFAULT_REALSENSE_SERIAL_DICT.
DEFAULT_REALSENSE_SERIAL_DICT = {
    "realsense_0": "135522071053",
    "realsense_1": "327122073649",
}

DEFAULT_GRIPPER_OBSERVATION_MM = 0.0
DEFAULT_HOME_JOINT_DEG = (15.0, -100.0, 90.0, -80.0, -90.0, -45.0)
DEFAULT_HOME_SETTLE_TIME_S = 3.0

# ANSI colors for gripper debug: green=open, red=close.
GREEN = "\033[32m"
RED = "\033[31m"
BOLD_GREEN = "\033[1;32m"
BOLD_RED = "\033[1;31m"
RESET = "\033[0m"

__all__ = [
    "DEFAULT_GRIPPER_MAX_D",
    "DEFAULT_REALSENSE_SERIAL_DICT",
    "DEFAULT_GRIPPER_OBSERVATION_MM",
    "DEFAULT_HOME_JOINT_DEG",
    "DEFAULT_HOME_SETTLE_TIME_S",
    "GREEN",
    "RED",
    "BOLD_GREEN",
    "BOLD_RED",
    "RESET",
]
