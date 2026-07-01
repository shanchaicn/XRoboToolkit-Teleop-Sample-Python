"""CLI entry point for TB6-R5 ACT hardware inference."""

from __future__ import annotations

import argparse

from .constants import (
    DEFAULT_ARM_RPC_RATE_HZ,
    DEFAULT_CONTROL_FPS,
    DEFAULT_GRIPPER_MAX_D,
    DEFAULT_GRIPPER_MIN_D,
    DEFAULT_GRIPPER_RPC_RATE_HZ,
    DEFAULT_HOME_JOINT_DEG,
    DEFAULT_HOME_SETTLE_TIME_S,
    DEFAULT_JOG_ANY_JOINT_ACC,
    DEFAULT_JOG_ANY_JOINT_DEC,
    DEFAULT_JOG_ANY_JOINT_VEL,
    DEFAULT_TWO_FINGERS_GRIPPER_CMD_DELTA,
    DEFAULT_TWO_FINGERS_GRIPPER_INTERVAL,
    DEFAULT_ZONE_RATIO,
)

_DOC = """
Run a trained LeRobot ACT policy on TB6-R5 hardware.

This follows the official LeRobot inference flow:
  - load policy + pre/post processors from a pretrained checkpoint
  - build observation dict (state + camera images: RealSense SN, V4L2, or HTTP URL)
  - predict action
  - apply robot-side safety limits
  - send command to TB6-R5

Observation / action layout (must match the training dataset):
  - observation.state : [q0..q5, gripper]  (7-D; gripper in mm or [0,1] with --gripper-normalized)
  - observation.images.realsense_0 / realsense_1 : RGB HWC uint8 (480x640x3)
  - action : [q0..q5, gripper]  (7-D; same unit as training)

On start and Ctrl+C exit the arm moves to --home-joint-deg via SubLoop1 (MoveAbsJ + open gripper).

Use --dry-run first to validate outputs before sending commands to the robot.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=_DOC)
    parser.add_argument("--robot-ip", required=True, help="TB6-R5 robot IP")
    parser.add_argument("--rpc-port", type=int, default=5868, help="TB6 RPC port")
    parser.add_argument("--policy-path", required=True, help="Path (or HF repo) of ACT pretrained checkpoint")
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Optional LeRobot dataset root (only needed if loading stats from the dataset)",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Optional LeRobot repo_id (only used together with --dataset-root)",
    )
    parser.add_argument("--task", default="tb6r5 teleoperation", help="Task string passed to LeRobot policy")
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto (default), cuda, or cpu. Falls back to CPU if CUDA is unavailable.",
    )
    parser.add_argument("--fps", type=float, default=DEFAULT_CONTROL_FPS, help="Control loop frequency (record script: 30)")
    parser.add_argument("--joint-step-max-rad", type=float, default=0.03, help="Per-step joint delta clamp (rad)")
    parser.add_argument(
        "--joint-vel",
        type=float,
        default=DEFAULT_JOG_ANY_JOINT_VEL,
        help="JogAnyJ joint_vel in RPC (default: 6.0, same as teleop_tb6r5_hardware.py)",
    )
    parser.add_argument(
        "--joint-acc",
        type=float,
        default=DEFAULT_JOG_ANY_JOINT_ACC,
        help="JogAnyJ joint_acc in RPC (default: 3.0)",
    )
    parser.add_argument(
        "--joint-dec",
        type=float,
        default=DEFAULT_JOG_ANY_JOINT_DEC,
        help="JogAnyJ joint_dec in RPC (default: 3.0)",
    )
    parser.add_argument(
        "--zone-ratio",
        type=float,
        default=DEFAULT_ZONE_RATIO,
        help="JogAnyJ zone_ratio in RPC (default: 0.05)",
    )
    parser.add_argument(
        "--arm-rpc-rate-hz",
        type=float,
        default=DEFAULT_ARM_RPC_RATE_HZ,
        help="Arm SubLoop1 RPC rate in Hz (default 30, match --fps)",
    )
    parser.add_argument(
        "--gripper-rpc-rate-hz",
        type=float,
        default=DEFAULT_GRIPPER_RPC_RATE_HZ,
        help="Gripper SubLoop1 update rate in Hz (default 2)",
    )
    parser.add_argument(
        "--gripper-observation-constant",
        type=float,
        default=None,
        help="Force constant mm value for observation.state[6] (default: read actual_pos feedback).",
    )
    parser.add_argument(
        "--gripper-max-distance",
        type=float,
        default=DEFAULT_GRIPPER_MAX_D,
        help="Gripper max open distance in mm (record script GRIPPER_MAX_D, default 70)",
    )
    parser.add_argument(
        "--gripper-min-distance",
        type=float,
        default=DEFAULT_GRIPPER_MIN_D,
        help="Gripper min distance in mm (record script GRIPPER_MIN_D, default 30)",
    )
    parser.add_argument(
        "--gripper-normalized",
        action="store_true",
        help=(
            "Training used gripper in [0,1]: obs.state[6]=feedback_mm/max_distance, "
            "action[6]=policy_norm*max_distance before sending to robot."
        ),
    )
    parser.add_argument(
        "--gripper-interval",
        type=float,
        default=DEFAULT_TWO_FINGERS_GRIPPER_INTERVAL,
        help="MoveTwoFingersGripper interval",
    )
    parser.add_argument(
        "--gripper-cmd-delta",
        type=float,
        default=DEFAULT_TWO_FINGERS_GRIPPER_CMD_DELTA,
        help="Minimum gripper distance change (mm) before re-sending SubLoop1 RPC.",
    )
    parser.add_argument(
        "--gripper-continuous",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Map action[6] directly to mm distance (default). Use --no-gripper-continuous for legacy hysteresis.",
    )
    parser.add_argument(
        "--gripper-close-mm",
        type=float,
        default=40.0,
        help="Legacy hysteresis: latched open→closed when action mm <= this (only with --no-gripper-continuous).",
    )
    parser.add_argument(
        "--gripper-open-mm",
        type=float,
        default=50.0,
        help="Legacy hysteresis: latched closed→open when action mm >= this (only with --no-gripper-continuous).",
    )
    parser.add_argument(
        "--gripper-edge-min-interval",
        type=float,
        default=2.0,
        help="Legacy hysteresis: min seconds between gripper RPC edges.",
    )
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Override ACT n_action_steps at inference (1..chunk_size). "
            "Smaller values re-infer more often without retraining. Default: use checkpoint value."
        ),
    )
    parser.add_argument(
        "--temporal-ensemble-coeff",
        type=float,
        default=None,
        metavar="COEFF",
        help=(
            "Enable ACT temporal ensembling (original ACT paper uses 0.01). "
            "Every control step runs inference and fuses overlapping chunk predictions. "
            "Implies n_action_steps=1; do not combine with --refresh-policy-every-step."
        ),
    )
    parser.add_argument(
        "--refresh-policy-every-step",
        action="store_true",
        help="Call policy.reset() every control step (disables ACT action queue; more reactive, slower).",
    )
    parser.add_argument(
        "--camera-serials",
        default=None,
        help=(
            "RealSense mode: comma-separated name=serial pairs, e.g. "
            "'realsense_0=135522071053,realsense_1=327122073649'. "
            "Ignored when --camera-devices or --camera-urls is set. Defaults to the teleop serial dict."
        ),
    )
    parser.add_argument(
        "--camera-devices",
        default=None,
        help=(
            "V4L2 mode: comma-separated name=device pairs, e.g. "
            "'realsense_0=/dev/video0,realsense_1=/dev/video2' or 'realsense_0=0,realsense_1=2'. "
            "Ignored when --camera-urls is set. Uses OpenCV VideoCapture (no pyrealsense2)."
        ),
    )
    parser.add_argument(
        "--camera-urls",
        default=None,
        help=(
            "HTTP mode: comma-separated name=url pairs, e.g. "
            "'realsense_0=http://192.168.2.42:8888/RsCameraSensor/0/0/color,"
            "realsense_1=http://192.168.2.42:8888/RsCameraSensor/1/0/color'. "
            "Each URL is polled via GET; response must be JPEG/PNG image bytes."
        ),
    )
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument(
        "--camera-preview-fps",
        type=int,
        default=30,
        help="OpenCV preview refresh rate (independent of control loop; default: 30).",
    )
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="Skip camera capture and feed black frames (pipeline test only; outputs are meaningless).",
    )
    parser.add_argument(
        "--show-camera",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Display RGB preview windows during inference (default: on).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Predict and print actions without sending to robot")
    parser.add_argument(
        "--home-joint-deg",
        type=float,
        nargs=6,
        default=DEFAULT_HOME_JOINT_DEG,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="Home joint angles in degrees; used on start and Ctrl+C exit (default: teleop home pose)",
    )
    parser.add_argument(
        "--home-settle-time",
        type=float,
        default=DEFAULT_HOME_SETTLE_TIME_S,
        help="Seconds to wait after MoveAbsJ homing",
    )
    parser.add_argument(
        "--no-home-on-start",
        action="store_true",
        help="Skip homing when the script starts",
    )
    parser.add_argument(
        "--no-home-on-exit",
        action="store_true",
        help="Skip homing when the script exits (Ctrl+C)",
    )
    parser.add_argument(
        "--print-every",
        type=float,
        default=0.5,
        help="Minimum print interval (seconds) for action debug lines",
    )
    return parser


def main() -> int:
    from .runner import run_inference

    args = build_parser().parse_args()
    return run_inference(args)


def cli() -> None:
    raise SystemExit(main())
