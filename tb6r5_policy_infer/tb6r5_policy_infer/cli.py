"""CLI entry point for TB6-R5 ACT hardware inference."""

from __future__ import annotations

import argparse

from .constants import DEFAULT_GRIPPER_MAX_D, DEFAULT_HOME_JOINT_DEG, DEFAULT_HOME_SETTLE_TIME_S
from .runner import run_inference

_DOC = """
Run a trained LeRobot ACT policy on TB6-R5 hardware.

This follows the official LeRobot inference flow:
  - load policy + pre/post processors from a pretrained checkpoint
  - build observation dict (state + RealSense camera images)
  - predict action
  - apply robot-side safety limits
  - send command to TB6-R5

Observation / action layout (must match the training dataset):
  - observation.state : [q0..q5, gripper_mm]  (7-D)
  - observation.images.realsense_0 / realsense_1 : RGB HWC uint8 (480x640x3)
  - action : [q0..q5, gripper_mm]  (7-D)

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
    parser.add_argument("--fps", type=float, default=20.0, help="Control loop frequency")
    parser.add_argument("--joint-step-max-rad", type=float, default=0.03, help="Per-step joint delta clamp (rad)")
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
        help="Gripper max open distance in mm (must match data collection, default 70)",
    )
    parser.add_argument("--gripper-interval", type=float, default=5.0, help="MoveTwoFingersGripper interval")
    parser.add_argument(
        "--gripper-cmd-delta",
        type=float,
        default=0.5,
        help="Minimum gripper distance change (mm) before re-sending SubLoop1 RPC.",
    )
    parser.add_argument(
        "--gripper-continuous",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Map action[6] directly to mm distance (default). Use --no-gripper-continuous for legacy hysteresis.",
    )
    parser.add_argument(
        "--gripper-close-norm",
        type=float,
        default=0.1,
        help="Legacy hysteresis: open→close when norm > this (only with --no-gripper-continuous).",
    )
    parser.add_argument(
        "--gripper-open-norm",
        type=float,
        default=0.5,
        help="Legacy hysteresis: close→open when norm < this (only with --no-gripper-continuous).",
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
            "Comma-separated name=serial pairs, e.g. "
            "'realsense_0=135522071053,realsense_1=327122073649'. "
            "Defaults to the teleop serial dict."
        ),
    )
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="Skip RealSense capture and feed black frames (pipeline test only; outputs are meaningless).",
    )
    parser.add_argument(
        "--show-camera",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Display RealSense RGB preview windows during inference (default: on).",
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
    args = build_parser().parse_args()
    return run_inference(args)


def cli() -> None:
    raise SystemExit(main())
