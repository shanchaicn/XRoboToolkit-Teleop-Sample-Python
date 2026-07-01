#!/usr/bin/env python3
"""
TB6-R5 LeRobot ACT hardware inference
=====================================

Runs a trained ACT checkpoint on the real robot (or ``--dry-run`` without RPC).

Equivalent packaged CLI: ``tb6r5-policy-infer`` (see ``tb6r5_policy_infer/README.md``).

Full interface reference: ``docs/ACT_Policy_Infer_Interface.md``

Observation / action contract (must match training dataset)
-----------------------------------------------------------

**observation.state** — 7-D float32 ``[q0..q5, gripper]``

+-------+------+----------------------------------------------------------+
| Index | Unit | Source at inference                                      |
+=======+======+==========================================================+
| 0–5   | rad  | ``TB6R5Interface.get_joint_positions()`` (Topic feedback)|
| 6     | mm   | Gripper ``actual_pos`` feedback, or ``--gripper-observation-constant`` |
+-------+------+----------------------------------------------------------+

**observation.images.*** — RGB HWC ``uint8``, default shape ``(480, 640, 3)``

+----------------------------+-----------------------------------------------+
| Key                        | Typical source                                |
+============================+===============================================+
| ``observation.images.realsense_0`` | Camera logical name #0 (see Camera modes) |
| ``observation.images.realsense_1`` | Camera logical name #1                  |
+----------------------------+-----------------------------------------------+

Logical names (left column) **must match the dataset**; only the physical
device binding changes between RealSense SN and V4L2 ``/dev/video*``.

**action** — 7-D float32 ``[q0..q5, gripper_mm]`` from ``predict_action()``:

+-------+------+----------------------------------------------------------+
| Index | Unit | Deployed to robot                                        |
+=======+======+==========================================================+
| 0–5   | rad  | Clamped by ``--joint-step-max-rad``, sent as ``JogAnyJ`` |
| 6     | mm   | 0 = closed, ``--gripper-max-distance`` = open (default 70)|
+-------+------+----------------------------------------------------------+

Default gripper mode (``--gripper-continuous``): continuous mm via SubLoop1
``JogAnyJ + MoveTwoFingersGripper``. Legacy binary hysteresis:
``--no-gripper-continuous``.

Robot command path
------------------

``TB6R5Interface`` → vendor ``rpc.so`` → TCP ``--robot-ip:--rpc-port`` (default 5868).

Requires ARM/x86 SDK binaries under ``dependencies/`` (not in git). Verify::

    python scripts/hardware/verify_tb6r5_sdk.py --robot-ip 192.168.11.11

Camera modes
------------

**1. RealSense (default)** — ``pyrealsense2`` via ``RealSenseCameraInterface``::

    --camera-serials 'realsense_0=135522071053,realsense_1=327122073649'

Omit ``--camera-serials`` to use ``DEFAULT_REALSENSE_SERIAL_DICT`` in this file.

**2. V4L2 /dev/video*** — OpenCV ``VideoCapture``, no ``pyrealsense2``::

    --camera-devices 'realsense_0=/dev/video0,realsense_1=/dev/video4'
    # or numeric index: realsense_0=0,realsense_1=4

When ``--camera-devices`` is set, ``--camera-serials`` is ignored.
Find RGB nodes (RealSense exposes many ``/dev/video*``; not all are color)::

    v4l2-ctl --list-devices
    ls -l /dev/video*

**3. HTTP URL** — poll JPEG/PNG via GET (no ``pyrealsense2``, no local device)::

    --camera-urls 'realsense_0=http://192.168.2.42:8888/RsCameraSensor/0/0/color,realsense_1=http://192.168.2.42:8888/RsCameraSensor/1/0/color'

When ``--camera-urls`` is set, ``--camera-serials`` and ``--camera-devices`` are ignored.
Each URL must return image bytes (JPEG/PNG). Logical names must match the training dataset.

**4. No camera** — black frames (debug only)::

    --no-camera

Shared sizing: ``--camera-width`` (640), ``--camera-height`` (480), ``--camera-fps`` (30).

Quick start
-----------

Dry-run (no RPC, validate policy + cameras)::

    python scripts/hardware/policy_infer_tb6r5_act.py \\
      --robot-ip 192.168.11.11 \\
      --policy-path model/act/080000/pretrained_model \\
      --dry-run

TER30 with V4L2 (bypass libusb / pyrealsense2 issues)::

    python scripts/hardware/policy_infer_tb6r5_act.py \\
      --robot-ip 192.168.11.11 \\
      --policy-path model/act/080000/pretrained_model \\
      --camera-devices 'realsense_0=/dev/video0,realsense_1=/dev/video4' \\
      --device cpu \\
      --dry-run

HTTP camera server (e.g. RsCameraSensor on port 8888)::

    python scripts/hardware/policy_infer_tb6r5_act.py \\
      --robot-ip 192.168.11.11 \\
      --policy-path model/act/080000/pretrained_model \\
      --camera-urls 'realsense_0=http://192.168.2.42:8888/RsCameraSensor/0/0/color,realsense_1=http://192.168.2.42:8888/RsCameraSensor/1/0/color' \\
      --dry-run

Real robot (remove ``--dry-run``; ensure e-stop and workspace clearance)::

    python scripts/hardware/policy_infer_tb6r5_act.py \\
      --robot-ip 192.168.11.11 \\
      --policy-path model/act/080000/pretrained_model \\
      --fps 10 --joint-step-max-rad 0.03

On start and Ctrl+C exit the arm homing uses ``--home-joint-deg`` via SubLoop1
(``MoveAbsJ`` + open gripper) unless ``--no-home-on-start`` / ``--no-home-on-exit``.
"""

from __future__ import annotations

import argparse
import threading
import time

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.control_utils import predict_action

from xrobotoolkit_teleop.common.camera_streams import (
    DEFAULT_REALSENSE_SERIAL_DICT,
    create_camera_stream as _create_camera_stream,
    parse_camera_serials as _parse_camera_serials,
)
from xrobotoolkit_teleop.hardware.interface.tb6r5 import (
    DEFAULT_GRIPPER_MAX_D,
    DEFAULT_TWO_FINGERS_GRIPPER_INTERVAL,
    TB6R5Interface,
)
from xrobotoolkit_teleop.hardware.tb6r5_teleop_controller import (
    DEFAULT_GRIPPER_RPC_RATE_HZ,
    DEFAULT_HOME_JOINT_DEG,
    DEFAULT_TWO_FINGERS_GRIPPER_CMD_DELTA,
)

# Match scripts/hardware/record_tb6r5_lerobot.sh (GRIPPER_MIN_D=30).
DEFAULT_GRIPPER_MIN_D = 30.0
DEFAULT_CONTROL_FPS = 30.0
DEFAULT_ARM_RPC_RATE_HZ = 30.0
# Fallback when gripper feedback is unavailable.
DEFAULT_GRIPPER_OBSERVATION_MM = 0.0
DEFAULT_HOME_SETTLE_TIME_S = 3.0

# ANSI colors for gripper debug: green=open, red=close.
_GREEN = "\033[32m"
_RED = "\033[31m"
_BOLD_GREEN = "\033[1;32m"
_BOLD_RED = "\033[1;31m"
_RESET = "\033[0m"


def _cuda_usable() -> bool:
    """True only if CUDA init and a tiny allocation succeed."""
    if not torch.cuda.is_available():
        return False
    try:
        torch.zeros(1, device="cuda")
        return True
    except RuntimeError:
        return False


def _resolve_inference_device(device: str) -> str:
    """Map auto to cuda/cpu; fall back when CUDA is broken or unavailable."""
    requested = (device or "auto").strip().lower()
    if requested == "auto":
        if _cuda_usable():
            return "cuda"
        print("[ACT] CUDA unavailable, using CPU (pass --device cpu to silence; --device cuda when GPU is ready)")
        return "cpu"
    if requested.startswith("cuda") and not _cuda_usable():
        print(f"[ACT] WARNING: --device {device} requested but CUDA unavailable, using CPU")
        return "cpu"
    return device


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
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
    # Camera options
    parser.add_argument(
        "--camera-serials",
        default=None,
        help=(
            "RealSense mode: comma-separated name=serial pairs, e.g. "
            "'realsense_0=135522071053,realsense_1=327122073649'. "
            "Ignored when --camera-devices or --camera-urls is set. Defaults to DEFAULT_REALSENSE_SERIAL_DICT."
        ),
    )
    parser.add_argument(
        "--camera-devices",
        default=None,
        help=(
            "V4L2 mode: comma-separated name=device pairs, e.g. "
            "'realsense_0=/dev/video0,realsense_1=/dev/video4' or 'realsense_0=0,realsense_1=4'. "
            "Ignored when --camera-urls is set. OpenCV VideoCapture."
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


def _clip_gripper_mm(distance_mm: float, min_distance: float, max_distance: float) -> float:
    lo = min(float(min_distance), float(max_distance))
    hi = max(float(min_distance), float(max_distance))
    return float(np.clip(float(distance_mm), lo, hi))


def _rpc_strides(control_fps: float, arm_rpc_rate_hz: float, gripper_rpc_rate_hz: float) -> tuple[int, int]:
    control_fps = max(float(control_fps), 0.1)
    arm_rpc_rate_hz = max(float(arm_rpc_rate_hz), 0.1)
    gripper_rpc_rate_hz = max(float(gripper_rpc_rate_hz), 0.1)
    arm_stride = max(1, round(control_fps / arm_rpc_rate_hz))
    grip_stride = max(1, round(control_fps / gripper_rpc_rate_hz))
    return arm_stride, grip_stride


def _on_rpc_tick(control_step: int, stride: int) -> bool:
    return control_step % max(int(stride), 1) == 0


def _clamp_joint_step(q_target: np.ndarray, q_current: np.ndarray, max_step: float) -> np.ndarray:
    dq = np.clip(q_target - q_current, -max_step, max_step)
    return q_current + dq


def _go_home(
    arm: TB6R5Interface,
    home_joint_deg: tuple[float, ...],
    settle_time_s: float,
    *,
    gripper_interval: float,
    gripper_max_distance: float,
    gripper_min_distance: float,
) -> None:
    home_q = np.deg2rad(np.asarray(home_joint_deg, dtype=float))
    print(f"[ACT] Homing to {tuple(home_joint_deg)} deg ...")
    print(
        f"{_BOLD_GREEN}[ACT][GRIPPER] 复位：SubLoop1 MoveAbsJ + "
        f"MoveTwoFingersGripper(distance={gripper_max_distance:.1f}mm, "
        f"interval={gripper_interval:.1f}){_RESET}"
    )
    arm.go_home(
        home_q,
        gripper_distance=gripper_max_distance,
        interval=gripper_interval,
        max_distance=gripper_max_distance,
        min_distance=gripper_min_distance,
    )
    if settle_time_s > 0:
        time.sleep(settle_time_s)
    print("[ACT] Homing done.")


def _gripper_state_label_mm(distance_mm: float, min_distance: float, max_distance: float) -> str:
    lo = min(float(min_distance), float(max_distance))
    hi = max(float(min_distance), float(max_distance))
    if distance_mm <= lo + 0.05 * (hi - lo):
        return "闭合"
    if distance_mm >= hi - 0.05 * (hi - lo):
        return "张开"
    return "中间"


def _resolve_gripper_observation_mm(arm: TB6R5Interface | None, constant_mm: float | None) -> float:
    if constant_mm is not None:
        return float(constant_mm)
    if arm is not None:
        feedback = arm.get_gripper_distance_mm()
        if feedback is not None:
            return float(feedback)
    return DEFAULT_GRIPPER_OBSERVATION_MM


def _validate_gripper_hysteresis_mm(
    close_mm: float,
    open_mm: float,
    min_distance: float,
    max_distance: float,
) -> None:
    lo = min(float(min_distance), float(max_distance))
    hi = max(float(min_distance), float(max_distance))
    if not (lo <= close_mm <= hi):
        raise ValueError(f"--gripper-close-mm ({close_mm}) must be within [{lo}, {hi}]")
    if not (lo <= open_mm <= hi):
        raise ValueError(f"--gripper-open-mm ({open_mm}) must be within [{lo}, {hi}]")
    if close_mm >= open_mm:
        raise ValueError(
            f"--gripper-close-mm ({close_mm}) must be < --gripper-open-mm ({open_mm}) "
            "(deadband: mm <= close → closed, mm >= open → open)"
        )


def _gripper_desired_open_mm(
    gripper_mm: float,
    held_open: bool | None,
    close_mm: float,
    open_mm: float,
) -> bool:
    """Asymmetric hysteresis on mm distance (min_distance=closed, max_distance=open)."""
    if held_open is None:
        if gripper_mm >= open_mm:
            return True
        if gripper_mm <= close_mm:
            return False
        return gripper_mm >= (close_mm + open_mm) / 2.0

    if held_open:
        if gripper_mm <= close_mm:
            return False
        return True

    if gripper_mm >= open_mm:
        return True
    return False


def _gripper_edge_min_steps(fps: float, min_interval_s: float) -> int:
    """Convert edge min interval (seconds) to control-loop steps using deployment fps."""
    return max(1, int(round(fps * min_interval_s)))


def _latched_gripper_mm(
    held_open: bool | None,
    *,
    min_distance: float,
    max_distance: float,
) -> float | None:
    if held_open is None:
        return None
    return float(max_distance) if held_open else float(min_distance)


def _update_legacy_gripper_state(
    *,
    gripper_mm: float,
    held_open: bool | None,
    pending_mm: float | None,
    close_mm: float,
    open_mm: float,
    control_step: int,
    last_edge_step: int,
    edge_min_steps: int,
    min_distance: float,
    max_distance: float,
) -> tuple[bool | None, float | None, int, bool]:
    desired_open = _gripper_desired_open_mm(gripper_mm, held_open, close_mm, open_mm)
    edge = held_open is None or desired_open != held_open
    edge_accepted = False
    if edge and control_step - last_edge_step >= edge_min_steps:
        held_open = desired_open
        pending_mm = float(max_distance) if desired_open else float(min_distance)
        last_edge_step = control_step
        edge_accepted = True
    return held_open, pending_mm, last_edge_step, edge_accepted


def _apply_act_inference_overrides(
    policy,
    *,
    n_action_steps: int | None,
    temporal_ensemble_coeff: float | None,
    refresh_policy_every_step: bool,
) -> None:
    """Apply deployment-time ACT inference settings (no retraining required)."""
    chunk_size = int(policy.config.chunk_size)
    ckpt_n_action = int(policy.config.n_action_steps)

    if temporal_ensemble_coeff is not None:
        if refresh_policy_every_step:
            raise ValueError(
                "--temporal-ensemble-coeff and --refresh-policy-every-step are incompatible "
                "(reset every step destroys the temporal ensemble buffer)."
            )
        if temporal_ensemble_coeff < 0:
            raise ValueError("--temporal-ensemble-coeff must be >= 0")
        from lerobot.policies.act.modeling_act import ACTTemporalEnsembler

        policy.config.temporal_ensemble_coeff = float(temporal_ensemble_coeff)
        policy.config.n_action_steps = 1
        policy.temporal_ensembler = ACTTemporalEnsembler(float(temporal_ensemble_coeff), chunk_size)
        policy.reset()
        print(
            f"[ACT] Temporal Ensemble ON: coeff={temporal_ensemble_coeff:g}, "
            f"chunk_size={chunk_size}, every-step inference"
        )
        return

    new_n_action = ckpt_n_action if n_action_steps is None else int(n_action_steps)
    if not 1 <= new_n_action <= chunk_size:
        raise ValueError(f"--n-action-steps must be in [1, {chunk_size}], got {new_n_action}")

    if new_n_action != ckpt_n_action:
        policy.config.n_action_steps = new_n_action
        policy.reset()
        print(
            f"[ACT] n_action_steps override: {ckpt_n_action} -> {new_n_action} "
            f"(chunk_size={chunk_size}, re-infer every {new_n_action} control steps)"
        )
    elif refresh_policy_every_step:
        print(f"[ACT] Action queue chunk_size={chunk_size}, n_action_steps={new_n_action}, refresh every step")


def _act_chunk_info(policy) -> tuple[int | None, int | None]:
    """Return (step_index_in_queue, queue_len) for ACT action queue, if available."""
    if getattr(policy.config, "temporal_ensemble_coeff", None) is not None:
        return None, None
    queue_len = getattr(policy.config, "n_action_steps", None)
    queue = getattr(policy, "_action_queue", None)
    if queue_len is None or queue is None:
        return None, queue_len
    remaining = len(queue)
    step_index = max(queue_len - remaining - 1, 0)
    return step_index, queue_len


def _print_gripper_status(
    *,
    gripper_raw_mm: float,
    gripper_cmd_mm: float,
    gripper_obs_mm: float,
    sent: bool,
    gripper_min_distance: float,
    gripper_max_distance: float,
    gripper_interval: float,
    chunk_step: int | None,
    chunk_size: int | None,
    legacy_mode: bool = False,
    desired_open: bool | None = None,
    send_gripper: bool = False,
    gripper_subloop: str | None = None,
    pending_gripper_mm: float | None = None,
    edge_accepted: bool = False,
) -> None:
    state = _gripper_state_label_mm(gripper_cmd_mm, gripper_min_distance, gripper_max_distance)
    if legacy_mode:
        if gripper_subloop == "NotRunExecute":
            pending_info = f" pending={pending_gripper_mm:.0f}mm" if pending_gripper_mm is not None else ""
            cmd_status = (
                f"legacy arm JogAnyJ, gripper=NotRunExecute"
                f"{pending_info} (edge={edge_accepted})"
            )
        elif gripper_subloop:
            cmd_status = f"legacy gripper {gripper_subloop} interval={gripper_interval:.1f}"
        else:
            cmd_status = f"legacy hysteresis edge={edge_accepted}"
    elif gripper_subloop == "NotRunExecute":
        cmd_status = "SubLoop1 arm JogAnyJ, gripper=NotRunExecute"
    elif gripper_subloop:
        cmd_status = f"SubLoop1 gripper {gripper_subloop} interval={gripper_interval:.1f}"
    elif sent:
        cmd_status = "SubLoop1 sent"
    else:
        cmd_status = "no arm RPC this step"
    chunk_info = ""
    if chunk_step is not None and chunk_size is not None:
        chunk_info = f" chunk={chunk_step + 1}/{chunk_size}"
    latched_info = ""
    if legacy_mode and desired_open is not None:
        latched_info = f" latched={'open' if desired_open else 'closed'}"
    line = (
        f"[ACT][GRIPPER] action[6]={gripper_raw_mm:.2f}mm latched_cmd={gripper_cmd_mm:.2f}mm "
        f"obs={gripper_obs_mm:.2f}mm state={state}{latched_info} sent={sent}{chunk_info} | {cmd_status}"
    )
    print(line)


def _print_gripper_config(
    gripper_max_distance: float,
    gripper_min_distance: float,
    gripper_interval: float,
    gripper_cmd_delta: float,
    gripper_continuous: bool,
    chunk_size: int | None,
    n_action_steps: int | None,
    temporal_ensemble_coeff: float | None,
    refresh_policy_every_step: bool,
    arm_rpc_rate_hz: float,
    gripper_rpc_rate_hz: float,
    control_fps: float,
    gripper_close_mm: float | None = None,
    gripper_open_mm: float | None = None,
) -> None:
    mode = (
        "continuous mm + SubLoop1"
        if gripper_continuous
        else "legacy hysteresis (arm SubLoop1 JogAnyJ + binary gripper)"
    )
    arm_stride, grip_stride = _rpc_strides(control_fps, arm_rpc_rate_hz, gripper_rpc_rate_hz)
    print(
        f"{_RED}[ACT][GRIPPER] 配置: min_dist={gripper_min_distance:.1f}mm "
        f"max_dist={gripper_max_distance:.1f}mm "
        f"interval={gripper_interval:.1f} cmd_delta={gripper_cmd_delta:.2f}mm mode={mode}{_RESET}"
    )
    print(
        f"{_RED}[ACT][GRIPPER] RPC 分频: control={control_fps:.0f}Hz "
        f"arm={arm_rpc_rate_hz:.0f}Hz (stride={arm_stride}) "
        f"gripper={gripper_rpc_rate_hz:.0f}Hz (stride={grip_stride}){_RESET}"
    )
    print(
        f"{_RED}[ACT][GRIPPER] action[6]/state[6] 单位为 mm"
        f"（{gripper_min_distance:.0f}=闭合，{gripper_max_distance:.0f}=张开）；"
        f"与 record_tb6r5_lerobot.sh 一致。{_RESET}"
    )
    if not gripper_continuous and gripper_close_mm is not None and gripper_open_mm is not None:
        print(
            f"{_RED}[ACT][GRIPPER] 滞回阈值: mm<={gripper_close_mm:.1f}→{gripper_min_distance:.0f}mm(闭合), "
            f"mm>={gripper_open_mm:.1f}→{gripper_max_distance:.0f}mm(张开){_RESET}"
        )
    if temporal_ensemble_coeff is not None:
        print(
            f"{_RED}[ACT][ACTION] Temporal Ensemble coeff={temporal_ensemble_coeff:g}, "
            f"chunk_size={chunk_size}：每步推理并融合重叠 chunk 预测。{_RESET}"
        )
    elif refresh_policy_every_step:
        print(
            f"{_RED}[ACT][ACTION] 每步 policy.reset() + 重推理（action queue 不累积；"
            f"chunk_size={chunk_size}）。{_RESET}"
        )
    elif n_action_steps is not None:
        print(
            f"{_RED}[ACT][ACTION] Action queue: chunk_size={chunk_size}, n_action_steps={n_action_steps}，"
            f"每 {n_action_steps} 步重推理。可调 --n-action-steps 或 --temporal-ensemble-coeff 0.01。{_RESET}"
        )


_PREVIEW_WINDOWS: set[str] = set()


def _show_camera_rgb(images: dict[str, np.ndarray]) -> None:
    """Show RealSense RGB frames in OpenCV windows (RGB -> BGR for imshow)."""
    import cv2

    for name, rgb in images.items():
        if rgb is None:
            continue
        bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
        window = f"ACT RGB - {name}"
        if window not in _PREVIEW_WINDOWS:
            cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
            _PREVIEW_WINDOWS.add(window)
        cv2.imshow(window, bgr)
    cv2.waitKey(1)


class _CameraPreview:
    def __init__(self, cam_stream, fps: float = 30.0):
        self._cam_stream = cam_stream
        self._dt = 1.0 / max(fps, 1.0)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            start_t = time.time()
            imgs = self._cam_stream.get_images()
            if imgs:
                _show_camera_rgb(imgs)
            elapsed = time.time() - start_t
            if elapsed < self._dt:
                time.sleep(self._dt - elapsed)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


def _load_policy_components(policy_path: str, dataset_root: str | None, repo_id: str | None, device: str):
    cfg = PreTrainedConfig.from_pretrained(policy_path)
    cfg.pretrained_path = policy_path
    cfg.device = _resolve_inference_device(device)

    dataset_stats = None
    if dataset_root and repo_id:
        # Optional: pull normalization stats from the training dataset metadata.
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

        ds_meta = LeRobotDatasetMetadata(repo_id=repo_id, root=dataset_root)
        dataset_stats = ds_meta.stats
        from lerobot.policies.factory import make_policy

        policy = make_policy(cfg=cfg, ds_meta=ds_meta)
    else:
        # Self-contained: the checkpoint already carries baked-in normalization stats.
        policy_cls = get_policy_class(cfg.type)
        policy = policy_cls.from_pretrained(policy_path, config=cfg)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=policy_path,
        dataset_stats=dataset_stats,
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )
    return policy, preprocessor, postprocessor


def main() -> int:
    args = _build_parser().parse_args()

    if args.fps <= 0:
        raise ValueError("--fps must be > 0")
    if args.gripper_max_distance <= 0:
        raise ValueError("--gripper-max-distance must be > 0")
    if args.gripper_min_distance < 0:
        raise ValueError("--gripper-min-distance must be >= 0")
    if args.gripper_min_distance > args.gripper_max_distance:
        raise ValueError("--gripper-min-distance must be <= --gripper-max-distance")
    if args.arm_rpc_rate_hz <= 0 or args.gripper_rpc_rate_hz <= 0:
        raise ValueError("--arm-rpc-rate-hz and --gripper-rpc-rate-hz must be > 0")
    if not args.gripper_continuous:
        _validate_gripper_hysteresis_mm(
            args.gripper_close_mm,
            args.gripper_open_mm,
            args.gripper_min_distance,
            args.gripper_max_distance,
        )
        if args.gripper_edge_min_interval < 0:
            raise ValueError("--gripper-edge-min-interval must be >= 0")

    policy, preprocessor, postprocessor = _load_policy_components(
        policy_path=args.policy_path,
        dataset_root=args.dataset_root,
        repo_id=args.repo_id,
        device=args.device,
    )
    _apply_act_inference_overrides(
        policy,
        n_action_steps=args.n_action_steps,
        temporal_ensemble_coeff=args.temporal_ensemble_coeff,
        refresh_policy_every_step=args.refresh_policy_every_step,
    )

    # Camera setup
    cam_stream = None
    black = np.zeros((args.camera_height, args.camera_width, 3), dtype=np.uint8)
    if not args.no_camera:
        cam_stream, camera_names, _ = _create_camera_stream(
            camera_urls=args.camera_urls,
            camera_devices=args.camera_devices,
            camera_serials=args.camera_serials,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
            log_prefix="[ACT][camera]",
        )
        cam_stream.start()
        cam_stream.wait_ready()
    else:
        camera_names = sorted(_parse_camera_serials(args.camera_serials).keys())
        print("[ACT] --no-camera: feeding black frames (predictions will be meaningless)")

    arm = None
    home_joint_deg = tuple(args.home_joint_deg)
    arm_stride, grip_stride = _rpc_strides(args.fps, args.arm_rpc_rate_hz, args.gripper_rpc_rate_hz)
    gripper_edge_min_steps = _gripper_edge_min_steps(args.fps, args.gripper_edge_min_interval)
    if not args.dry_run:
        arm = TB6R5Interface(ip=args.robot_ip, rpc_port=args.rpc_port, joint_count=6, rpc_cmd_rate_hz=max(args.fps, 20))
        arm.connect()
        print(f"[ACT] Connected to TB6-R5 at {args.robot_ip}:{args.rpc_port}")
        if not args.no_home_on_start:
            _go_home(
                arm,
                home_joint_deg,
                args.home_settle_time,
                gripper_interval=args.gripper_interval,
                gripper_max_distance=args.gripper_max_distance,
                gripper_min_distance=args.gripper_min_distance,
            )
    else:
        print("[ACT] Dry-run mode: no command will be sent")

    dt = 1.0 / args.fps
    last_print = 0.0
    last_images: dict[str, np.ndarray] = {name: black.copy() for name in camera_names}
    held_gripper_open: bool | None = True if (arm is not None and not args.no_home_on_start) else None
    pending_gripper_mm: float | None = None
    control_step = 0
    last_gripper_rpc_step = 0 if (arm is not None and not args.no_home_on_start) else -gripper_edge_min_steps

    chunk_size = getattr(policy.config, "chunk_size", None)
    n_action_steps = getattr(policy.config, "n_action_steps", None)
    _print_gripper_config(
        args.gripper_max_distance,
        args.gripper_min_distance,
        args.gripper_interval,
        args.gripper_cmd_delta,
        args.gripper_continuous,
        chunk_size,
        n_action_steps,
        args.temporal_ensemble_coeff,
        args.refresh_policy_every_step,
        args.arm_rpc_rate_hz,
        args.gripper_rpc_rate_hz,
        args.fps,
        gripper_close_mm=None if args.gripper_continuous else args.gripper_close_mm,
        gripper_open_mm=None if args.gripper_continuous else args.gripper_open_mm,
    )
    if args.show_camera and cam_stream is not None:
        print("[ACT] RGB preview enabled (windows: ACT RGB - realsense_0/1). Use --no-show-camera to disable.")
    cam_preview = None
    if args.show_camera and cam_stream is not None:
        cam_preview = _CameraPreview(cam_stream, fps=float(args.camera_preview_fps))
        cam_preview.start()
    print("[ACT] Inference loop started. Press Ctrl+C to stop.")
    print(
        f"[ACT] Control loop {args.fps:.0f} Hz | "
        f"arm RPC {args.arm_rpc_rate_hz:.0f} Hz | gripper RPC {args.gripper_rpc_rate_hz:.0f} Hz"
    )
    try:
        while True:
            start_t = time.time()

            if arm is not None:
                q_current = np.asarray(arm.get_joint_positions(), dtype=np.float32)[:6]
            else:
                q_current = np.zeros(6, dtype=np.float32)

            gripper_obs_mm = _clip_gripper_mm(
                _resolve_gripper_observation_mm(arm, args.gripper_observation_constant),
                args.gripper_min_distance,
                args.gripper_max_distance,
            )
            observation = {
                "observation.state": np.concatenate(
                    [q_current, np.array([gripper_obs_mm], dtype=np.float32)],
                    axis=0,
                )
            }

            if cam_stream is not None:
                imgs = cam_stream.get_images()
                for name in camera_names:
                    if name in imgs:
                        last_images[name] = imgs[name]
            for name in camera_names:
                observation[f"observation.images.{name}"] = last_images[name]

            if args.refresh_policy_every_step:
                policy.reset()

            action_tensor = predict_action(
                observation=observation,
                policy=policy,
                device=torch.device(policy.config.device),
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=False,
                task=args.task,
                robot_type="tb6r5",
            )
            action = action_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)
            if action.shape[0] < 7:
                raise ValueError(f"Expected action dim >= 7, got {action.shape}")

            q_target = action[:6]
            q_cmd = _clamp_joint_step(q_target, q_current, args.joint_step_max_rad)

            gripper_raw_mm = float(action[6])
            now = time.time()
            sent = False
            gripper_cmd_mm = _clip_gripper_mm(
                gripper_raw_mm,
                args.gripper_min_distance,
                args.gripper_max_distance,
            )
            send_gripper = False
            edge_accepted = False

            if args.gripper_continuous:
                gripper_cmd_dist = gripper_cmd_mm
            else:
                held_gripper_open, pending_gripper_mm, last_gripper_rpc_step, edge_accepted = (
                    _update_legacy_gripper_state(
                        gripper_mm=gripper_cmd_mm,
                        held_open=held_gripper_open,
                        pending_mm=pending_gripper_mm,
                        close_mm=args.gripper_close_mm,
                        open_mm=args.gripper_open_mm,
                        control_step=control_step,
                        last_edge_step=last_gripper_rpc_step,
                        edge_min_steps=gripper_edge_min_steps,
                        min_distance=args.gripper_min_distance,
                        max_distance=args.gripper_max_distance,
                    )
                )
                send_gripper = edge_accepted

            chunk_step, chunk_size = _act_chunk_info(policy)

            gripper_subloop: str | None = None
            if arm is not None and _on_rpc_tick(control_step, arm_stride):
                on_gripper_rpc_tick = _on_rpc_tick(control_step, grip_stride)
                gripper_cmd_delta = args.gripper_cmd_delta if on_gripper_rpc_tick else float("inf")
                if args.gripper_continuous:
                    gripper_will_send = arm._should_send_gripper(
                        gripper_cmd_mm,
                        cmd_delta=gripper_cmd_delta,
                        max_distance=args.gripper_max_distance,
                        min_distance=args.gripper_min_distance,
                    )
                    sent = arm.set_joint_positions_with_gripper(
                        q_cmd,
                        gripper_cmd_mm,
                        interval=args.gripper_interval,
                        max_distance=args.gripper_max_distance,
                        min_distance=args.gripper_min_distance,
                        cmd_delta=gripper_cmd_delta,
                    )
                    gripper_subloop = f"distance={gripper_cmd_mm:.2f}mm" if gripper_will_send else "NotRunExecute"
                else:
                    latched_mm = _latched_gripper_mm(
                        held_gripper_open,
                        min_distance=args.gripper_min_distance,
                        max_distance=args.gripper_max_distance,
                    )
                    gripper_mm_for_rpc = (
                        pending_gripper_mm
                        if pending_gripper_mm is not None
                        else (latched_mm if latched_mm is not None else float(args.gripper_max_distance))
                    )
                    legacy_gripper_cmd_delta = (
                        args.gripper_cmd_delta
                        if (on_gripper_rpc_tick and pending_gripper_mm is not None)
                        else float("inf")
                    )
                    gripper_will_send = arm._should_send_gripper(
                        gripper_mm_for_rpc,
                        cmd_delta=legacy_gripper_cmd_delta,
                        max_distance=args.gripper_max_distance,
                        min_distance=args.gripper_min_distance,
                    )
                    sent = arm.set_joint_positions_with_gripper(
                        q_cmd,
                        gripper_mm_for_rpc,
                        interval=args.gripper_interval,
                        max_distance=args.gripper_max_distance,
                        min_distance=args.gripper_min_distance,
                        cmd_delta=legacy_gripper_cmd_delta,
                    )
                    if gripper_will_send and pending_gripper_mm is not None:
                        pending_gripper_mm = None
                    gripper_subloop = (
                        f"distance={gripper_mm_for_rpc:.2f}mm" if gripper_will_send else "NotRunExecute"
                    )

            if now - last_print >= args.print_every:
                print(
                    "[ACT] "
                    f"q_cur={np.round(q_current, 3)} "
                    f"q_tgt={np.round(q_target, 3)} "
                    f"q_cmd={np.round(q_cmd, 3)} "
                    f"action[6]={gripper_raw_mm:.2f}mm"
                )
                latched_display_mm = (
                    _latched_gripper_mm(
                        held_gripper_open,
                        min_distance=args.gripper_min_distance,
                        max_distance=args.gripper_max_distance,
                    )
                    if not args.gripper_continuous
                    else None
                )
                _print_gripper_status(
                    gripper_raw_mm=gripper_raw_mm,
                    gripper_cmd_mm=latched_display_mm if latched_display_mm is not None else gripper_cmd_mm,
                    gripper_obs_mm=gripper_obs_mm,
                    sent=sent,
                    gripper_min_distance=args.gripper_min_distance,
                    gripper_max_distance=args.gripper_max_distance,
                    gripper_interval=args.gripper_interval,
                    chunk_step=chunk_step,
                    chunk_size=chunk_size,
                    legacy_mode=not args.gripper_continuous,
                    desired_open=held_gripper_open if not args.gripper_continuous else None,
                    send_gripper=send_gripper,
                    gripper_subloop=gripper_subloop,
                    pending_gripper_mm=pending_gripper_mm,
                    edge_accepted=edge_accepted,
                )
                last_print = now

            control_step += 1
            elapsed = time.time() - start_t
            if elapsed < dt:
                time.sleep(dt - elapsed)
    except KeyboardInterrupt:
        print("\n[ACT] Stopped by user.")
    finally:
        if cam_preview is not None:
            cam_preview.stop()
        if args.show_camera and cam_stream is not None:
            import cv2

            _PREVIEW_WINDOWS.clear()
            cv2.destroyAllWindows()
        if cam_stream is not None:
            cam_stream.stop()
        if arm is not None:
            try:
                if not args.no_home_on_exit:
                    _go_home(
                        arm,
                        home_joint_deg,
                        args.home_settle_time,
                        gripper_interval=args.gripper_interval,
                        gripper_max_distance=args.gripper_max_distance,
                        gripper_min_distance=args.gripper_min_distance,
                    )
                arm.disable()
            except Exception:
                pass
            print("[ACT] Robot disabled.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
