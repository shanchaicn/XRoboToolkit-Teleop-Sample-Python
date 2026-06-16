#!/usr/bin/env python3
"""
Run a trained LeRobot ACT policy on TB6-R5 hardware.

This script follows the official LeRobot inference flow:
  - load policy + pre/post processors from a pretrained checkpoint
  - build observation dict (state + RealSense camera images)
  - predict action
  - apply robot-side safety limits
  - send command to TB6-R5

Observation / action layout (must match the training dataset):
  - observation.state : [q0..q5, gripper_mm]  (7-D)
      * the 6 arm joints are read from the robot (rad)
      * gripper_mm is actual_pos feedback from TwoFingersGripperSerialYS (mm, 0=closed, max=open)
  - observation.images.realsense_0 / realsense_1 : RGB HWC uint8 (480x640x3)
  - action : [q0..q5, gripper_mm]  (7-D)
      * action[6] is commanded gripper distance in mm (0=closed, gripper_max_distance=open)
      * deployed via SubLoop1: JogAnyJ + MoveTwoFingersGripper in one RPC

On start and Ctrl+C exit the arm moves to --home-joint-deg via SubLoop1 (MoveAbsJ + open gripper).

Use --dry-run first to validate outputs before sending commands to the robot.
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

from xrobotoolkit_teleop.hardware.interface.tb6r5 import DEFAULT_GRIPPER_MAX_D, TB6R5Interface

# Source of truth lives in tb6r5_teleop_controller.DEFAULT_REALSENSE_SERIAL_DICT.
# Duplicated here to keep this inference script free of heavy controller imports.
DEFAULT_REALSENSE_SERIAL_DICT = {
    "realsense_0": "135522071053",
    "realsense_1": "327122073649",
}
# Fallback when gripper feedback is unavailable.
DEFAULT_GRIPPER_OBSERVATION_MM = 0.0
# Same as tb6r5_teleop_controller.DEFAULT_HOME_JOINT_DEG (degrees).
DEFAULT_HOME_JOINT_DEG = (15.0, -100.0, 90.0, -80.0, -90.0, -45.0)
DEFAULT_HOME_SETTLE_TIME_S = 3.0

# ANSI colors for gripper debug: green=open, red=close.
_GREEN = "\033[32m"
_RED = "\033[31m"
_BOLD_GREEN = "\033[1;32m"
_BOLD_RED = "\033[1;31m"
_RESET = "\033[0m"


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
    parser.add_argument("--device", default="cuda", help="Inference device, e.g. cuda/cpu")
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
    # Camera options
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


def _parse_camera_serials(spec: str | None) -> dict[str, str]:
    if not spec:
        return dict(DEFAULT_REALSENSE_SERIAL_DICT)
    out: dict[str, str] = {}
    for pair in spec.split(","):
        pair = pair.strip()
        if not pair:
            continue
        name, _, serial = pair.partition("=")
        if not name or not serial:
            raise ValueError(f"Invalid --camera-serials entry: '{pair}' (expected name=serial)")
        out[name.strip()] = serial.strip()
    return out


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
    )
    if settle_time_s > 0:
        time.sleep(settle_time_s)
    print("[ACT] Homing done.")


def _gripper_state_label_mm(distance_mm: float, max_distance: float) -> str:
    if distance_mm <= max_distance * 0.05:
        return "闭合"
    if distance_mm >= max_distance * 0.95:
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


def _gripper_desired_closed(
    gripper_norm: float,
    held_closed: bool | None,
    close_norm: float,
    open_norm: float,
) -> bool:
    """Asymmetric hysteresis: open→close when norm > close_norm; close→open when norm < open_norm."""
    if held_closed is None:
        return gripper_norm > close_norm

    if held_closed:
        if gripper_norm < open_norm:
            return False
        return True

    if gripper_norm > close_norm:
        return True
    return False


def _gripper_edge_min_steps(fps: float, min_interval_s: float) -> int:
    """Convert edge min interval (seconds) to control-loop steps using deployment fps."""
    return max(1, int(round(fps * min_interval_s)))


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
    gripper_max_distance: float,
    gripper_interval: float,
    chunk_step: int | None,
    chunk_size: int | None,
    legacy_mode: bool = False,
    desired_closed: bool | None = None,
    send_gripper: bool = False,
) -> None:
    state = _gripper_state_label_mm(gripper_cmd_mm, gripper_max_distance)
    if legacy_mode:
        cmd_status = f"legacy hysteresis send={send_gripper}"
    elif sent:
        cmd_status = f"SubLoop1 distance={gripper_cmd_mm:.2f}mm interval={gripper_interval:.1f}"
    else:
        cmd_status = "throttled, no SubLoop1 this step"
    chunk_info = ""
    if chunk_step is not None and chunk_size is not None:
        chunk_info = f" chunk={chunk_step + 1}/{chunk_size}"
    line = (
        f"[ACT][GRIPPER] raw={gripper_raw_mm:.2f}mm cmd={gripper_cmd_mm:.2f}mm "
        f"obs={gripper_obs_mm:.2f}mm state={state} sent={sent}{chunk_info} | {cmd_status}"
    )
    print(line)


def _print_gripper_config(
    gripper_max_distance: float,
    gripper_interval: float,
    gripper_cmd_delta: float,
    gripper_continuous: bool,
    chunk_size: int | None,
    n_action_steps: int | None,
    temporal_ensemble_coeff: float | None,
    refresh_policy_every_step: bool,
) -> None:
    mode = "continuous mm + SubLoop1" if gripper_continuous else "legacy hysteresis"
    print(
        f"{_RED}[ACT][GRIPPER] 配置: max_dist={gripper_max_distance:.1f}mm "
        f"interval={gripper_interval:.1f} cmd_delta={gripper_cmd_delta:.2f}mm mode={mode}{_RESET}"
    )
    print(
        f"{_RED}[ACT][GRIPPER] action[6]/state[6] 单位为 mm（0=闭合，{gripper_max_distance:.0f}=张开）；"
        f"obs 优先读 actual_pos 反馈。{_RESET}"
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


def _to_rgb_hwc_uint8(color: np.ndarray, height: int, width: int) -> np.ndarray:
    """RealSense color stream is already RGB HWC uint8; resize if needed."""
    arr = np.asarray(color)
    if arr.ndim == 3 and (arr.shape[0] != height or arr.shape[1] != width):
        import cv2

        arr = cv2.resize(arr, (width, height), interpolation=cv2.INTER_AREA)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _show_camera_rgb(images: dict[str, np.ndarray]) -> None:
    """Show RealSense RGB frames in OpenCV windows (RGB -> BGR for imshow)."""
    import cv2

    for name, rgb in images.items():
        if rgb is None:
            continue
        bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
        window = f"ACT RGB - {name}"
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(window, bgr)
    cv2.waitKey(1)


class _CameraStream:
    """Owns a RealSenseCameraInterface plus a background polling thread."""

    def __init__(self, serial_dict: dict[str, str], width: int, height: int, fps: int):
        from xrobotoolkit_teleop.hardware.interface.realsense import RealSenseCameraInterface

        self.serial_dict = serial_dict
        self.serial_to_name = {serial: name for name, serial in serial_dict.items()}
        self.width = width
        self.height = height
        self.cam = RealSenseCameraInterface(
            width=width,
            height=height,
            fps=fps,
            serial_numbers=list(serial_dict.values()),
            enable_depth=False,
            enable_compression=False,
        )
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.cam.start()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.cam.update_frames()
            except Exception as exc:  # keep streaming even on transient errors
                print(f"[ACT][camera] update_frames error: {exc}")
                time.sleep(0.02)

    def wait_ready(self, timeout_s: float = 10.0) -> None:
        deadline = time.time() + timeout_s
        needed = set(self.serial_dict.values())
        while time.time() < deadline:
            frames = self.cam.get_frames()
            if needed.issubset(set(frames.keys())):
                print("[ACT][camera] all cameras streaming")
                return
            time.sleep(0.1)
        print("[ACT][camera] WARNING: not all cameras produced frames before timeout")

    def get_images(self) -> dict[str, np.ndarray]:
        frames = self.cam.get_frames()
        out: dict[str, np.ndarray] = {}
        for serial, name in self.serial_to_name.items():
            fd = frames.get(serial)
            if fd is not None and fd.get("color") is not None:
                out[name] = _to_rgb_hwc_uint8(fd["color"], self.height, self.width)
        return out

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        try:
            self.cam.stop()
        except Exception:
            pass


def _load_policy_components(policy_path: str, dataset_root: str | None, repo_id: str | None, device: str):
    cfg = PreTrainedConfig.from_pretrained(policy_path)
    cfg.pretrained_path = policy_path
    cfg.device = device

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
    if not args.gripper_continuous:
        if not 0.0 <= args.gripper_close_norm <= 1.0:
            raise ValueError("--gripper-close-norm must be in [0, 1]")
        if not 0.0 <= args.gripper_open_norm <= 1.0:
            raise ValueError("--gripper-open-norm must be in [0, 1]")
        if args.gripper_close_norm >= args.gripper_open_norm:
            raise ValueError(
                f"--gripper-close-norm ({args.gripper_close_norm}) must be < "
                f"--gripper-open-norm ({args.gripper_open_norm}) for asymmetric hysteresis"
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
    serial_dict = _parse_camera_serials(args.camera_serials)
    camera_names = sorted(serial_dict.keys())
    cam_stream: _CameraStream | None = None
    black = np.zeros((args.camera_height, args.camera_width, 3), dtype=np.uint8)
    if not args.no_camera:
        cam_stream = _CameraStream(serial_dict, args.camera_width, args.camera_height, args.camera_fps)
        cam_stream.start()
        cam_stream.wait_ready()
    else:
        print("[ACT] --no-camera: feeding black frames (predictions will be meaningless)")

    arm = None
    home_joint_deg = tuple(args.home_joint_deg)
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
            )
    else:
        print("[ACT] Dry-run mode: no command will be sent")

    dt = 1.0 / args.fps
    last_print = 0.0
    last_images: dict[str, np.ndarray] = {name: black.copy() for name in camera_names}
    held_gripper_closed: bool | None = False if (arm is not None and not args.no_home_on_start) else None
    control_step = 0
    last_gripper_rpc_step = 0 if (arm is not None and not args.no_home_on_start) else -gripper_edge_min_steps

    chunk_size = getattr(policy.config, "chunk_size", None)
    n_action_steps = getattr(policy.config, "n_action_steps", None)
    _print_gripper_config(
        args.gripper_max_distance,
        args.gripper_interval,
        args.gripper_cmd_delta,
        args.gripper_continuous,
        chunk_size,
        n_action_steps,
        args.temporal_ensemble_coeff,
        args.refresh_policy_every_step,
    )
    if args.show_camera and cam_stream is not None:
        print("[ACT] RGB preview enabled (windows: ACT RGB - realsense_0/1). Use --no-show-camera to disable.")
    print("[ACT] Inference loop started. Press Ctrl+C to stop.")
    try:
        while True:
            start_t = time.time()

            if arm is not None:
                q_current = np.asarray(arm.get_joint_positions(), dtype=np.float32)[:6]
            else:
                q_current = np.zeros(6, dtype=np.float32)

            gripper_obs_mm = _resolve_gripper_observation_mm(arm, args.gripper_observation_constant)
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
                if args.show_camera and last_images:
                    _show_camera_rgb(last_images)
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
            gripper_cmd_mm = float(np.clip(gripper_raw_mm, 0.0, args.gripper_max_distance))
            send_gripper = False
            gripper_cmd_dist: float | None = None

            if args.gripper_continuous:
                gripper_cmd_dist = gripper_cmd_mm
            else:
                gripper_norm = gripper_cmd_mm / args.gripper_max_distance
                desired_closed = _gripper_desired_closed(
                    gripper_norm,
                    held_gripper_closed,
                    args.gripper_close_norm,
                    args.gripper_open_norm,
                )
                gripper_edge = held_gripper_closed is None or desired_closed != held_gripper_closed
                gripper_steps_since_rpc = control_step - last_gripper_rpc_step
                if gripper_edge:
                    gripper_cmd_dist = float(args.gripper_max_distance) if desired_closed else 0.0
                    if gripper_steps_since_rpc >= gripper_edge_min_steps:
                        send_gripper = True
                        held_gripper_closed = desired_closed
                        last_gripper_rpc_step = control_step

            chunk_step, chunk_size = _act_chunk_info(policy)

            if now - last_print >= args.print_every:
                print(
                    "[ACT] "
                    f"q_cur={np.round(q_current, 3)} "
                    f"q_tgt={np.round(q_target, 3)} "
                    f"q_cmd={np.round(q_cmd, 3)}"
                )
                _print_gripper_status(
                    gripper_raw_mm=gripper_raw_mm,
                    gripper_cmd_mm=gripper_cmd_dist if gripper_cmd_dist is not None else gripper_cmd_mm,
                    gripper_obs_mm=gripper_obs_mm,
                    sent=sent,
                    gripper_max_distance=args.gripper_max_distance,
                    gripper_interval=args.gripper_interval,
                    chunk_step=chunk_step,
                    chunk_size=chunk_size,
                    legacy_mode=not args.gripper_continuous,
                    desired_closed=held_gripper_closed if not args.gripper_continuous else None,
                    send_gripper=send_gripper,
                )
                last_print = now

            if arm is not None:
                if args.gripper_continuous:
                    sent = arm.set_joint_positions_with_gripper(
                        q_cmd,
                        gripper_cmd_mm,
                        force=True,
                        interval=args.gripper_interval,
                        max_distance=args.gripper_max_distance,
                        cmd_delta=args.gripper_cmd_delta,
                    )
                else:
                    arm.set_joint_positions(q_cmd, force=True)
                    if send_gripper and gripper_cmd_dist is not None:
                        sent = arm.move_two_fingers_gripper(
                            distance=gripper_cmd_dist,
                            interval=args.gripper_interval,
                            max_distance=args.gripper_max_distance,
                        )

            elapsed = time.time() - start_t
            if elapsed < dt:
                time.sleep(dt - elapsed)
            control_step += 1
    except KeyboardInterrupt:
        print("\n[ACT] Stopped by user.")
    finally:
        if args.show_camera and cam_stream is not None:
            import cv2

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
                    )
                arm.disable()
            except Exception:
                pass
            print("[ACT] Robot disabled.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
