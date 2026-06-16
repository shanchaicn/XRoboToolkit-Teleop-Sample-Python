"""Hardware ACT inference control loop for TB6-R5."""

from __future__ import annotations

import time

import numpy as np
import torch

from .lerobot_compat import predict_action
from xrobotoolkit_teleop.hardware.interface.tb6r5 import TB6R5Interface

from .camera import CameraStream, show_camera_rgb
from .constants import BOLD_GREEN, DEFAULT_REALSENSE_SERIAL_DICT, RESET
from .gripper import (
    clamp_joint_step,
    gripper_desired_closed,
    gripper_edge_min_steps,
    print_gripper_config,
    print_gripper_status,
    resolve_gripper_observation_mm,
)
from .policy import act_chunk_info, apply_act_inference_overrides, load_policy_components


def go_home(
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
        f"{BOLD_GREEN}[ACT][GRIPPER] 复位：SubLoop1 MoveAbsJ + "
        f"MoveTwoFingersGripper(distance={gripper_max_distance:.1f}mm, "
        f"interval={gripper_interval:.1f}){RESET}"
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


def run_inference(args) -> int:
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

    policy, preprocessor, postprocessor = load_policy_components(
        policy_path=args.policy_path,
        dataset_root=args.dataset_root,
        repo_id=args.repo_id,
        device=args.device,
    )
    apply_act_inference_overrides(
        policy,
        n_action_steps=args.n_action_steps,
        temporal_ensemble_coeff=args.temporal_ensemble_coeff,
        refresh_policy_every_step=args.refresh_policy_every_step,
    )

    from .camera import parse_camera_serials

    serial_dict = parse_camera_serials(args.camera_serials, DEFAULT_REALSENSE_SERIAL_DICT)
    camera_names = sorted(serial_dict.keys())
    cam_stream: CameraStream | None = None
    black = np.zeros((args.camera_height, args.camera_width, 3), dtype=np.uint8)
    if not args.no_camera:
        cam_stream = CameraStream(serial_dict, args.camera_width, args.camera_height, args.camera_fps)
        cam_stream.start()
        cam_stream.wait_ready()
    else:
        print("[ACT] --no-camera: feeding black frames (predictions will be meaningless)")

    arm = None
    home_joint_deg = tuple(args.home_joint_deg)
    gripper_edge_min_steps_val = gripper_edge_min_steps(args.fps, args.gripper_edge_min_interval)
    if not args.dry_run:
        arm = TB6R5Interface(
            ip=args.robot_ip, rpc_port=args.rpc_port, joint_count=6, rpc_cmd_rate_hz=max(args.fps, 20)
        )
        arm.connect()
        print(f"[ACT] Connected to TB6-R5 at {args.robot_ip}:{args.rpc_port}")
        if not args.no_home_on_start:
            go_home(
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
    last_gripper_rpc_step = 0 if (arm is not None and not args.no_home_on_start) else -gripper_edge_min_steps_val

    chunk_size = getattr(policy.config, "chunk_size", None)
    n_action_steps = getattr(policy.config, "n_action_steps", None)
    print_gripper_config(
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

            gripper_obs_mm = resolve_gripper_observation_mm(arm, args.gripper_observation_constant)
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
                    show_camera_rgb(last_images)
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
            q_cmd = clamp_joint_step(q_target, q_current, args.joint_step_max_rad)

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
                desired_closed = gripper_desired_closed(
                    gripper_norm,
                    held_gripper_closed,
                    args.gripper_close_norm,
                    args.gripper_open_norm,
                )
                gripper_edge = held_gripper_closed is None or desired_closed != held_gripper_closed
                gripper_steps_since_rpc = control_step - last_gripper_rpc_step
                if gripper_edge:
                    gripper_cmd_dist = float(args.gripper_max_distance) if desired_closed else 0.0
                    if gripper_steps_since_rpc >= gripper_edge_min_steps_val:
                        send_gripper = True
                        held_gripper_closed = desired_closed
                        last_gripper_rpc_step = control_step

            chunk_step, chunk_size = act_chunk_info(policy)

            if now - last_print >= args.print_every:
                print(
                    "[ACT] "
                    f"q_cur={np.round(q_current, 3)} "
                    f"q_tgt={np.round(q_target, 3)} "
                    f"q_cmd={np.round(q_cmd, 3)}"
                )
                print_gripper_status(
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
                    go_home(
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
