"""Gripper observation and command helpers for TB6-R5 ACT inference."""

from __future__ import annotations

import numpy as np

from xrobotoolkit_teleop.hardware.interface.tb6r5 import TB6R5Interface

from .constants import DEFAULT_GRIPPER_OBSERVATION_MM, RED, RESET


def clamp_joint_step(q_target: np.ndarray, q_current: np.ndarray, max_step: float) -> np.ndarray:
    dq = np.clip(q_target - q_current, -max_step, max_step)
    return q_current + dq


def gripper_state_label_mm(distance_mm: float, max_distance: float) -> str:
    if distance_mm <= max_distance * 0.05:
        return "闭合"
    if distance_mm >= max_distance * 0.95:
        return "张开"
    return "中间"


def resolve_gripper_observation_mm(arm: TB6R5Interface | None, constant_mm: float | None) -> float:
    if constant_mm is not None:
        return float(constant_mm)
    if arm is not None:
        feedback = arm.get_gripper_distance_mm()
        if feedback is not None:
            return float(feedback)
    return DEFAULT_GRIPPER_OBSERVATION_MM


def gripper_desired_closed(
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


def gripper_edge_min_steps(fps: float, min_interval_s: float) -> int:
    """Convert edge min interval (seconds) to control-loop steps using deployment fps."""
    return max(1, int(round(fps * min_interval_s)))


def print_gripper_status(
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
    state = gripper_state_label_mm(gripper_cmd_mm, gripper_max_distance)
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


def print_gripper_config(
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
        f"{RED}[ACT][GRIPPER] 配置: max_dist={gripper_max_distance:.1f}mm "
        f"interval={gripper_interval:.1f} cmd_delta={gripper_cmd_delta:.2f}mm mode={mode}{RESET}"
    )
    print(
        f"{RED}[ACT][GRIPPER] action[6]/state[6] 单位为 mm（0=闭合，{gripper_max_distance:.0f}=张开）；"
        f"obs 优先读 actual_pos 反馈。{RESET}"
    )
    if temporal_ensemble_coeff is not None:
        print(
            f"{RED}[ACT][ACTION] Temporal Ensemble coeff={temporal_ensemble_coeff:g}, "
            f"chunk_size={chunk_size}：每步推理并融合重叠 chunk 预测。{RESET}"
        )
    elif refresh_policy_every_step:
        print(
            f"{RED}[ACT][ACTION] 每步 policy.reset() + 重推理（action queue 不累积；"
            f"chunk_size={chunk_size}）。{RESET}"
        )
    elif n_action_steps is not None:
        print(
            f"{RED}[ACT][ACTION] Action queue: chunk_size={chunk_size}, n_action_steps={n_action_steps}，"
            f"每 {n_action_steps} 步重推理。可调 --n-action-steps 或 --temporal-ensemble-coeff 0.01。{RESET}"
        )
