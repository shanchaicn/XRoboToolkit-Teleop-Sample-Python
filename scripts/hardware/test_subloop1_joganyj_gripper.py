"""SubLoop1 流式测试：JogAnyJ + MoveTwoFingersGripper（与遥操作相同下发路径）。

关节 1/2/4/5/6 固定 (deg)：15, -100, -80, -90, -45；关节 3 在 85°~95° 间正弦摆动；
夹爪距离在 [10, 50] mm 间同步正弦摆动。

Usage:
    python scripts/hardware/test_subloop1_joganyj_gripper.py --robot-ip 192.168.11.11
    python scripts/hardware/test_subloop1_joganyj_gripper.py --sl-immediate --rate-hz 20 --duration-s 60
    python scripts/hardware/test_subloop1_joganyj_gripper.py --dry-run
"""

from __future__ import annotations

import math
import signal
import sys
import time

import numpy as np
import tyro

from xrobotoolkit_teleop.hardware.interface.tb6r5 import (
    DEFAULT_GRIPPER_MAX_D,
    DEFAULT_JOG_ANY_JOINT_ACC,
    DEFAULT_JOG_ANY_JOINT_DEC,
    DEFAULT_JOG_ANY_JOINT_VEL,
    DEFAULT_TWO_FINGERS_GRIPPER_INTERVAL,
    DEFAULT_ZONE_RATIO,
    TB6R5Interface,
)
from xrobotoolkit_teleop.hardware.tb6r5_teleop_controller import DEFAULT_ROBOT_IP, DEFAULT_RPC_PORT

# 关节 3 摆动；其余轴固定（与 DEFAULT_HOME_JOINT_DEG 一致，joint6=-45）
FIXED_JOINT_DEG = (15.0, -100.0, None, -80.0, -90.0, -45.0)  # index2=joint3 由摆动填充
JOINT3_MIN_DEG = 85.0
JOINT3_MAX_DEG = 95.0
GRIPPER_MIN_MM = 10.0
GRIPPER_MAX_MM = 50.0


def _sine_lerp(t: float, period_s: float, vmin: float, vmax: float) -> float:
    """在 [vmin, vmax] 间按正弦往返；t=0 时位于区间中点。"""
    mid = 0.5 * (vmin + vmax)
    amp = 0.5 * (vmax - vmin)
    phase = 2.0 * math.pi * t / max(period_s, 1e-6)
    return mid + amp * math.sin(phase)


def _build_joint_q_rad(t: float, period_s: float) -> np.ndarray:
    joint3_deg = _sine_lerp(t, period_s, JOINT3_MIN_DEG, JOINT3_MAX_DEG)
    deg = [
        FIXED_JOINT_DEG[0],
        FIXED_JOINT_DEG[1],
        joint3_deg,
        FIXED_JOINT_DEG[3],
        FIXED_JOINT_DEG[4],
        FIXED_JOINT_DEG[5],
    ]
    return np.deg2rad(deg)


def _build_gripper_mm(t: float, period_s: float) -> float:
    return _sine_lerp(t, period_s, GRIPPER_MIN_MM, GRIPPER_MAX_MM)


def _wait_rpc_drain(arm: TB6R5Interface, timeout_s: float) -> None:
    """Wait for in-flight SubLoop1 async RPCs (first-exec + stream counters)."""
    deadline = time.monotonic() + max(timeout_s, 0.0)
    while time.monotonic() < deadline:
        with arm._state_lock:
            pending = arm._jog_async_pending + arm._subloop1_stream_pending
        if pending == 0:
            return
        time.sleep(0.01)


def _graceful_shutdown(
    arm: TB6R5Interface,
    *,
    ctrlc_pause_s: float,
    drain_timeout_s: float,
    exit_timeout_ms: int,
) -> None:
    print(f"[test] Ctrl+C: wait {ctrlc_pause_s:.1f}s before exit ...")
    time.sleep(max(ctrlc_pause_s, 0.0))
    _wait_rpc_drain(arm, drain_timeout_s)

    if arm._subloop1_active or arm._subloop1_exiting:
        print("[test] SubLoop1 exit (async) ...")
        arm.exit_subloop1_if_active(timeout_ms=exit_timeout_ms, blocking_exit=False)
        _wait_rpc_drain(arm, exit_timeout_ms / 1000.0)

    print("[test] Disable ...")
    arm.disconnect()
    print("[test] shutdown complete.")


def main(
    robot_ip: str = DEFAULT_ROBOT_IP,
    rpc_port: int = DEFAULT_RPC_PORT,
    rate_hz: float = 20.0,
    duration_s: float = 30.0,
    period_s: float = 2.0,
    zone_ratio: float = DEFAULT_ZONE_RATIO,
    joint_vel: float = 6.0,
    joint_acc: float = DEFAULT_JOG_ANY_JOINT_ACC,
    joint_dec: float = DEFAULT_JOG_ANY_JOINT_DEC,
    gripper_interval: float = DEFAULT_TWO_FINGERS_GRIPPER_INTERVAL,
    gripper_max_d: float = DEFAULT_GRIPPER_MAX_D,
    gripper_cmd_delta: float = 0.1,
    always_send_gripper: bool = True,
    sl_immediate: bool = False,
    enable_topic: bool = True,
    print_interval_s: float = 0.5,
    shutdown_ctrlc_pause_s: float = 10.0,
    shutdown_drain_timeout_s: float = 2.0,
    shutdown_exit_timeout_ms: int = 10000,
    dry_run: bool = False,
) -> None:
    rate_hz = max(float(rate_hz), 1.0)
    dt = 1.0 / rate_hz
    stop = False
    shutting_down = False

    def _on_sigint(_signum, _frame):
        nonlocal stop, shutting_down
        if shutting_down:
            print("\n[test] force quit.")
            sys.exit(130)
        stop = True
        print("\n[test] Ctrl+C: stopping (press again to force quit) ...")

    signal.signal(signal.SIGINT, _on_sigint)

    fmt = TB6R5Interface(
        ip=robot_ip,
        zone_ratio=zone_ratio,
        joint_vel=joint_vel,
        joint_acc=joint_acc,
        joint_dec=joint_dec,
        subloop1_immediate=sl_immediate,
    )
    arm: TB6R5Interface | None = None
    if dry_run:
        print("[test] dry-run: print SubLoop1 commands only, no RPC.")
    else:
        arm = TB6R5Interface(
            ip=robot_ip,
            rpc_port=rpc_port,
            enable_topic=enable_topic,
            zone_ratio=zone_ratio,
            joint_vel=joint_vel,
            joint_acc=joint_acc,
            joint_dec=joint_dec,
            subloop1_immediate=sl_immediate,
        )
        arm.connect()
        arm.reset_joint_stream()
        arm._gripper_cmd_delta_mm = max(float(gripper_cmd_delta), 0.0)
        print(
            f"[test] SubLoop1 stream started: ip={robot_ip} rate={rate_hz:.1f}Hz "
            f"period={period_s:.1f}s sl_immediate={sl_immediate} "
            f"always_send_gripper={always_send_gripper}"
        )

    t0 = time.monotonic()
    last_print = 0.0
    sent = 0
    failed = 0

    try:
        while not stop:
            elapsed = time.monotonic() - t0
            if duration_s > 0 and elapsed >= duration_s:
                break

            q_rad = _build_joint_q_rad(elapsed, period_s)
            grip_mm = _build_gripper_mm(elapsed, period_s)
            joint3_deg = math.degrees(float(q_rad[2]))

            if dry_run:
                arm_inner = fmt._strip_cmd_braces(fmt._format_jog_any_j_cmd(q_rad, clear_buffer=0))
                grip_inner = fmt._format_gripper_inner(grip_mm, gripper_interval, gripper_max_d)
                cmd = fmt.format_subloop1_exec_cmd(arm_inner, grip_inner, immediate=sl_immediate)
                if elapsed - last_print >= print_interval_s:
                    print(
                        f"[dry-run] t={elapsed:5.2f}s j3={joint3_deg:6.2f}deg "
                        f"grip={grip_mm:5.2f}mm\n  {cmd}"
                    )
                    last_print = elapsed
            else:
                ok = arm.set_joint_positions_with_gripper(
                    q_rad,
                    grip_mm,
                    force=always_send_gripper,
                    interval=gripper_interval,
                    max_distance=gripper_max_d,
                    cmd_delta=gripper_cmd_delta,
                )
                if ok:
                    sent += 1
                else:
                    failed += 1

                if elapsed - last_print >= print_interval_s:
                    q_fb = arm.get_joint_positions()
                    grip_fb = arm.get_gripper_distance_mm()
                    j3_fb_deg = math.degrees(float(q_fb[2])) if len(q_fb) > 2 else float("nan")
                    print(
                        f"[test] t={elapsed:5.2f}s sent={sent} fail={failed} "
                        f"cmd_j3={joint3_deg:6.2f}deg fb_j3={j3_fb_deg:6.2f}deg "
                        f"cmd_grip={grip_mm:5.2f}mm fb_grip={grip_fb:.2f}mm "
                        f"subloop1_active={arm._subloop1_active}"
                    )
                    last_print = elapsed

            time.sleep(dt)
    finally:
        if arm is not None and not dry_run:
            shutting_down = True
            try:
                _graceful_shutdown(
                    arm,
                    ctrlc_pause_s=shutdown_ctrlc_pause_s,
                    drain_timeout_s=shutdown_drain_timeout_s,
                    exit_timeout_ms=shutdown_exit_timeout_ms,
                )
            except Exception as exc:
                print(f"[test] graceful shutdown failed: {exc}")
        print(f"[test] done. total_sent={sent} total_failed={failed}")


if __name__ == "__main__":
    tyro.cli(main)
