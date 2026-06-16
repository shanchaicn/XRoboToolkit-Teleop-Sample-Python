"""单模型直连 RPC 测试（无 SubLoop / SubLoop1）。

仅发单模型指令（全程无 || / 无 SubLoop）：
  init    -> {Enable}  {Start}
  arm     -> {JogAnyJ --jointtarget_value=...}
  gripper -> {MoveTwoFingersGripper --distance=... --interval=...}
  exit    -> {Stop}（仅 arm） {Disable}

关节 3 在 85°~95° 正弦摆动；夹爪 10~50 mm 同步摆动。

Usage:
    python scripts/hardware/test_direct_single_rpc.py --mode arm --robot-ip 192.168.11.11
    python scripts/hardware/test_direct_single_rpc.py --mode gripper --rate-hz 2
    python scripts/hardware/test_direct_single_rpc.py --mode arm --transport async --rate-hz 20
    python scripts/hardware/test_direct_single_rpc.py --dry-run --mode gripper
"""

from __future__ import annotations

import math
import random
import signal
import sys
import time
from typing import Callable, Literal, Optional

import numpy as np
import tyro

from xrobotoolkit_teleop.hardware.interface.tb6r5 import (
    DEFAULT_GRIPPER_MAX_D,
    DEFAULT_JOG_ANY_J_LAST_COUNT,
    DEFAULT_JOG_ANY_JOINT_ACC,
    DEFAULT_JOG_ANY_JOINT_DEC,
    DEFAULT_JOG_ANY_JOINT_VEL,
    DEFAULT_JOG_ASYNC_TIMEOUT_MS,
    DEFAULT_TWO_FINGERS_GRIPPER_INTERVAL,
    DEFAULT_ZONE_RATIO,
    _setup_rpc_import,
)

DEFAULT_ROBOT_IP = "192.168.11.11"
DEFAULT_RPC_PORT = 5868

FIXED_JOINT_DEG = (15.0, -100.0, None, -80.0, -90.0, -45.0)
JOINT3_MIN_DEG = 85.0
JOINT3_MAX_DEG = 95.0
GRIPPER_MIN_MM = 10.0
GRIPPER_MAX_MM = 50.0

TestMode = Literal["arm", "gripper"]
Transport = Literal["async", "sync"]


def _sine_lerp(t: float, period_s: float, vmin: float, vmax: float) -> float:
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


def _format_jointtarget_value(q: np.ndarray) -> str:
    q = np.asarray(q, dtype=float).ravel()
    values = [0.0] * 10
    for i in range(min(6, len(q))):
        values[i] = float(q[i])
    return "{" + ",".join(f"{v:.6f}" for v in values) + "}"


def _format_jog_any_j_cmd(
    q: np.ndarray,
    *,
    zone_ratio: float,
    clear_buffer: int,
    joint_vel: float,
    joint_acc: float,
    joint_dec: float,
    last_count: int = DEFAULT_JOG_ANY_J_LAST_COUNT,
) -> str:
    val_str = _format_jointtarget_value(q)
    inner = (
        f"JogAnyJ --jointtarget_value={val_str}"
        f" --zone_ratio={zone_ratio:.4f} --clear_buffer={int(clear_buffer)}"
        f" --last_count={int(last_count)}"
        f" --joint_vel={joint_vel:.4f} --joint_acc={joint_acc:.4f} --joint_dec={joint_dec:.4f}"
    )
    return "{" + inner + "}"


def _format_gripper_cmd(distance_mm: float, interval: float) -> str:
    d = max(0.0, min(float(distance_mm), DEFAULT_GRIPPER_MAX_D))
    inner = f"MoveTwoFingersGripper --distance={d:.4f} --interval={max(0.0, interval):.4f}"
    return "{" + inner + "}"


class DirectSingleRpc:
    """单模型 RPC：JogAnyJ 或 MoveTwoFingersGripper，无 SubLoop。"""

    def __init__(self, ip: str, port: int):
        _setup_rpc_import()
        import rpc

        self._rpc = rpc
        self.ip = ip
        self.port = port
        self.client = rpc.CPPClient(ip, port)
        self._async_pending = 0
        self._arm_stream_count = 0

    def _call_sync(self, cmd: str, timeout_ms: int = 5000, sleep_s: float = 0.0, ignore_errors: bool = False) -> bool:
        msg = self._rpc.Msg(cmd)
        msg.setMsgID(10001)
        msg.setMsgSeqID(random.randint(1, 10000))
        status, resp_list = self.client.CallAwait(msg, timeout_ms)
        if status != 0:
            print(f"[single] RPC sync failed: {cmd[:96]}... (status={status})")
            return False
        for r in resp_list or []:
            if r.code < 0 and not ignore_errors:
                print(f"[single] RPC error: {cmd[:96]}... -> {r.message}")
                return False
        if sleep_s > 0:
            time.sleep(sleep_s)
        return True

    def _call_async(self, cmd: str, timeout_ms: int, on_done: Optional[Callable[[int], None]] = None) -> bool:
        def _cb(status: int, _resp_list):
            self._async_pending = max(0, self._async_pending - 1)
            if on_done is not None:
                on_done(status)

        msg = self._rpc.Msg(cmd)
        msg.setMsgID(10001)
        msg.setMsgSeqID(random.randint(1, 10000))
        ok = self.client.CallAsync(msg, timeout_ms, _cb)
        if ok:
            self._async_pending += 1
        return bool(ok)

    def init_robot(self, mode: TestMode) -> bool:
        print(f"[single] connecting {self.ip}:{self.port} mode={mode}")
        init_cmds = [
            "{Clear}",
            "{Disable}",
            "{Mode}",
            "{SetMaxToq}",
            "{Recover}",
            "{SetRate}",
            "{Var --clear}",
            "{Recover}",
            "{SetUsingSP --state=on}",
            "{Var --type=jointtarget --name=teleop --value={0,0,0,0,0,0,0,0,0,0}}",
        ]
        for cmd in init_cmds:
            if not self._call_sync(cmd, timeout_ms=5000, sleep_s=0.1):
                print(f"[single] init failed at {cmd}")
                return False
        if not self._call_sync("{Enable}", timeout_ms=5000, sleep_s=0.1):
            print("[single] init failed at {Enable}")
            return False
        if not self._call_sync("{Start}", timeout_ms=5000, sleep_s=0.1, ignore_errors=True):
            print("[single] init failed at {Start}")
            return False
        print("[single] init done ({Enable} + {Start}, no ||).")
        return True

    def send_arm(
        self,
        q_rad: np.ndarray,
        *,
        zone_ratio: float,
        joint_vel: float,
        joint_acc: float,
        joint_dec: float,
        transport: Transport,
        sync_timeout_ms: int,
        async_timeout_ms: int,
    ) -> bool:
        clear_buffer = 0 if self._arm_stream_count == 0 else 1
        cmd = _format_jog_any_j_cmd(
            q_rad,
            zone_ratio=zone_ratio,
            clear_buffer=clear_buffer,
            joint_vel=joint_vel,
            joint_acc=joint_acc,
            joint_dec=joint_dec,
        )
        if transport == "sync":
            ok = self._call_sync(cmd, timeout_ms=sync_timeout_ms)
        else:
            ok = self._call_async(cmd, timeout_ms=async_timeout_ms)
        if ok:
            self._arm_stream_count += 1
        return ok

    def send_gripper(
        self,
        grip_mm: float,
        *,
        interval: float,
        transport: Transport,
        sync_timeout_ms: int,
        async_timeout_ms: int,
    ) -> bool:
        cmd = _format_gripper_cmd(grip_mm, interval)
        if transport == "sync":
            return self._call_sync(cmd, timeout_ms=sync_timeout_ms)
        return self._call_async(cmd, timeout_ms=async_timeout_ms)

    def wait_async_pending(self, timeout_s: float) -> None:
        deadline = time.monotonic() + max(timeout_s, 0.0)
        while time.monotonic() < deadline:
            if self._async_pending == 0:
                return
            time.sleep(0.01)

    def shutdown(self, *, mode: TestMode, ctrlc_pause_s: float, drain_timeout_s: float) -> None:
        print(f"[single] Ctrl+C: wait {ctrlc_pause_s:.1f}s before Stop/Disable ...")
        time.sleep(max(ctrlc_pause_s, 0.0))
        self.wait_async_pending(drain_timeout_s)
        if mode == "arm":
            print("[single] Stop ...")
            self._call_sync("{Stop}", timeout_ms=3000, sleep_s=0.1, ignore_errors=True)
        print("[single] Disable ...")
        self._call_sync("{Disable}", timeout_ms=3000, sleep_s=0.1)
        print("[single] shutdown complete.")


def main(
    mode: TestMode = "arm",
    robot_ip: str = DEFAULT_ROBOT_IP,
    rpc_port: int = DEFAULT_RPC_PORT,
    rate_hz: float | None = None,
    duration_s: float = 30.0,
    period_s: float = 2.0,
    zone_ratio: float = DEFAULT_ZONE_RATIO,
    joint_vel: float = DEFAULT_JOG_ANY_JOINT_VEL,
    joint_acc: float = DEFAULT_JOG_ANY_JOINT_ACC,
    joint_dec: float = DEFAULT_JOG_ANY_JOINT_DEC,
    gripper_interval: float = DEFAULT_TWO_FINGERS_GRIPPER_INTERVAL,
    transport: Transport = "async",
    sync_timeout_ms: int = 30_000,
    async_timeout_ms: int = DEFAULT_JOG_ASYNC_TIMEOUT_MS,
    print_interval_s: float = 0.5,
    shutdown_ctrlc_pause_s: float = 10.0,
    shutdown_drain_timeout_s: float = 2.0,
    dry_run: bool = False,
) -> None:
    if rate_hz is None:
        rate_hz = 20.0 if mode == "arm" else 2.0
    rate_hz = max(float(rate_hz), 0.1)
    dt = 1.0 / rate_hz
    stop = False
    shutting_down = False

    def _on_sigint(_signum, _frame):
        nonlocal stop, shutting_down
        if shutting_down:
            print("\n[single] force quit.")
            sys.exit(130)
        stop = True
        print("\n[single] Ctrl+C: stopping (press again to force quit) ...")

    signal.signal(signal.SIGINT, _on_sigint)

    rpc_client: DirectSingleRpc | None = None
    if not dry_run:
        rpc_client = DirectSingleRpc(robot_ip, rpc_port)
        if not rpc_client.init_robot(mode):
            print("[single] abort: init failed.")
            return
        print(
            f"[single] mode={mode} transport={transport} rate={rate_hz:.1f}Hz "
            f"period={period_s:.1f}s (single model, no ||)"
        )
    else:
        print(f"[single] dry-run mode={mode} transport={transport} rate={rate_hz:.1f}Hz")

    t0 = time.monotonic()
    last_print = 0.0
    sent = 0

    try:
        while not stop:
            elapsed = time.monotonic() - t0
            if duration_s > 0 and elapsed >= duration_s:
                break

            if mode == "arm":
                q_rad = _build_joint_q_rad(elapsed, period_s)
                joint3_deg = math.degrees(float(q_rad[2]))
                cmd = _format_jog_any_j_cmd(
                    q_rad,
                    zone_ratio=zone_ratio,
                    clear_buffer=0,
                    joint_vel=joint_vel,
                    joint_acc=joint_acc,
                    joint_dec=joint_dec,
                )
                if dry_run:
                    if elapsed - last_print >= print_interval_s:
                        print(f"[dry-run] t={elapsed:5.2f}s j3={joint3_deg:6.2f}deg\n  {cmd}")
                        last_print = elapsed
                else:
                    ok = rpc_client.send_arm(
                        q_rad,
                        zone_ratio=zone_ratio,
                        joint_vel=joint_vel,
                        joint_acc=joint_acc,
                        joint_dec=joint_dec,
                        transport=transport,
                        sync_timeout_ms=sync_timeout_ms,
                        async_timeout_ms=async_timeout_ms,
                    )
                    if ok:
                        sent += 1
                    if elapsed - last_print >= print_interval_s:
                        print(
                            f"[single] t={elapsed:5.2f}s sent={sent} pending={rpc_client._async_pending} "
                            f"j3={joint3_deg:6.2f}deg"
                        )
                        last_print = elapsed
            else:
                grip_mm = _build_gripper_mm(elapsed, period_s)
                cmd = _format_gripper_cmd(grip_mm, gripper_interval)
                if dry_run:
                    if elapsed - last_print >= print_interval_s:
                        print(f"[dry-run] t={elapsed:5.2f}s grip={grip_mm:5.2f}mm\n  {cmd}")
                        last_print = elapsed
                else:
                    ok = rpc_client.send_gripper(
                        grip_mm,
                        interval=gripper_interval,
                        transport=transport,
                        sync_timeout_ms=sync_timeout_ms,
                        async_timeout_ms=async_timeout_ms,
                    )
                    if ok:
                        sent += 1
                    if elapsed - last_print >= print_interval_s:
                        print(
                            f"[single] t={elapsed:5.2f}s sent={sent} pending={rpc_client._async_pending} "
                            f"grip={grip_mm:5.2f}mm"
                        )
                        last_print = elapsed

            time.sleep(dt)
    finally:
        if rpc_client is not None:
            shutting_down = True
            rpc_client.shutdown(
                mode=mode,
                ctrlc_pause_s=shutdown_ctrlc_pause_s,
                drain_timeout_s=shutdown_drain_timeout_s,
            )
        print(f"[single] done. total_sent={sent}")


if __name__ == "__main__":
    tyro.cli(main)
