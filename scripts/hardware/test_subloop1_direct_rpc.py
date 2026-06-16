"""SubLoop / SubLoop1 直连 RPC 测试（不经过遥操作控制器 / TB6R5Interface）。

定期发送双模型指令，支持：
  --subloop-cmd subloop1  流式会话（首帧 async → 后续 async → exit），默认
  --subloop-cmd subloop    无会话，每帧 CallAwait 同步阻塞（指令名为 SubLoop，不带 1）

模式 (--mode)：
  both    - JogAnyJ + MoveTwoFingersGripper
  arm     - 仅 JogAnyJ
  gripper - 仅 MoveTwoFingersGripper

mode=both 时臂/夹爪可独立设频率（默认臂 20Hz、夹爪 2Hz）；仅臂或仅夹爪时只用对应频率。

Usage:
    python scripts/hardware/test_subloop1_direct_rpc.py --robot-ip 192.168.11.11
    python scripts/hardware/test_subloop1_direct_rpc.py --arm-rate-hz 20 --gripper-rate-hz 2
    python scripts/hardware/test_subloop1_direct_rpc.py --subloop-cmd subloop --mode arm --arm-rate-hz 5
    python scripts/hardware/test_subloop1_direct_rpc.py --subloop-cmd subloop1 --mode gripper
    python scripts/hardware/test_subloop1_direct_rpc.py --dry-run --subloop-cmd subloop
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
    DEFAULT_SUBLOOP1_EXEC_TIMEOUT_MS,
    DEFAULT_TWO_FINGERS_GRIPPER_INTERVAL,
    DEFAULT_ZONE_RATIO,
    NOT_RUN_EXECUTE,
    _setup_rpc_import,
)

DEFAULT_ROBOT_IP = "192.168.11.11"
DEFAULT_RPC_PORT = 5868

FIXED_JOINT_DEG = (15.0, -100.0, None, -80.0, -90.0, -45.0)
JOINT3_MIN_DEG = 85.0
JOINT3_MAX_DEG = 95.0
GRIPPER_MIN_MM = 10.0
GRIPPER_MAX_MM = 50.0

TestMode = Literal["both", "arm", "gripper"]
SubloopCmd = Literal["subloop1", "subloop"]


def _subloop_rpc_name(variant: SubloopCmd) -> str:
    return "SubLoop1" if variant == "subloop1" else "SubLoop"


def _format_subloop_exit_cmd(variant: SubloopCmd) -> str:
    name = _subloop_rpc_name(variant)
    return f"{{{name} --exec={{exit}}||{name} --exec={{exit}}}}"


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


def _format_jog_any_j_inner(
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
    return (
        f"JogAnyJ --jointtarget_value={val_str}"
        f" --zone_ratio={zone_ratio:.4f} --clear_buffer={int(clear_buffer)}"
        f" --last_count={int(last_count)}"
        f" --joint_vel={joint_vel:.4f} --joint_acc={joint_acc:.4f} --joint_dec={joint_dec:.4f}"
    )


def _format_gripper_inner(distance_mm: float, interval: float) -> str:
    d = max(0.0, min(float(distance_mm), DEFAULT_GRIPPER_MAX_D))
    return f"MoveTwoFingersGripper --distance={d:.4f} --interval={max(0.0, interval):.4f}"


def _format_subloop_exec(
    arm_inner: str,
    grip_inner: str,
    *,
    variant: SubloopCmd,
    immediate: bool,
) -> str:
    name = _subloop_rpc_name(variant)
    imm = " --immediate=true" if immediate else ""
    return (
        f"{{{name} --exec={{{arm_inner}}}{imm}"
        f"||{name} --exec={{{grip_inner}}}{imm}}}"
    )


def _build_subloop_inners(
    mode: TestMode,
    q_rad: np.ndarray,
    grip_mm: float,
    *,
    send_arm: bool,
    send_gripper: bool,
    zone_ratio: float,
    clear_buffer: int,
    joint_vel: float,
    joint_acc: float,
    joint_dec: float,
    gripper_interval: float,
) -> tuple[str, str]:
    arm_inner = NOT_RUN_EXECUTE
    grip_inner = NOT_RUN_EXECUTE
    if send_arm and mode in ("both", "arm"):
        arm_inner = _format_jog_any_j_inner(
            q_rad,
            zone_ratio=zone_ratio,
            clear_buffer=clear_buffer,
            joint_vel=joint_vel,
            joint_acc=joint_acc,
            joint_dec=joint_dec,
        )
    if send_gripper and mode in ("both", "gripper"):
        grip_inner = _format_gripper_inner(grip_mm, gripper_interval)
    return arm_inner, grip_inner


def _rate_strides(arm_rate_hz: float, gripper_rate_hz: float) -> tuple[float, int, int]:
    arm_rate_hz = max(float(arm_rate_hz), 0.1)
    gripper_rate_hz = max(float(gripper_rate_hz), 0.1)
    loop_hz = max(arm_rate_hz, gripper_rate_hz)
    arm_stride = max(1, round(loop_hz / arm_rate_hz))
    grip_stride = max(1, round(loop_hz / gripper_rate_hz))
    return loop_hz, arm_stride, grip_stride


class DirectSubLoopRpc:
    """最小 RPC 客户端：init + SubLoop/SubLoop1 exec + exit + Disable。"""

    def __init__(self, ip: str, port: int, *, variant: SubloopCmd, sl_immediate: bool):
        _setup_rpc_import()
        import rpc

        self._rpc = rpc
        self.ip = ip
        self.port = port
        self.variant = variant
        self.subloop_name = _subloop_rpc_name(variant)
        self.sl_immediate = sl_immediate
        self.client = rpc.CPPClient(ip, port)
        self._session_active = False
        self._async_pending = 0

    def _call_sync(self, cmd: str, timeout_ms: int = 5000, sleep_s: float = 0.0, ignore_errors: bool = False) -> bool:
        msg = self._rpc.Msg(cmd)
        msg.setMsgID(10001)
        msg.setMsgSeqID(random.randint(1, 10000))
        status, resp_list = self.client.CallAwait(msg, timeout_ms)
        if status != 0:
            print(f"[direct] RPC sync failed: {cmd[:80]}... (status={status})")
            return False
        for r in resp_list or []:
            if r.code < 0 and not ignore_errors:
                print(f"[direct] RPC error: {cmd[:80]}... -> {r.message}")
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

    def init_robot(self) -> None:
        print(f"[direct] connecting {self.ip}:{self.port}")
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
            self._call_sync(cmd, timeout_ms=5000, sleep_s=0.1)
        self._call_sync(f"{{Enable||{NOT_RUN_EXECUTE}}}", timeout_ms=5000, sleep_s=0.1)
        self._call_sync(f"{{Start||{NOT_RUN_EXECUTE}}}", timeout_ms=5000, sleep_s=0.1, ignore_errors=True)
        print("[direct] init done (Enable + Start).")

    def send_subloop_exec(
        self,
        arm_inner: str,
        grip_inner: str,
        *,
        sync_timeout_ms: int,
    ) -> bool:
        cmd = _format_subloop_exec(
            arm_inner,
            grip_inner,
            variant=self.variant,
            immediate=self.sl_immediate,
        )
        if self.variant == "subloop":
            return self._call_sync(cmd, timeout_ms=sync_timeout_ms)
        if not self._session_active:
            ok = self._call_async(cmd, timeout_ms=5_000_000)
            if ok:
                self._session_active = True
            return ok
        return self._call_async(cmd, timeout_ms=DEFAULT_SUBLOOP1_EXEC_TIMEOUT_MS)

    def exit_session_async(self, timeout_ms: int = 10_000) -> bool:
        if self.variant != "subloop1" or not self._session_active:
            return True

        def _on_exit(status: int):
            if status < 0:
                print(f"[direct] {self.subloop_name} exit async failed (status={status})")
            self._session_active = False

        self._session_active = False
        exit_cmd = _format_subloop_exit_cmd("subloop1")
        return self._call_async(exit_cmd, timeout_ms=timeout_ms, on_done=_on_exit)

    def wait_async_pending(self, timeout_s: float) -> None:
        deadline = time.monotonic() + max(timeout_s, 0.0)
        while time.monotonic() < deadline:
            if self._async_pending == 0:
                return
            time.sleep(0.01)

    def shutdown(self, *, ctrlc_pause_s: float, drain_timeout_s: float, exit_timeout_ms: int) -> None:
        print(f"[direct] Ctrl+C: wait {ctrlc_pause_s:.1f}s before exit ...")
        time.sleep(max(ctrlc_pause_s, 0.0))
        self.wait_async_pending(drain_timeout_s)
        if self.variant == "subloop1" and self._session_active:
            print(f"[direct] {self.subloop_name} exit (async) ...")
            self.exit_session_async(timeout_ms=exit_timeout_ms)
            self.wait_async_pending(exit_timeout_ms / 1000.0)
        print("[direct] Disable ...")
        self._call_sync("{Disable}", timeout_ms=3000, sleep_s=0.1)
        print("[direct] shutdown complete.")


def main(
    mode: TestMode = "both",
    subloop_cmd: SubloopCmd = "subloop1",
    robot_ip: str = DEFAULT_ROBOT_IP,
    rpc_port: int = DEFAULT_RPC_PORT,
    arm_rate_hz: float = 20.0,
    gripper_rate_hz: float = 2.0,
    duration_s: float = 30.0,
    period_s: float = 2.0,
    zone_ratio: float = DEFAULT_ZONE_RATIO,
    joint_vel: float = DEFAULT_JOG_ANY_JOINT_VEL,
    joint_acc: float = DEFAULT_JOG_ANY_JOINT_ACC,
    joint_dec: float = DEFAULT_JOG_ANY_JOINT_DEC,
    gripper_interval: float = DEFAULT_TWO_FINGERS_GRIPPER_INTERVAL,
    sl_immediate: bool = False,
    sync_exec_timeout_ms: int = 30_000,
    print_interval_s: float = 0.5,
    shutdown_ctrlc_pause_s: float = 10.0,
    shutdown_drain_timeout_s: float = 2.0,
    shutdown_exit_timeout_ms: int = 10_000,
    dry_run: bool = False,
) -> None:
    loop_hz, arm_stride, grip_stride = _rate_strides(arm_rate_hz, gripper_rate_hz)
    dt = 1.0 / loop_hz
    stop = False
    shutting_down = False

    def _on_sigint(_signum, _frame):
        nonlocal stop, shutting_down
        if shutting_down:
            print("\n[direct] force quit.")
            sys.exit(130)
        stop = True
        print("\n[direct] Ctrl+C: stopping (press again to force quit) ...")

    signal.signal(signal.SIGINT, _on_sigint)

    rpc_client: DirectSubLoopRpc | None = None
    subloop_name = _subloop_rpc_name(subloop_cmd)
    if not dry_run:
        rpc_client = DirectSubLoopRpc(
            robot_ip,
            rpc_port,
            variant=subloop_cmd,
            sl_immediate=sl_immediate,
        )
        rpc_client.init_robot()
        transport = "sync CallAwait/帧" if subloop_cmd == "subloop" else "async 流式会话"
        print(
            f"[direct] {subloop_name} mode={mode} transport={transport} "
            f"arm={arm_rate_hz:.1f}Hz grip={gripper_rate_hz:.1f}Hz "
            f"period={period_s:.1f}s sl_immediate={sl_immediate}"
        )
    else:
        print(
            f"[direct] dry-run {subloop_name} mode={mode} "
            f"arm={arm_rate_hz:.1f}Hz grip={gripper_rate_hz:.1f}Hz"
        )

    t0 = time.monotonic()
    last_print = 0.0
    sent = 0
    arm_stream_count = 0
    frame = 0

    try:
        while not stop:
            elapsed = time.monotonic() - t0
            if duration_s > 0 and elapsed >= duration_s:
                break

            send_arm = mode in ("both", "arm") and (frame % arm_stride == 0)
            send_gripper = mode in ("both", "gripper") and (frame % grip_stride == 0)
            if send_arm or send_gripper:
                q_rad = _build_joint_q_rad(elapsed, period_s)
                grip_mm = _build_gripper_mm(elapsed, period_s)
                joint3_deg = math.degrees(float(q_rad[2]))
                clear_buffer = 0 if arm_stream_count == 0 else 1

                arm_inner, grip_inner = _build_subloop_inners(
                    mode,
                    q_rad,
                    grip_mm,
                    send_arm=send_arm,
                    send_gripper=send_gripper,
                    zone_ratio=zone_ratio,
                    clear_buffer=clear_buffer,
                    joint_vel=joint_vel,
                    joint_acc=joint_acc,
                    joint_dec=joint_dec,
                    gripper_interval=gripper_interval,
                )
                cmd = _format_subloop_exec(
                    arm_inner,
                    grip_inner,
                    variant=subloop_cmd,
                    immediate=sl_immediate,
                )
                tag = []
                if send_arm:
                    tag.append("arm")
                if send_gripper:
                    tag.append("grip")
                chan = "+".join(tag)

                if dry_run:
                    if elapsed - last_print >= print_interval_s:
                        extra = ""
                        if send_arm:
                            extra += f" j3={joint3_deg:6.2f}deg"
                        if send_gripper:
                            extra += f" grip={grip_mm:5.2f}mm"
                        print(f"[dry-run] t={elapsed:5.2f}s [{chan}]{extra}\n  {cmd}")
                        last_print = elapsed
                else:
                    ok = rpc_client.send_subloop_exec(
                        arm_inner,
                        grip_inner,
                        sync_timeout_ms=sync_exec_timeout_ms,
                    )
                    if ok:
                        sent += 1
                        if send_arm:
                            arm_stream_count += 1
                    if elapsed - last_print >= print_interval_s:
                        extra = ""
                        if send_arm:
                            extra += f" j3={joint3_deg:6.2f}deg"
                        if send_gripper:
                            extra += f" grip={grip_mm:5.2f}mm"
                        session = (
                            f" session={rpc_client._session_active}"
                            if subloop_cmd == "subloop1"
                            else ""
                        )
                        print(
                            f"[direct] t={elapsed:5.2f}s [{chan}] sent={sent} "
                            f"pending={rpc_client._async_pending}{extra}{session}"
                        )
                        last_print = elapsed

            frame += 1
            time.sleep(dt)
    finally:
        if rpc_client is not None:
            shutting_down = True
            rpc_client.shutdown(
                ctrlc_pause_s=shutdown_ctrlc_pause_s,
                drain_timeout_s=shutdown_drain_timeout_s,
                exit_timeout_ms=shutdown_exit_timeout_ms,
            )
        print(f"[direct] done. total_sent={sent}")


if __name__ == "__main__":
    tyro.cli(main)
