"""Shared teleop command gate for TB6-R5 LeRobot robot + PICO teleop plugins."""

from __future__ import annotations

import threading

_lock = threading.Lock()
_grip_session_active = False
_gripper_trigger_active = False


def set_grip_session_active(active: bool) -> None:
    global _grip_session_active
    with _lock:
        _grip_session_active = bool(active)


def is_grip_session_active() -> bool:
    with _lock:
        return _grip_session_active


def set_gripper_trigger_active(active: bool) -> None:
    global _gripper_trigger_active
    with _lock:
        _gripper_trigger_active = bool(active)


def is_gripper_trigger_active() -> bool:
    with _lock:
        return _gripper_trigger_active


def reset_command_gate() -> None:
    set_grip_session_active(False)
    set_gripper_trigger_active(False)
