"""TB6R5TeleopController adapter for LeRobot record (VR buttons, no internal logger)."""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional

from xrobotoolkit_teleop.hardware.tb6r5_teleop_controller import (
    DEFAULT_TB6R5_MANIPULATOR_CONFIG,
    DEFAULT_URDF_PATH,
    TB6R5TeleopController,
)

from .record_step_display import get_display


class LeRobotPicoCtlTB6R5Controller(TB6R5TeleopController):
    """Runs the hardware teleop IK loop and maps PICO A/B to LeRobot episode events."""

    def __init__(
        self,
        robot_urdf_path: str = DEFAULT_URDF_PATH,
        manipulator_config: dict = DEFAULT_TB6R5_MANIPULATOR_CONFIG,
        **kwargs,
    ):
        kwargs.setdefault("enable_log_data", False)
        kwargs.setdefault("enable_lerobot_log", False)
        kwargs.setdefault("enable_camera", False)
        super().__init__(
            robot_urdf_path=robot_urdf_path,
            manipulator_config=manipulator_config,
            **kwargs,
        )
        self._lerobot_stop_event = threading.Event()
        self._ik_worker: Optional[threading.Thread] = None
        self._terminate_episode = False
        self._rerecord_episode = False
        self._stop_recording = False
        self._prev_b_lerobot = False
        self._prev_a_lerobot = False

    def start_background_loop(self) -> None:
        if self._ik_worker is not None and self._ik_worker.is_alive():
            return
        self._robot_setup()
        self._start_time = time.time()
        self._lerobot_stop_event.clear()
        self._ik_worker = threading.Thread(
            name="_ik_thread",
            target=self._ik_thread,
            args=(self._lerobot_stop_event,),
            daemon=True,
        )
        self._ik_worker.start()
        msg = (
            f"IK/RPC 已启动 ({self.control_rate_hz} Hz, arm {self.arm_rpc_rate_hz:.0f} Hz, "
            f"gripper {self.gripper_rpc_rate_hz:.0f} Hz)"
        )
        display = get_display()
        if display is not None:
            display.notify(msg)
            display.notify("手柄: B=结束并保存 | A=丢弃重录 | X=回 home | Esc=停止")
        else:
            print(f"[pico_ctl_tb6r5] Control loop started {msg}.")
            print(
                "[pico_ctl_tb6r5] Episode: B=save/end+home, A=discard+rerecord+home, "
                "X=home, Esc=stop (optional keyboard)."
            )

    def stop_background_loop(self) -> None:
        self._lerobot_stop_event.set()
        if self._ik_worker is not None:
            self._ik_worker.join(timeout=3.0)
            self._ik_worker = None
        self._shutdown_robot()

    def _pre_ik_update(self):
        super()._pre_ik_update()
        self._check_lerobot_episode_buttons()

    def _check_lerobot_episode_buttons(self) -> None:
        """B: end/save episode + home. A: discard episode, rerecord + home (same as hardware teleop)."""
        try:
            b_pressed = self.xr_client.get_button_state_by_name("B")
            a_pressed = self.xr_client.get_button_state_by_name("A")
        except Exception:
            return

        if b_pressed and not self._prev_b_lerobot:
            self._terminate_episode = True
            self._notify_step("B：结束当前 episode，回 home，随后 Reset → 保存")
            self._go_to_home_pose()

        if a_pressed and not self._prev_a_lerobot:
            self._terminate_episode = True
            self._rerecord_episode = True
            self._notify_step("A：丢弃当前 buffer，回 home，随后重录同一条")
            self._go_to_home_pose()

        self._prev_b_lerobot = b_pressed
        self._prev_a_lerobot = a_pressed

    def _notify_step(self, message: str) -> None:
        display = get_display()
        if display is not None:
            display.notify(message)
        else:
            print(f"[pico_ctl_tb6r5] {message}")

    def consume_record_events(self) -> Dict[str, bool]:
        """Return and clear pending LeRobot record-loop event flags."""
        events = {
            "exit_early": self._terminate_episode,
            "rerecord_episode": self._rerecord_episode,
            "stop_recording": self._stop_recording,
        }
        self._terminate_episode = False
        self._rerecord_episode = False
        self._stop_recording = False
        return events

    def get_action_dict(self) -> Dict[str, float]:
        log = self._get_robot_state_for_logging()
        action = log["action"]
        return {f"state_{i}": float(action[i]) for i in range(min(7, len(action)))}

    def get_teleop_events(self) -> Dict[str, Any]:
        ev = self.consume_record_events()
        return {
            "terminate_episode": ev["exit_early"],
            "rerecord_episode": ev["rerecord_episode"],
            "success": False,
            "is_intervention": any(self.active.get(name, False) for name in self.manipulator_config),
        }
