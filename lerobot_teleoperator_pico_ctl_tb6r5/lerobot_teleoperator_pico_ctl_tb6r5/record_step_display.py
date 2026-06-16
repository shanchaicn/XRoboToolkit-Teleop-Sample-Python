"""Terminal step visualization for lerobot-record-pico-ctl (stderr, RPC-spam safe)."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TextIO

# RPC debug prints go to stdout; status banners go to stderr so they stay visible.
_STATUS_OUT: TextIO = sys.stderr


class RecordPhase(str, Enum):
    IDLE = "idle"
    CONNECTING = "connecting"
    RECORDING = "recording"
    RESET = "reset"
    SAVING = "saving"
    RERECORD = "rerecord"
    STOPPING = "stopping"
    DONE = "done"


_PHASE_LABELS: dict[RecordPhase, str] = {
    RecordPhase.IDLE: "待机",
    RecordPhase.CONNECTING: "连接设备",
    RecordPhase.RECORDING: "录制中",
    RecordPhase.RESET: "Reset（整理场景，不写盘）",
    RecordPhase.SAVING: "保存 Episode",
    RecordPhase.RERECORD: "丢弃并重录",
    RecordPhase.STOPPING: "停止采集",
    RecordPhase.DONE: "完成",
}

_PHASE_HINTS: dict[RecordPhase, str] = {
    RecordPhase.CONNECTING: "正在连接 robot / teleop / 相机…",
    RecordPhase.RECORDING: "按住 right_grip 遥操作 | B=结束并保存 | A=丢弃重录 | X=回 home",
    RecordPhase.RESET: "整理物体/等待回 home | 按 B 可提前进入下一条 | 此阶段不写入 dataset",
    RecordPhase.SAVING: "正在写入 parquet 与视频，请稍候…",
    RecordPhase.RERECORD: "已清空当前 buffer，将重新录制同一条 episode",
}


@dataclass
class RecordStepDisplay:
    """Prints workflow phase banners and periodic progress to stderr."""

    num_episodes: int = 1
    fps: int = 30
    episode_time_s: float = 60.0
    reset_time_s: float = 60.0
    _phase: RecordPhase = field(default=RecordPhase.IDLE, init=False)
    _episode_index: int = field(default=0, init=False)
    _last_progress_t: float = field(default=0.0, init=False)
    _progress_interval_s: float = field(default=2.0, init=False)
    _enabled: bool = field(default=True, init=False)

    def set_phase(
        self,
        phase: RecordPhase,
        *,
        episode_index: int | None = None,
        extra: str | None = None,
    ) -> None:
        if not self._enabled:
            return
        if episode_index is not None:
            self._episode_index = episode_index
        if phase == self._phase and extra is None and episode_index is None:
            return
        self._phase = phase
        self._print_banner(extra=extra)

    def notify(self, message: str) -> None:
        if not self._enabled:
            return
        print(f"\n[pico_ctl] {message}", file=_STATUS_OUT, flush=True)

    def maybe_update_progress(
        self,
        *,
        elapsed_s: float,
        limit_s: float | None,
        frame_count: int | None = None,
    ) -> None:
        if not self._enabled:
            return
        now = time.perf_counter()
        if now - self._last_progress_t < self._progress_interval_s:
            return
        self._last_progress_t = now
        parts: list[str] = [f"▶ {_PHASE_LABELS.get(self._phase, self._phase.value)}"]
        parts.append(f"ep {self._episode_index + 1}/{self.num_episodes}")
        parts.append(f"{elapsed_s:.1f}s")
        if limit_s is not None:
            parts.append(f"/ {limit_s:.0f}s")
        if frame_count is not None:
            parts.append(f"| {frame_count} 帧")
        print(f"[pico_ctl] {' | '.join(parts)}", file=_STATUS_OUT, flush=True)

    def _print_banner(self, *, extra: str | None = None) -> None:
        label = _PHASE_LABELS.get(self._phase, self._phase.value)
        ep_line = ""
        if self._phase in (RecordPhase.RECORDING, RecordPhase.RESET, RecordPhase.SAVING, RecordPhase.RERECORD):
            ep_line = f"Episode {self._episode_index + 1}/{self.num_episodes}"
        hint = _PHASE_HINTS.get(self._phase, "")
        width = 72
        print("\n" + "=" * width, file=_STATUS_OUT, flush=True)
        title = f"  [{self._phase.value.upper():^10}]  {label}"
        if ep_line:
            title = f"{title}  |  {ep_line}"
        print(title, file=_STATUS_OUT, flush=True)
        if hint:
            print(f"  → {hint}", file=_STATUS_OUT, flush=True)
        if extra:
            print(f"  → {extra}", file=_STATUS_OUT, flush=True)
        print("=" * width + "\n", file=_STATUS_OUT, flush=True)


_DISPLAY: RecordStepDisplay | None = None


def get_display() -> RecordStepDisplay | None:
    return _DISPLAY


def install_display(**kwargs: Any) -> RecordStepDisplay:
    global _DISPLAY
    _DISPLAY = RecordStepDisplay(**kwargs)
    return _DISPLAY


def clear_display() -> None:
    global _DISPLAY
    _DISPLAY = None
