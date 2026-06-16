"""lerobot-record wrapper: PICO A/B episode events + terminal step visualization."""

import time
from functools import wraps
from typing import Any

import lerobot.scripts.lerobot_record as lr_record
import lerobot.utils.utils as lerobot_utils
from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.lerobot_record import RecordConfig
from lerobot.utils.robot_utils import precise_sleep

from .record_step_display import RecordPhase, clear_display, get_display, install_display

_original_record_loop = lr_record.record_loop
_original_record_impl = lr_record.record.__wrapped__
_original_record_entry = lr_record.record
_original_log_say = lerobot_utils.log_say
_record_entry_patched = False
_patched_dataset_ids: set[int] = set()
_original_dataset_create = LeRobotDataset.create
_original_dataset_resume = getattr(LeRobotDataset, "resume", None)
_original_dataset_init = LeRobotDataset.__init__
_dataset_factory_patched = False


def _episode_buffer_size(dataset: LeRobotDataset | None) -> int | None:
    if dataset is None:
        return None
    buf = getattr(dataset, "episode_buffer", None)
    if buf is None:
        return None
    return int(buf.get("size", 0))


def _wrap_dataset_lifecycle(dataset: LeRobotDataset) -> None:
    ds_id = id(dataset)
    if ds_id in _patched_dataset_ids:
        return
    _patched_dataset_ids.add(ds_id)

    orig_save = dataset.save_episode
    orig_clear = dataset.clear_episode_buffer

    @wraps(orig_save)
    def save_episode(*args: Any, **kwargs: Any) -> None:
        display = get_display()
        frames = _episode_buffer_size(dataset)
        if display is not None:
            display.set_phase(
                RecordPhase.SAVING,
                episode_index=dataset.num_episodes,
                extra=f"缓冲 {frames or 0} 帧 → 写入 {dataset.root}",
            )
        orig_save(*args, **kwargs)
        if display is not None:
            display.notify(f"Episode {dataset.num_episodes} 已保存（累计 {dataset.num_episodes} 条）")

    @wraps(orig_clear)
    def clear_episode_buffer(*args: Any, **kwargs: Any) -> None:
        display = get_display()
        if display is not None:
            display.set_phase(
                RecordPhase.RERECORD,
                episode_index=dataset.num_episodes,
                extra="按 A 触发：清空 buffer，不写入磁盘",
            )
        orig_clear(*args, **kwargs)

    dataset.save_episode = save_episode  # type: ignore[method-assign]
    dataset.clear_episode_buffer = clear_episode_buffer  # type: ignore[method-assign]


def _patch_dataset_factory() -> None:
    global _dataset_factory_patched
    if _dataset_factory_patched:
        return

    @wraps(_original_dataset_create)
    def create_wrapper(*args: Any, **kwargs: Any) -> LeRobotDataset:
        dataset = _original_dataset_create(*args, **kwargs)
        _wrap_dataset_lifecycle(dataset)
        return dataset

    LeRobotDataset.create = create_wrapper  # type: ignore[method-assign]

    if _original_dataset_resume is not None:
        @wraps(_original_dataset_resume)
        def resume_wrapper(*args: Any, **kwargs: Any) -> LeRobotDataset:
            dataset = _original_dataset_resume(*args, **kwargs)
            _wrap_dataset_lifecycle(dataset)
            return dataset

        LeRobotDataset.resume = resume_wrapper  # type: ignore[method-assign, assignment]
    else:
        # lerobot < 0.5: resume loads via LeRobotDataset(...) constructor.
        @wraps(_original_dataset_init)
        def init_wrapper(self, *args: Any, **kwargs: Any) -> None:
            _original_dataset_init(self, *args, **kwargs)
            _wrap_dataset_lifecycle(self)

        LeRobotDataset.__init__ = init_wrapper  # type: ignore[method-assign]

    _dataset_factory_patched = True


def _restore_dataset_factory() -> None:
    global _dataset_factory_patched
    if not _dataset_factory_patched:
        return
    LeRobotDataset.create = _original_dataset_create  # type: ignore[method-assign]
    if _original_dataset_resume is not None:
        LeRobotDataset.resume = _original_dataset_resume  # type: ignore[method-assign, assignment]
    else:
        LeRobotDataset.__init__ = _original_dataset_init  # type: ignore[method-assign]
    _dataset_factory_patched = False


def _patched_log_say(text: str, *args: Any, **kwargs: Any) -> Any:
    display = get_display()
    if display is not None:
        if text.startswith("Recording episode"):
            try:
                ep = int(text.rsplit(" ", 1)[-1])
            except ValueError:
                ep = display._episode_index
            display.set_phase(RecordPhase.RECORDING, episode_index=ep)
        elif text == "Reset the environment":
            display.set_phase(RecordPhase.RESET, episode_index=display._episode_index)
        elif text == "Re-record episode":
            display.set_phase(RecordPhase.RERECORD, episode_index=display._episode_index)
        elif text == "Stop recording":
            display.set_phase(RecordPhase.STOPPING)
        elif text == "Exiting":
            display.set_phase(RecordPhase.DONE)
    return _original_log_say(text, *args, **kwargs)


def _record_loop_pico_ctl(*args, **kwargs):
    teleop = kwargs.get("teleop")
    events = kwargs.get("events")
    if teleop is None or events is None or not hasattr(teleop, "poll_record_events"):
        return _original_record_loop(*args, **kwargs)

    fps: int = kwargs["fps"]
    robot = kwargs["robot"]
    teleop_action_processor = kwargs["teleop_action_processor"]
    robot_action_processor = kwargs["robot_action_processor"]
    robot_observation_processor = kwargs["robot_observation_processor"]
    dataset = kwargs.get("dataset")
    policy = kwargs.get("policy")
    preprocessor = kwargs.get("preprocessor")
    postprocessor = kwargs.get("postprocessor")
    control_time_s = kwargs.get("control_time_s")
    single_task = kwargs.get("single_task")
    display_data = kwargs.get("display_data", False)
    display_compressed_images = kwargs.get("display_compressed_images", False)

    from lerobot.datasets.utils import build_dataset_frame
    from lerobot.policies.utils import make_robot_action
    from lerobot.utils.constants import ACTION, OBS_STR
    from lerobot.utils.control_utils import predict_action
    from lerobot.utils.utils import get_safe_torch_device
    from lerobot.utils.visualization_utils import log_rerun_data

    display = get_display()
    is_recording = dataset is not None
    if display is not None:
        if is_recording:
            display.set_phase(RecordPhase.RECORDING, episode_index=display._episode_index)
        else:
            display.set_phase(RecordPhase.RESET, episode_index=display._episode_index)

    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")

    if policy is not None and preprocessor is not None and postprocessor is not None:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

    timestamp = 0.0
    start_episode_t = time.perf_counter()
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        teleop.poll_record_events(events)
        if events["exit_early"]:
            events["exit_early"] = False
            if display is not None:
                if events.get("rerecord_episode"):
                    display.notify("收到结束信号（将丢弃并重录）")
                elif is_recording:
                    display.notify("收到结束信号（B：结束录制，随后 Reset → 保存）")
                else:
                    display.notify("Reset 提前结束，即将保存或进入下一条")
            break

        obs = robot.get_observation()
        obs_processed = robot_observation_processor(obs)

        if policy is not None or dataset is not None:
            observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)

        if policy is not None and preprocessor is not None and postprocessor is not None:
            action_values = predict_action(
                observation=observation_frame,
                policy=policy,
                device=get_safe_torch_device(policy.config.device),
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy.config.use_amp,
                task=single_task,
                robot_type=robot.robot_type,
            )
            act_processed_policy = make_robot_action(action_values, dataset.features)
            robot_action_to_send = robot_action_processor((act_processed_policy, obs))
            action_values = act_processed_policy
        else:
            act = teleop.get_action()
            act_processed_teleop = teleop_action_processor((act, obs))
            action_values = act_processed_teleop
            robot_action_to_send = robot_action_processor((act_processed_teleop, obs))

        robot.send_action(robot_action_to_send)

        if dataset is not None:
            action_frame = build_dataset_frame(dataset.features, action_values, prefix=ACTION)
            frame = {**observation_frame, **action_frame, "task": single_task}
            dataset.add_frame(frame)

        if display_data:
            log_rerun_data(
                observation=obs_processed, action=action_values, compress_images=display_compressed_images
            )

        timestamp = time.perf_counter() - start_episode_t
        if display is not None:
            display.maybe_update_progress(
                elapsed_s=timestamp,
                limit_s=control_time_s,
                frame_count=_episode_buffer_size(dataset) if is_recording else None,
            )

        dt_s = time.perf_counter() - start_loop_t
        sleep_time_s = 1 / fps - dt_s
        precise_sleep(max(sleep_time_s, 0.0))


@parser.wrap()
def _record_with_display(cfg: RecordConfig):
    """Replace lerobot-record entry: install step UI then run original record body."""
    display = install_display(
        num_episodes=cfg.dataset.num_episodes,
        fps=cfg.dataset.fps,
        episode_time_s=cfg.dataset.episode_time_s,
        reset_time_s=cfg.dataset.reset_time_s,
    )
    display.set_phase(RecordPhase.CONNECTING, extra=f"root={cfg.dataset.root}")
    try:
        return _original_record_impl(cfg)
    finally:
        clear_display()


def _patch_record_entry() -> None:
    global _record_entry_patched
    if _record_entry_patched:
        return
    lr_record.record = _record_with_display  # type: ignore[assignment]
    _record_entry_patched = True


def _restore_record_entry() -> None:
    global _record_entry_patched
    if not _record_entry_patched:
        return
    lr_record.record = _original_record_entry
    _record_entry_patched = False


def cli() -> None:
    """Entry point: patch record hooks then delegate to lerobot-record main."""
    _patch_dataset_factory()
    _patch_record_entry()
    lr_record.record_loop = _record_loop_pico_ctl
    lerobot_utils.log_say = _patched_log_say
    try:
        from lerobot.scripts.lerobot_record import main as lerobot_record_main

        lerobot_record_main()
    finally:
        lr_record.record_loop = _original_record_loop
        lerobot_utils.log_say = _original_log_say
        _restore_record_entry()
        _restore_dataset_factory()


if __name__ == "__main__":
    cli()
