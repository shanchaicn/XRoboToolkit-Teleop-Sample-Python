# lerobot_teleoperator_pico_ctl_tb6r5

PICO VR teleoperator for TB6-R5 using **`TB6R5TeleopController`** (same control path as `teleop_tb6r5_hardware.py`).

- Robot RPC is driven by the controller IK loop (not the split bridge in `pico_tb6r5`).
- Episode control uses **PICO buttons** (no keyboard required).
- Pair with `--robot.passive_mode=true` so the robot plugin only provides cameras + topic observation.

## Install

```bash
pip install -e ./lerobot_robot_tb6r5 --no-deps
pip install -e ./lerobot_teleoperator_pico_ctl_tb6r5 --no-deps
```

## Record

Use **`lerobot-record-pico-ctl`** (not plain `lerobot-record`) so PICO A/B map to episode events.
The CLI prints the current workflow step (recording / reset / saving) and live frame progress:

```bash
lerobot-record-pico-ctl \
  --robot.type=tb6r5 \
  --robot.passive_mode=true \
  --robot.robot_ip=192.168.11.11 \
  --robot.id=tb6r5_01 \
  --robot.cameras='{
    realsense_0: {type: intelrealsense, serial_number_or_name: "135522071053", width: 640, height: 480, fps: 30},
    realsense_1: {type: intelrealsense, serial_number_or_name: "347622071274", width: 640, height: 480, fps: 30}
  }' \
  --teleop.type=pico_ctl_tb6r5 \
  --teleop.robot_ip=192.168.11.11 \
  --teleop.control_rate_hz=50 \
  --teleop.id=pico_ctl_01 \
  --dataset.repo_id=local/tb6r5_ctl_live \
  --dataset.root=data/lerobot/tb6r5_ctl_live \
  --dataset.single_task="tb6r5 teleoperation" \
  --dataset.num_episodes=10 \
  --dataset.fps=30 \
  --dataset.streaming_encoding=true \
  --dataset.push_to_hub=false
```

## PICO controls

| Input | Action |
|-------|--------|
| `right_grip` (hold) | Move arm |
| `right_trigger` | Gripper (2 Hz, 5 mm deadband) |
| `right_axis_click` | Teleop arm gate (`require_joystick_arm=true`) |
| **B** | End / save episode + **回 home** |
| **A** | Discard episode & rerecord + **回 home** |
| **X** | 回 home |
| Esc / Ctrl+C 退出 | **关机前自动回 home**（`_shutdown_robot`） |

## vs `pico_tb6r5`

| | `pico_tb6r5` | `pico_ctl_tb6r5` |
|--|--------------|------------------|
| Control | Bridge → robot `send_action` | `TB6R5TeleopController` RPC |
| Episode | Keyboard `→` / `←` | PICO **B** / **A** |
| Robot plugin | Full RPC | `passive_mode=true` (obs only) |
