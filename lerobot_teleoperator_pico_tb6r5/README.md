# lerobot_teleoperator_pico_tb6r5

PICO VR teleoperator plugin for TB6-R5, compatible with `lerobot-record`.

## Install

```bash
pip install -e .
pip install -e ./lerobot_robot_tb6r5
pip install -e ./lerobot_teleoperator_pico_tb6r5
```

## Record

```bash
lerobot-record \
  --robot.type=tb6r5 \
  --robot.robot_ip=192.168.11.11 \
  --robot.id=tb6r5_01 \
  --robot.cameras='{
    realsense_0: {type: intelrealsense, serial_number_or_name: "135522071053", width: 640, height: 480, fps: 30},
    realsense_1: {type: intelrealsense, serial_number_or_name: "327122073649", width: 640, height: 480, fps: 30}
  }' \
  --teleop.type=pico_tb6r5 \
  --teleop.robot_ip=192.168.11.11 \
  --teleop.id=pico_01 \
  --dataset.repo_id=local/tb6r5_live \
  --dataset.root=data/lerobot/tb6r5_live \
  --dataset.single_task="tb6r5 teleoperation" \
  --dataset.num_episodes=10 \
  --dataset.fps=20 \
  --dataset.streaming_encoding=true \
  --dataset.push_to_hub=false
```

## PICO controls

| Input | Action |
|-------|--------|
| `right_grip` (hold) | Move arm (Placo IK) |
| `right_trigger` | Gripper open/close |
| `right_axis_click` | Arm teleop on/off (when `require_joystick_arm=true`) |

Episode control uses LeRobot keyboard shortcuts: `→` end episode, `←` rerecord, `Esc` stop.
