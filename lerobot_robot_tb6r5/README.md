# lerobot_robot_tb6r5

LeRobot plugin for the TB6-R5 robotic arm (6 joints + YS two-finger gripper).

## Install

```bash
# From the teleop repo root (editable install of the main package + plugin)
pip install -e .
pip install -e ./lerobot_robot_tb6r5
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
  --dataset.repo_id=local/tb6r5_live \
  --dataset.root=data/lerobot/tb6r5_live \
  --dataset.single_task="tb6r5 teleoperation" \
  --dataset.num_episodes=10 \
  --dataset.fps=20 \
  --dataset.streaming_encoding=true \
  --dataset.push_to_hub=false
```

`--robot.type=tb5r6` is accepted as an alias for `tb6r5`.

For PICO VR teleoperation, also install `lerobot_teleoperator_pico_tb6r5` and pass
`--teleop.type=pico_tb6r5`.

State/action layout (7-D, compatible with existing TB6R5 LeRobot datasets):

| Index | Unit | Description |
|-------|------|-------------|
| 0–5   | rad  | Joint 1–6 positions |
| 6     | mm   | Gripper opening distance (0 = closed) |
