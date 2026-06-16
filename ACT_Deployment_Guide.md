# TB6-R5 ACT 模型部署指南（评估 → Dry-Run → 真机）

本文档说明如何把 LeRobot 训练出的 ACT 模型部署到 TB6-R5 机器人。
所有命令在项目根目录 `~/study/XRoboToolkit-Teleop-Sample-Python` 下运行，并已激活 `pico` 环境。

```bash
cd ~/study/XRoboToolkit-Teleop-Sample-Python
conda activate pico
```

> 注意：终端的当前目录必须是项目根目录，否则相对路径 `scripts/...` 会报
> `can't open file ... No such file or directory`。

---

## 0. 资源路径（已下载并验证）

| 资源 | 路径 | 说明 |
|---|---|---|
| 模型 040000（本地） | `model/act_tb6r5_yellow_yogurt/checkpoints/040000/pretrained_model` | 单独下载的最终 checkpoint |
| 模型 020000（HF 缓存） | `~/.cache/huggingface/hub/models--shanchai--act_tb6r5_yellow_yogurt/snapshots/<hash>/checkpoints/020000/pretrained_model` | `hf download` 下载的早期 checkpoint |
| v3 数据集 | `data/lerobot/tb6r5_yellow_yogurt_47_v3` | 47 集 / 51829 帧，repo_id `shanchai/tb6r5_yellow_yogurt_47_v3` |

`<hash>` 当前是 `ec25f5eaf99845e8dea1d66f8a39615c741b492b`。下面命令用 `$(ls -d ...)` 自动展开，无需手填。

> 关键：LeRobot `from_pretrained` 需要的是 **`pretrained_model` 目录**（含
> `config.json` / `model.safetensors` / `policy_preprocessor*.json` /
> `policy_postprocessor*.json`），不是单个 `.safetensors` 文件，也不是上一级
> `checkpoints/` 目录。

---

## 1. 模型输入 / 输出约定（必须与训练一致）

模型 `config.json` 要求的输入特征：

- `observation.state`：7 维 = `[q0..q5, gripper_mm]`
  - 前 6 维是机械臂关节角（弧度），真机时从机器人读取
  - 第 7 维 `gripper_mm` 是 YS 夹爪 **actual_pos 反馈（mm）**，0=全闭合，70=全开
- `observation.images.realsense_0`：RGB，480×640×3，相机序列号 `135522071053`
- `observation.images.realsense_1`：RGB，480×640×3，相机序列号 `327122073649`

输出动作 `action`：7 维 = `[q0..q5, gripper_mm]`

- 前 6 维是关节目标角（弧度）
- 第 7 维 `gripper_mm` 是**夹爪指令距离（mm）**，0=全闭合，`gripper_max_distance`=全开（默认 70）
- 真机通过 **SubLoop1** 合并下发：`JogAnyJ` + `MoveTwoFingersGripper`
- 遥操作采集：`right_trigger` 控制距离（按下=闭合，松开=张开），不再使用摇杆

**注意：** 旧数据集（13mm + 归一化 [0,1] + 无夹爪反馈）与新 pipeline 不兼容，需重新采集并训练。

模型权重已烘焙归一化统计，**推理和评估都无需依赖数据集**（数据集仅用于离线对比 MAE）。

---

## 2. 遥操作采集 LeRobot 数据集（lerobot_record 格式）

使用 PICO VR 遥操作，边控机器人边写入 LeRobot v3 数据集（与官方 `lerobot_record` 同 schema：
`state_0..6` / `action_0..6`、双相机 RGB 视频、无 depth）。

| 方案 | 入口 | Robot | Teleop | Episode 控制 | 适用场景 |
|------|------|-------|--------|--------------|----------|
| **A** | `teleop_tb6r5_hardware.py` | 脚本内置 | 脚本内置 | 手柄 **B/A** | 与硬件脚本完全一致、单进程采集 |
| **B1** | `lerobot-record` | `--robot.type=tb6r5` | `--teleop.type=pico_tb6r5` | 键盘 **→/←** | 官方 LeRobot 插件、Robot 发 RPC |
| **B2** | `lerobot-record-pico-ctl` | `--robot.type=tb6r5` + `passive_mode` | `--teleop.type=pico_ctl_tb6r5` | 手柄 **B/A** | 与方案 A 同控制器路径、官方 dataset 流程 |

方案 B 需先安装插件（`pico` 环境）：

```bash
pip install -e . --no-deps
pip install -e ./lerobot_robot_tb6r5 --no-deps
pip install -e ./lerobot_teleoperator_pico_tb6r5 --no-deps      # B1
pip install -e ./lerobot_teleoperator_pico_ctl_tb6r5 --no-deps # B2
```

完整 CLI：`lerobot-record --help` / `lerobot-record-pico-ctl --help`（`--robot.type` 含 `tb6r5`/`tb5r6`）。

---

### 2.1 方案 A：一体化脚本 `teleop_tb6r5_hardware.py`

输出目录默认 `data/lerobot/tb6r5_live`。

#### 推荐命令

```bash
python scripts/hardware/teleop_tb6r5_hardware.py \
  --robot-ip 192.168.11.11 \
  --teleop-mode placo_ik \
  --control-rate-hz 50 \
  --scale-factor 1.5 \
  --zone-ratio 0.00 \
  --gripper-max-d 70 \
  --enable-log-data \
  --enable-camera \
  --enable-lerobot-log \
  --lerobot-root data/lerobot/tb6r5_live \
  --lerobot-repo-id local/tb6r5_live \
  --lerobot-task "tb6r5 teleoperation" \
  --lerobot-streaming-encoding \
  --lerobot-overwrite \
  --lerobot-image-writer-processes 0 \
  --lerobot-image-writer-threads 4 \
  --lerobot-encoder-threads 2 \
  --no-enable-camera-depth
```

> `--lerobot-overwrite` 会删除已有 `data/lerobot/tb6r5_live` 后重建。续采请改
> `--lerobot-resume`（去掉 overwrite）。

### 2.2 VR 手柄操作

| 输入 | 作用 |
|------|------|
| **按住 `right_grip`** | 移动机械臂（松开自动 SubLoop1 exit） |
| **`right_trigger`** | 夹爪开合（0mm=闭合，`--gripper-max-d`=全开） |
| **`B`** | 开始 / 结束采集并保存当前 episode |
| **`A`** | 丢弃当前 episode 并回 home |
| **`X`** | 回 home |

默认需**按住 `right_grip` 才会下发 RPC**（含夹爪）。若希望不握把也能控夹爪，加
`--no-require-grip-to-send-commands`。

底层下发：臂 **50Hz**、夹爪 **2Hz**（SubLoop1 流式 `JogAnyJ` + `MoveTwoFingersGripper`）。

### 2.3 采集参数说明（`teleop_tb6r5_hardware.py --help`）

查看完整帮助：

```bash
python scripts/hardware/teleop_tb6r5_hardware.py --help
```

#### 机器人连接 / URDF

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--robot-urdf-path` | `assets/TB6-R5-RevA1-urdf/urdf/7260501-000000-001 TB6-R5-RevA1-urdf.urdf` | Placo IK 用 URDF 路径 |
| `--robot-ip` | `192.168.11.11` | 机器人 RPC / Topic 地址；`none` 为仅 Placo 可视化 |
| `--rpc-port` | `5868` | RPC 端口 |

#### 遥操作模式与 PICO 映射

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--teleop-mode` | `placo_ik` | `placo_ik`（PICO→Placo IK→JogAnyJ）或 `jog_any_c`（Topic robottarget→JogAnyC） |
| `--scale-factor` | `1.5` | PICO 手柄位姿增量缩放，越大越“跟手” |
| `--control-rate-hz` | `50` | IK / 控制主循环频率（Hz）；臂 RPC 默认同频 50Hz |
| `--require-grip-to-send-commands` | on | 仅按住 `right_grip` 时发 RPC；`--no-require-grip-to-send-commands` 关闭 |
| `--require-joystick-arm` | off | on 时需先按 `--teleop-arm-button` 才允许遥控 |
| `--teleop-arm-button` | `right_axis_click` | 遥控门控按键（配合 `require-joystick-arm`） |
| `--jog-any-c-preview` | off | on 时强制 sim 预览（不发 RPC），等价 `robot-ip=none` + `jog_any_c` |

**`--manipulator-config.right-hand.*`（右手操控链）**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--manipulator-config.right-hand.link-name` | `ee_Link` | 末端连杆名 |
| `--manipulator-config.right-hand.pose-source` | `right_controller` | PICO 位姿源 |
| `--manipulator-config.right-hand.control-trigger` | `right_grip` | 激活移动的握把键 |

#### `placo_ik` 关节模式

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--zone-ratio` | `0.05` | `JogAnyJ` zone_ratio |
| `--joint-vel` | `6.0` | `JogAnyJ` 关节速度 |
| `--joint-acc` | `3.0` | `JogAnyJ` 关节加速度 |
| `--joint-dec` | `3.0` | `JogAnyJ` 关节减速度 |
| `--sl-immediate` | off | SubLoop1 immediate 模式 |
| `--safe-tcp-z-min-m` | `0.05` | TCP Z 下限（m），`None` 关闭 |
| `--safe-tcp-z-max-m` | `0.65` | TCP Z 上限（m），`None` 关闭 |
| `--print-ik-tcp-pose` | off | 打印 IK 目标 TCP 位姿 |
| `--print-ik-tcp-pose-interval-s` | `0.2` | 打印间隔（秒） |
| `--visualize-placo` | off | 开启 Placo Meshcat 可视化 |

#### `jog_any_c` 笛卡尔模式

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--cartesian-max-step-pos-m` | `0.03` | 每步最大平移（m）；50Hz 下会被安全上限钳到约 `0.01` |
| `--cartesian-max-step-rot-rad` | `0.1` | 每步最大旋转（rad） |
| `--jog-any-c-position-only` | on | 仅跟踪位置（工具 Z 锁向世界 -Z） |
| `--jog-any-c-orientation-only` | off | 仅跟踪姿态（与 position-only 互斥） |
| `--jog-any-c-interrupt` | `off` | `on` / `off`，JogAnyC 中断模式 |
| `--jog-any-c-async-timeout-ms` | `5000000` | JogAnyC 异步超时（ms） |
| `--cartesian-vel` | `None` | 笛卡尔速度，默认由接口决定 |
| `--cartesian-acc` | `None` | 笛卡尔加速度 |
| `--cartesian-dec` | `None` | 笛卡尔减速度 |

#### 夹爪

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--gripper-trigger-name` | `right_trigger` | 夹爪映射键（按下=闭合，松开=张开） |
| `--gripper-max-d` | `70.0` | 最大张开距离（mm），须与训练一致 |
| `--two-fingers-gripper-interval` | `5.0` | `MoveTwoFingersGripper` interval |
| `--gripper-observation-default` | `0.0` | 无反馈时的夹爪观测默认值 |
| `--disable-gripper` | off | 禁用夹爪控制 |

夹爪 RPC 固定 **2Hz**；与上次指令差小于 `two_fingers_gripper_cmd_delta`（控制器内部默认 0.5mm）时不重复下发。

#### 相机

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--enable-camera` | on | 启用 RealSense（`enable-log-data` 时默认可开） |
| `--camera-width` | `640` | 图像宽 |
| `--camera-height` | `480` | 图像高 |
| `--camera-fps` | `30` | 相机帧率 |
| `--enable-camera-depth` | on | 预览/日志是否读 depth；数据集用 `--no-lerobot-include-depth` |
| `--enable-camera-compression` | on | 日志 JPG 压缩 |
| `--camera-jpg-quality` | `85` | JPG 质量 |

**`--camera-serial-dict.*`（逻辑名→序列号，非自动检测）**

| 参数 | 默认值 |
|------|--------|
| `--camera-serial-dict.realsense-0` | `135522071053` |
| `--camera-serial-dict.realsense-1` | `327122073649` |

换机查序列号：`rs-enumerate-devices | grep "Serial Number"`。

#### 数据记录（pkl / LeRobot）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--enable-log-data` | on | 启用数据记录线程 |
| `--log-dir` | `logs/tb6r5` | pkl 分段日志目录（非 LeRobot 时） |
| `--log-freq` | `50` | 采集帧率（Hz），建议与 `--control-rate-hz` 一致 |
| `--log-joint-count` | `6` | 日志关节维数 |

**LeRobot v3 在线写入（与 `lerobot_record` 同 schema）**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--enable-lerobot-log` | off | 直接写 LeRobot v3（推荐采集时打开） |
| `--lerobot-root` | `data/lerobot/tb6r5_live` | 数据集根目录 |
| `--lerobot-repo-id` | `local/tb6r5_live` | repo id，须与 `meta/info.json` 一致 |
| `--lerobot-task` | `tb6r5 teleoperation` | 每帧 `task` 字段 |
| `--lerobot-streaming-encoding` | on | 流式 AV1 编码，B 键保存更快 |
| `--lerobot-overwrite` | off | 清空已有 root 后新建 |
| `--lerobot-resume` | off | 续采已有数据集（与 overwrite 二选一） |
| `--lerobot-image-writer-processes` | `0` | PNG 写盘进程数；流式编码时一般为 0 |
| `--lerobot-image-writer-threads` | `4` | 每相机写线程数 |
| `--lerobot-encoder-threads` | `2` | 视频编码线程数 |
| `--lerobot-include-depth` | off | 数据集中写入 depth（训练一般不需要） |

#### 跟手调参速查

| 现象 | 可尝试 |
|------|--------|
| 臂动得慢、跟不上手 | `--scale-factor 2.0`、`--joint-vel 6~8`（默认 vel/acc/dec=`6/3/3`） |
| 录制卡顿拖累体感 | `--no-enable-camera-depth`、`--lerobot-encoder-threads 2`、减相机路数 |
| Z 方向被挡住 | 检查 `--safe-tcp-z-min-m` / `--safe-tcp-z-max-m` |
| 必须握把才动 | 默认 `--require-grip-to-send-commands`；松开 `right_grip` 即停 |

### 2.4 方案 B：LeRobot 官方 `lerobot-record` + 插件

架构：**Robot 插件**负责相机 + 状态观测（+ 可选 RPC）；**Teleop 插件**负责 PICO→动作。
数据集参数使用 LeRobot 标准 `--dataset.*`（与官方 `lerobot-record` 一致）。

#### 2.4.1 方案 B1：`pico_tb6r5` + `tb6r5`（Robot 发 RPC）

Teleop 经 Bridge 算 Placo IK，由 **robot `send_action`** 下发 SubLoop1；与方案 A 控制逻辑相近，
但录制循环在 `lerobot-record` 内（相机读图与编码可能略占 CPU）。

```bash
lerobot-record \
  --robot.type=tb6r5 \
  --robot.robot_ip=192.168.11.11 \
  --robot.id=tb6r5_01 \
  --robot.arm_rpc_rate_hz=50 \
  --robot.gripper_rpc_rate_hz=2 \
  --robot.gripper_cmd_delta=5.0 \
  --robot.max_joint_step_rad=0.05 \
  --robot.cameras='{
    realsense_0: {type: intelrealsense, serial_number_or_name: "135522071053", width: 640, height: 480, fps: 30},
    realsense_1: {type: intelrealsense, serial_number_or_name: "347622071274", width: 640, height: 480, fps: 30}
  }' \
  --teleop.type=pico_tb6r5 \
  --teleop.robot_ip=192.168.11.11 \
  --teleop.control_rate_hz=50 \
  --teleop.scale_factor=1.5 \
  --teleop.id=pico_01 \
  --dataset.repo_id=local/tb6r5_live \
  --dataset.root=data/lerobot/tb6r5_live \
  --dataset.single_task="tb6r5 teleoperation" \
  --dataset.num_episodes=10 \
  --dataset.fps=30 \
  --dataset.streaming_encoding=true \
  --dataset.push_to_hub=false
```

**Episode 控制（键盘）**：`→` 结束并保存 episode；`←` 丢弃并重录；`Esc` 停止录制。

**`--robot.*`（`lerobot_robot_tb6r5`）**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--robot.type` | — | `tb6r5` 或别名 `tb5r6` |
| `--robot.id` | — | 实例 ID |
| `--robot.robot_ip` | `192.168.11.11` | RPC + Topic 地址 |
| `--robot.rpc_port` | `5868` | RPC 端口 |
| `--robot.passive_mode` | `false` | B1 保持 `false`（Robot 发 RPC） |
| `--robot.enable_topic` | `true` | Topic 关节/夹爪反馈 |
| `--robot.topic_wait_timeout_s` | `5.0` | 等待 Topic 超时（秒） |
| `--robot.zone_ratio` | `0.0` | `JogAnyJ` zone_ratio |
| `--robot.joint_vel` / `acc` / `dec` | `6.0` / `3.0` / `3.0` | `JogAnyJ` 运动参数 |
| `--robot.subloop1_immediate` | `false` | SubLoop1 immediate |
| `--robot.gripper_max_d` | `70.0` | 夹爪最大张开（mm） |
| `--robot.gripper_interval` | `25.0` | `MoveTwoFingersGripper` interval |
| `--robot.gripper_cmd_delta` | `5.0` | 与上次夹爪**输入**差 &lt; 此值（mm）不发送 |
| `--robot.arm_rpc_rate_hz` | `50.0` | 臂 RPC 频率上限 |
| `--robot.gripper_rpc_rate_hz` | `2.0` | 夹爪 RPC 频率上限 |
| `--robot.arm_cmd_eps_rad` | `0.001` | 臂目标与反馈接近时跳过 RPC |
| `--robot.max_joint_step_rad` | `0.03` | 每步关节增量钳位（rad）；`null` 关闭 |
| `--robot.cameras` | `{}` | RealSense 字典，见上例 |

**`--teleop.*`（`lerobot_teleoperator_pico_tb6r5`）**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--teleop.type` | — | `pico_tb6r5` |
| `--teleop.id` | — | 实例 ID |
| `--teleop.robot_ip` | `192.168.11.11` | Topic 反馈 IP（与 robot 同机） |
| `--teleop.topic_wait_timeout_s` | `5.0` | Topic 等待超时 |
| `--teleop.robot_urdf_path` | `None` | 默认仓库内 TB6-R5 URDF |
| `--teleop.scale_factor` | `1.5` | PICO 位姿缩放 |
| `--teleop.control_rate_hz` | `50` | 与 `--dataset.fps` 对齐可减少延迟感 |
| `--teleop.require_joystick_arm` | `false` | `true` 时需 `right_axis_click` 门控 |
| `--teleop.gripper_max_d` | `70.0` | 与 robot 一致 |
| `--teleop.safe_tcp_z_min_m` / `max_m` | `0.05` / `0.65` | TCP Z 安全区 |

未按 `right_grip` 时 **不发任何 RPC**（共享 command gate）。

---

#### 2.4.2 方案 B2：`pico_ctl_tb6r5` + `tb6r5`（Controller 发 RPC，推荐）

与 **方案 A** 相同：后台 `TB6R5TeleopController` IK 线程直接 RPC；Robot 插件仅
**Topic + 相机**（`passive_mode=true`），避免双通道抢控制权。Episode 用手柄 **B/A**，
需使用包装命令 **`lerobot-record-pico-ctl`**（终端会显示当前步骤：录制 / Reset / 保存，及帧数进度）。

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
  --teleop.rpc_port=5868 \
  --teleop.control_rate_hz=50 \
  --teleop.scale_factor=1.5 \
  --teleop.joint_vel=6.0 \
  --teleop.joint_acc=3.0 \
  --teleop.joint_dec=3.0 \
  --teleop.id=pico_ctl_01 \
  --dataset.repo_id=local/tb6r5_ctl_live \
  --dataset.root=data/lerobot/tb6r5_ctl_live \
  --dataset.single_task="tb6r5 teleoperation" \
  --dataset.num_episodes=10 \
  --dataset.fps=30 \
  --dataset.streaming_encoding=true \
  --dataset.push_to_hub=false
```

**Episode 控制（手柄）**：**B** 结束并保存 + 回 home；**A** 丢弃并重录 + 回 home；**X** 回 home；
**Esc/Ctrl+C** 退出前自动回 home。

**`--robot.*`（B2 常用子集）**

| 参数 | B2 推荐 | 说明 |
|------|---------|------|
| `--robot.passive_mode` | **`true`** | 必须开启；不发 RPC，仅观测与相机 |
| `--robot.robot_ip` | `192.168.11.11` | 仅 Topic（无 RPC init） |
| `--robot.cameras` | 见上 | 逻辑名须为 `realsense_0` / `realsense_1` |
| 其余 `arm_rpc_*` / `gripper_*` | — | passive 模式下由 teleop 控制，可忽略 |

**`--teleop.*`（`lerobot_teleoperator_pico_ctl_tb6r5`）**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--teleop.type` | — | `pico_ctl_tb6r5` |
| `--teleop.robot_ip` | `192.168.11.11` | **RPC + Topic**（真机控制） |
| `--teleop.rpc_port` | `5868` | RPC 端口 |
| `--teleop.teleop_mode` | `placo_ik` | `placo_ik` 或 `jog_any_c` |
| `--teleop.scale_factor` | `1.5` | PICO 位姿缩放 |
| `--teleop.control_rate_hz` | `50` | IK/RPC 主循环（Hz） |
| `--teleop.require_grip_to_send_commands` | `true` | 仅 `right_grip` 时发运动 RPC |
| `--teleop.require_joystick_arm` | `false` | `true` 时需 `right_axis_click` 开门控 |
| `--teleop.gripper_max_d` | `70.0` | 夹爪最大张开（mm） |
| `--teleop.two_fingers_gripper_interval` | `25.0` | 夹爪 interval |
| `--teleop.two_fingers_gripper_cmd_delta` | `5.0` | 夹爪输入死区（mm） |
| `--teleop.zone_ratio` | `0.0` | `JogAnyJ` zone_ratio |
| `--teleop.joint_vel` / `acc` / `dec` | `6.0` / `3.0` / `3.0` | `JogAnyJ` 参数 |
| `--teleop.subloop1_immediate` | `false` | SubLoop1 immediate |
| `--teleop.safe_tcp_z_min_m` / `max_m` | `0.05` / `0.65` | TCP Z 限位 |
| `--teleop.visualize_placo` | `false` | Placo Meshcat |
| `--teleop.robot_urdf_path` | `None` | 默认仓库 URDF |

臂 **50Hz**、夹爪 **2Hz**；`right_trigger` 松开映射为张开（须仍按住 `right_grip` 才发令）。

---

#### 2.4.3 方案 B 共用：`--dataset.*` 与其它 LeRobot 参数

| 参数 | 推荐 | 说明 |
|------|------|------|
| `--dataset.repo_id` | `local/tb6r5_live` | 本地 repo id |
| `--dataset.root` | `data/lerobot/tb6r5_live` | 数据集根目录 |
| `--dataset.single_task` | `tb6r5 teleoperation` | 任务描述 |
| `--dataset.num_episodes` | `10` | episode 数量 |
| `--dataset.fps` | `30` | 与相机 fps 一致；B2 控制环可 50Hz 独立运行 |
| `--dataset.episode_time_s` | `60` | 每集最长录制时间 |
| `--dataset.reset_time_s` | `60` | 集间重置等待 |
| `--dataset.streaming_encoding` | `true` | 流式 AV1，保存更快 |
| `--dataset.push_to_hub` | `false` | 是否上传 HF |
| `--dataset.resume` | `false` | 续采已有数据集（勿对空壳目录使用） |
| `--display_data` | `false` | Rerun 实时可视化（占 CPU） |

续采：加 `--dataset.resume=true`，且 **不要** 对仅有 `meta/info.json` 的空目录 resume。

---

#### 2.4.4 方案对比速查

| | 方案 A | B1 `pico_tb6r5` | B2 `pico_ctl_tb6r5` |
|--|--------|-----------------|---------------------|
| 命令 | `teleop_tb6r5_hardware.py` | `lerobot-record` | `lerobot-record-pico-ctl` |
| 谁发 RPC | 脚本 Controller | Robot 插件 | Teleop Controller |
| Episode | B / A | → / ← 键盘 | B / A |
| 跟手 | 较好 | 受录制 fps 影响 | 较好（IK 独立线程） |
| 插件 | 无 | robot + teleop | robot + teleop |

---

### 2.5 可视化刚采集的数据（tb6r5_live）

本机 Rerun 查看（episode 从 `0` 起，按已保存数量递增）：

```bash
lerobot-dataset-viz \
  --repo-id local/tb6r5_live \
  --root data/lerobot/tb6r5_live \
  --episode-index 1
```

查看第 0 集把 `--episode-index` 改为 `0` 即可。

SSH / 无显示器：先导出 `.rrd` 再本机打开（见第 3.3 节），把 `--repo-id` / `--root` 换成
`local/tb6r5_live` / `data/lerobot/tb6r5_live`。

---

## 3. 用 Rerun 可视化本地 LeRobot 数据集

把 HF 数据集下载到本地后，可用 LeRobot 自带的 `lerobot-dataset-viz` 在 **Rerun** 里逐帧查看
相机图像、`observation.state`、`action` 等训练数据。

### 3.1 下载数据集到本地（若尚未下载）

```bash
huggingface-cli download shanchai/tb6r5_yellow_yogurt_47_v3 \
  --repo-type dataset \
  --local-dir data/lerobot/tb6r5_yellow_yogurt_47_v3
```

本地目录需包含 `meta/info.json`、`data/`、`videos/` 等 LeRobot v3 结构。

### 3.2 本机直接打开 Rerun 查看器（推荐）

在项目根目录、`pico` 环境下运行：

```bash
lerobot-dataset-viz \
  --repo-id shanchai/tb6r5_yellow_yogurt_47_v3 \
  --root data/lerobot/tb6r5_yellow_yogurt_47_v3 \
  --episode-index 0
```

参数说明：

- `--repo-id`：数据集的 HF 名称，需与 `meta/info.json` 中一致
- `--root`：本地数据集根目录（**必须指向你下载/转换后的文件夹**）
- `--episode-index`：要看的 episode 编号（本数据集共 47 个，可用 `0`–`46`）

成功后会自动弹出 Rerun 窗口。左侧时间轴可拖动，可看到：

- `observation.images.realsense_0` / `realsense_1`：两台相机 RGB
- `observation.state`：7 维状态（6 关节 + 夹爪）
- `action`：7 维动作

查看其他 episode，改 `--episode-index` 即可，例如：

```bash
lerobot-dataset-viz \
  --repo-id shanchai/tb6r5_yellow_yogurt_47_v3 \
  --root data/lerobot/tb6r5_yellow_yogurt_47_v3 \
  --episode-index 5
```

### 3.3 无图形界面 / SSH 远程：先导出 `.rrd` 再本地查看

在服务器上生成 Rerun 录制文件：

```bash
lerobot-dataset-viz \
  --repo-id shanchai/tb6r5_yellow_yogurt_47_v3 \
  --root data/lerobot/tb6r5_yellow_yogurt_47_v3 \
  --episode-index 0 \
  --save 1 \
  --output-dir outputs/rerun
```

会在 `outputs/rerun/` 下生成 `.rrd` 文件。拷到本机后：

```bash
rerun outputs/rerun/lerobot_tb6r5_yellow_yogurt_47_v3_episode_0.rrd
```

### 3.4 远程机器流式查看（distant 模式）

数据在远程 GPU/机器人电脑上，本机用 Rerun 客户端连接：

```bash
# 远程机器
lerobot-dataset-viz \
  --repo-id shanchai/tb6r5_yellow_yogurt_47_v3 \
  --root data/lerobot/tb6r5_yellow_yogurt_47_v3 \
  --episode-index 0 \
  --mode distant \
  --grpc-port 9876

# 本机（把 REMOTE_IP 换成远程 IP）
rerun rerun+http://REMOTE_IP:9876/proxy
```

### 3.5 自己转换的本地数据集

若用 `convert_tb6r5_pkl_to_lerobot_v3.py` 转出的数据集，把 `--repo-id` 和 `--root` 换成你创建时
写的值即可，例如：

```bash
lerobot-dataset-viz \
  --repo-id local/tb6r5-20260602-pnp \
  --root data/lerobot/tb6r5-20260602-pnp \
  --episode-index 0
```

> `repo_id` 必须与转换时 `--repo-id` 一致；`root` 指向数据集根目录（含 `meta/` 的那一层）。

---

## 4. 第一步：离线评估（验证模型预测精度）

在真机前，先用数据集对比"模型预测动作"与"真实动作"的 MAE。

```bash
# 用 040000（本地）评估
python scripts/misc/eval_act_on_lerobot_tb6r5.py \
  --policy-path model/act_tb6r5_yellow_yogurt/checkpoints/040000/pretrained_model \
  --dataset-root data/lerobot/tb6r5_yellow_yogurt_47_v3 \
  --repo-id shanchai/tb6r5_yellow_yogurt_47_v3 \
  --device cuda \
  --max-samples 200 \
  --stride 50
```

```bash
# 用 020000（HF 缓存）评估
CKPT=$(ls -d ~/.cache/huggingface/hub/models--shanchai--act_tb6r5_yellow_yogurt/snapshots/*/checkpoints/020000/pretrained_model)
python scripts/misc/eval_act_on_lerobot_tb6r5.py \
  --policy-path "$CKPT" \
  --dataset-root data/lerobot/tb6r5_yellow_yogurt_47_v3 \
  --repo-id shanchai/tb6r5_yellow_yogurt_47_v3 \
  --device cuda \
  --max-samples 200 \
  --stride 50
```

参数：

- `--max-samples`：评估样本数上限
- `--stride`：每隔多少帧取一个样本（脚本会对每个样本 `policy.reset()`，保证逐样本独立）

参考结果（020000，30 样本）：`MAE(all) ≈ 0.0117`，各关节 MAE ≈ 0.006–0.020 rad（约 0.3–1.2°），
夹爪 MAE ≈ 0.012。**MAE 越低越好**；若某维 MAE 明显偏大，说明该自由度学习不充分。

---

## 5. 第二步：Dry-Run（不发指令，只看预测）

`--dry-run` 不连接机器人、不发任何指令，只打印模型预测，用于上真机前的安全确认。

### 5a. 纯流水线测试（无相机，喂黑图）

仅验证模型能加载、循环能跑（预测无意义，因为图像是黑的）：

```bash
python scripts/hardware/policy_infer_tb6r5_act.py \
  --robot-ip 0.0.0.0 \
  --policy-path model/act_tb6r5_yellow_yogurt/checkpoints/040000/pretrained_model \
  --device cuda \
  --fps 5 \
  --no-camera \
  --dry-run \
  --print-every 0.2
```

### 5b. 带真实相机的 Dry-Run（推荐）

接好两台 RealSense 后，喂真实图像，观察预测是否随场景合理变化：

```bash
python scripts/hardware/policy_infer_tb6r5_act.py \
  --robot-ip 192.168.11.11 \
  --policy-path model/act_tb6r5_yellow_yogurt/checkpoints/040000/pretrained_model \
  --device cuda \
  --fps 20 \
  --gripper-max-distance 70 \
  --gripper-interval 25 \
  --dry-run \
  --print-every 0.5
```

打印含义：

- `q_cur`：当前关节角（dry-run 下机器人未连接，恒为 0）
- `q_tgt`：模型预测的关节目标
- `q_cmd`：经每步限速钳制后的实际下发值（`--joint-step-max-rad` 默认 0.03 rad/步）
- `cmd` / `obs`：夹爪指令/反馈距离（mm，0=闭合，70=张开）

确认 `q_tgt` 落在合理关节范围、夹爪 mm 在 0–70 之间，再上真机。

---

## 6. 第三步：真机运行

**去掉 `--dry-run`** 即开始向机器人下发指令。务必先完成 dry-run 验证，并保证周围安全、急停可及。

```bash
python scripts/hardware/policy_infer_tb6r5_act.py \
  --robot-ip 192.168.11.11 \
  --policy-path model/act_tb6r5_yellow_yogurt/checkpoints/040000/pretrained_model \
  --device cuda \
  --fps 20 \
  --joint-step-max-rad 0.03 \
  --gripper-max-distance 70 \
  --gripper-interval 25
```

真机参数：

- `--robot-ip` / `--rpc-port`：机器人地址（RPC 端口默认 5868）
- `--fps`：控制频率，首次建议先低（如 10）观察后再提高
- `--joint-step-max-rad`：每步关节最大变化量，越小越慢越安全（默认 0.03 rad）
- `--gripper-max-distance`：夹爪最大张开距离（mm），必须与训练一致（YS 默认 70）
- `--gripper-interval`：`MoveTwoFingersGripper` 的 interval（默认 25）
- `--gripper-cmd-delta`：夹爪距离变化小于该值（mm）时跳过 SubLoop1 重发（默认 0.5）

相机相关（一般用默认即可）：

- `--camera-serials`：覆盖默认序列号映射，格式
  `realsense_0=135522071053,realsense_1=327122073649`
- `--camera-width/height/fps`：默认 640×480×30
- `--no-camera`：跳过相机喂黑图（**仅调试**，真机请勿使用）

按 `Ctrl+C` 停止，脚本会自动停相机线程并 `disable()` 机器人。

---

## 7. RPC / SubLoop1 硬件测试脚本

上 ACT 真机前，可用下列脚本单独验证 TB6-R5 的 RPC 下发是否正常。三者共用同一套测试轨迹：

- 关节 1/2/4/5/6 固定：`(15, -100, -80, -90, -45)` deg
- **关节 3**：85°~95° 正弦摆动（`--period-s` 默认 2s）
- **夹爪**：10~50 mm 正弦摆动（0=闭合，70=全开）

默认 `--robot-ip 192.168.11.11`、`--rpc-port 5868`。所有脚本支持 `--dry-run`（只打印命令不发 RPC）、
`--duration-s`（默认 30s，0=不限时）、`Ctrl+C` 优雅退出。完整参数见各脚本 `--help`。

### 7.1 经 TB6R5Interface（与遥操作同路径）

走 `TB6R5Interface.set_joint_positions_with_gripper()`，与遥操作 / ACT 推理的下发路径一致
（SubLoop1 流式 `JogAnyJ` + `MoveTwoFingersGripper`）。

```bash
# 默认 20Hz，臂+夹爪合并 SubLoop1
python scripts/hardware/test_subloop1_joganyj_gripper.py --robot-ip 192.168.11.11

# 调频率与时长
python scripts/hardware/test_subloop1_joganyj_gripper.py \
  --robot-ip 192.168.11.11 --rate-hz 20 --duration-s 60

# 仅打印 SubLoop1 命令
python scripts/hardware/test_subloop1_joganyj_gripper.py --dry-run
```

常用参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--rate-hz` | 20 | SubLoop1 发送频率 |
| `--sl-immediate` | off | SubLoop1 首帧是否带 `--immediate` |
| `--gripper-cmd-delta` | 0.1 | 夹爪变化小于该值（mm）时跳过重发 |
| `--always-send-gripper` | on | 每帧强制带夹爪（忽略 delta 节流） |
| `--shutdown-ctrlc-pause-s` | 10 | Ctrl+C 后等待队列排空再 exit |

### 7.2 直连 RPC（SubLoop / SubLoop1，双模型 `||`）

不经过遥操作控制器，直接发 `{SubLoop1 ...}` 或 `{SubLoop ...}` 双模型指令
（`JogAnyJ||MoveTwoFingersGripper` 或单侧 `NotRunExecute`）。

```bash
# SubLoop1 流式：臂 20Hz + 夹爪 2Hz（默认 mode=both）
python scripts/hardware/test_subloop1_direct_rpc.py \
  --robot-ip 192.168.11.11 --mode both --arm-rate-hz 20 --gripper-rate-hz 2

# 仅臂 / 仅夹爪
python scripts/hardware/test_subloop1_direct_rpc.py --mode arm --arm-rate-hz 20
python scripts/hardware/test_subloop1_direct_rpc.py --mode gripper --gripper-rate-hz 2

# SubLoop 同步（每帧 CallAwait 阻塞，无流式会话）
python scripts/hardware/test_subloop1_direct_rpc.py \
  --subloop-cmd subloop --mode arm --arm-rate-hz 5

python scripts/hardware/test_subloop1_direct_rpc.py --dry-run --subloop-cmd subloop1
```

常用参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--mode` | both | `both` / `arm` / `gripper` |
| `--subloop-cmd` | subloop1 | `subloop1`（流式会话）或 `subloop`（同步） |
| `--arm-rate-hz` | 20 | 臂指令频率；`mode=both` 时与夹爪分频 |
| `--gripper-rate-hz` | 2 | 夹爪指令频率 |
| `--sl-immediate` | off | SubLoop1 首帧 immediate |
| `--shutdown-ctrlc-pause-s` | 10 | Ctrl+C 后暂停再 SubLoop1 exit + Disable |

### 7.3 单模型直连 RPC（无 SubLoop / 无 `||`）

每条 RPC 只发一个模型指令，用于排查双模型 / SubLoop 相关问题：

- init：`{Enable}` → `{Start}`（**不用** `{Enable||NotRunExecute}`）
- arm：`{JogAnyJ ...}`
- gripper：`{MoveTwoFingersGripper ...}`
- exit：arm 模式先发 `{Stop}`，再 `{Disable}`

```bash
# 仅臂，async 20Hz
python scripts/hardware/test_direct_single_rpc.py --mode arm --robot-ip 192.168.11.11

# 仅夹爪，默认 2Hz
python scripts/hardware/test_direct_single_rpc.py --mode gripper --robot-ip 192.168.11.11

# sync 阻塞发送
python scripts/hardware/test_direct_single_rpc.py --mode arm --transport sync --rate-hz 5

python scripts/hardware/test_direct_single_rpc.py --dry-run --mode gripper
```

常用参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--mode` | arm | `arm` 或 `gripper` |
| `--transport` | async | `async`（Call）或 `sync`（CallAwait） |
| `--rate-hz` | arm=20 / gripper=2 | 发送频率 |
| `--shutdown-ctrlc-pause-s` | 10 | Ctrl+C 后等待 pending 再 Stop/Disable |

### 7.4 选用建议

| 场景 | 推荐脚本 |
|---|---|
| 验证遥操作 / ACT 同款 SubLoop1 路径 | `test_subloop1_joganyj_gripper.py` |
| 调试臂/夹爪分频、SubLoop vs SubLoop1 | `test_subloop1_direct_rpc.py` |
| 排查 init / 单模型 RPC 连通性 | `test_direct_single_rpc.py` |

> 测试前请清空工作空间、急停在手边。若 `{Enable}` 后 `safe_recv failed`，检查 IP、
> 防火墙及控制器是否已被其他客户端占用。

---

## 8. 安全清单

- [ ] 先离线评估，确认 MAE 合理
- [ ] 再 dry-run（最好带真实相机），确认 `q_tgt` / 夹爪 mm 合理
- [ ] 真机首次用低 `--fps`（如 10）+ 默认 `--joint-step-max-rad 0.03`
- [ ] 机器人工作空间清空，急停在手边
- [ ] 相机序列号、`--gripper-max-distance` 与训练数据完全一致

---

## 9. 常见问题

- **`can't open file ... scripts/...`**：当前目录不对，请 `cd` 回项目根目录。
- **`Parquet magic bytes not found in footer`**：数据集损坏（转换中断），换用
  `data/lerobot/tb6r5_yellow_yogurt_47_v3`。
- **`BackwardCompatibilityError ... 2.1 format`**：该数据集是 v2.1，本环境 lerobot 是
  v3.0。请用已转好的 v3 数据集，或 `python -m lerobot.datasets.v30.convert_dataset_v21_to_v30` 转换。
- **HF 首次连接卡住**：偶发超时，重试即可；必要时
  `export HF_HUB_DOWNLOAD_TIMEOUT=30`，或临时 `export HF_ENDPOINT=https://hf-mirror.com`。
- **相机超时 / 不出图**：检查 RealSense 序列号、USB 带宽；脚本会等待两台相机就绪，
  超时会告警并用上一帧/黑图兜底。
- **Rerun 窗口不弹出**：SSH/无显示器环境请用 `--save 1` 导出 `.rrd`，本机执行 `rerun xxx.rrd`。
- **`lerobot-dataset-viz: command not found`**：确认已 `conda activate pico`，且 lerobot 已安装
  （`pip show lerobot`）。
- **数据集版本报错**：Rerun 可视化同样需要 v3 格式数据集；v2.1 需先转换（见上文
  `BackwardCompatibilityError`）。
- **B 键采集报错 `Missing features ... depth`**：数据集中启用了 depth 但帧里缺图；用
  `--no-enable-camera-depth` 且不要加 `--lerobot-include-depth`（见第 2.1 节推荐命令）。
- **B 键采集后线程退出**：旧版 `task` 参数不兼容已修复；请 `--lerobot-overwrite` 重采，
  确保 `meta/info.json` 中特征名为 `state_0..6` / `action_0..6`。

---

## 10. 相关文件

- 遥操作采集（方案 A）：`scripts/hardware/teleop_tb6r5_hardware.py`（见第 2.1 节）
- 真机推理：`scripts/hardware/policy_infer_tb6r5_act.py`
- 离线评估：`scripts/misc/eval_act_on_lerobot_tb6r5.py`
- SubLoop1 接口测试：`scripts/hardware/test_subloop1_joganyj_gripper.py`（见第 7.1 节）
- SubLoop 直连 RPC 测试：`scripts/hardware/test_subloop1_direct_rpc.py`（见第 7.2 节）
- 单模型直连 RPC 测试：`scripts/hardware/test_direct_single_rpc.py`（见第 7.3 节）
- 数据集 Rerun 可视化：`lerobot-dataset-viz`（LeRobot 自带，见第 2.5 / 第 3 节）
- LeRobot 插件采集：`lerobot-record` + `pico_tb6r5`，或 `lerobot-record-pico-ctl` + `pico_ctl_tb6r5`（见第 2.4 节）
- pkl 曲线图：`scripts/misc/plot_tb6r5_pkl_curves.py`
- pkl → LeRobot v3：`scripts/misc/convert_tb6r5_pkl_to_lerobot_v3.py`
