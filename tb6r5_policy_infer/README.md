# tb6r5_policy_infer

独立于 `xrobotoolkit_teleop` 遥操作框架的 TB6-R5 **LeRobot 策略推理与离线评估**包。

支持 LeRobot 训练的 **ACT**、**Diffusion** 等策略（由 checkpoint 内 `config.json` 的 `type` 字段决定）。

**真机推理数据接口（观测/动作/相机/RPC）**：[docs/ACT_Policy_Infer_Interface.md](../docs/ACT_Policy_Infer_Interface.md)

---

## 快速开始（推荐流程）

```text
1. 离线评估（验证 checkpoint + 数据集对齐）
      tb6r5-policy-eval ...

2. Dry-run（验证相机 + 推理输出，不发 RPC）
      tb6r5-policy-infer ... --dry-run

3. 真机（去掉 --dry-run，首次建议 --fps 10）
      tb6r5-policy-infer ...
```

---

## 安装

`xrobotoolkit_teleop` 不在 PyPI，不能作为 pip 自动依赖；请在本仓库根目录分步安装：

```bash
# 仅离线评估：只装推理包即可
pip install -e ./tb6r5_policy_infer

# 真机推理：还需主包（TB6-R5 RPC）；RealSense 模式还需 pyrealsense2
pip install -e . --no-deps
pip install -e ./tb6r5_policy_infer --no-deps
```

与 `lerobot_robot_tb6r5` 相同，主包用 `--no-deps` 避免拉取 PyPI 上不存在的包。

### TER30 / 新机器最小安装

在**仓库根目录**（含 `pyproject.toml` 与 `xrobotoolkit_teleop/` 的目录）执行：

```bash
cd ~/study/XRoboToolkit-Teleop-Sample-Python
conda activate pico   # 或你的 venv

# 1. 推理包 + LeRobot 栈（按下文 lerobot 0.5.x 对齐 transformers）
pip install "lerobot==0.5.1" "transformers==5.3.0" "huggingface-hub>=1.16,<2.0"
pip install -e ./tb6r5_policy_infer

# 2. 主包（提供 xrobotoolkit_teleop，真机 RPC/RealSense 需要）
pip install -e . --no-deps

# 3. 真机常用依赖（勿 pip install -e . 不带 --no-deps，会拉 mujoco/placo 等仿真大包）
pip install pyrealsense2 tyro numpy torch opencv-python
```

验证主包已装上：

```bash
python -c "import xrobotoolkit_teleop; print(xrobotoolkit_teleop.__file__)"
```

**仅测策略、不连机器人/相机**（可不装主包）：

```bash
tb6r5-policy-infer --robot-ip 192.168.11.11 --policy-path model/... \
  --dry-run --no-camera
```

连真机或开 RealSense 则必须完成步骤 2，并拷贝 ARM 版 `rpc.so` / `topic.so`（见下文「机器人 SDK」）。

### 远程机器更新

```bash
cd XRoboToolkit-Teleop-Sample-Python
git pull origin YS          # 或你的分支名
pip install -e . --no-deps
pip install -e ./tb6r5_policy_infer --no-deps
```

模型权重（`pretrained_model/`）需单独 `rsync`，不在 git 里。

---

## 真机推理（`tb6r5-policy-infer`）

### Checkpoint 路径

`--policy-path` 须指向 **`pretrained_model` 目录**（含 `config.json`、`model.safetensors`、processor json），不是上一级 `checkpoints/` 目录：

```text
outputs/train/tb6r5_rings_P05/checkpoints/100000/pretrained_model   ✓
model/act/080000/pretrained_model                                   ✓
outputs/train/.../checkpoints/100000/                               ✗
```

一般**不需要**传 `--dataset-root` / `--repo-id`：checkpoint 已烘焙归一化统计。

### 支持的策略类型

| `config.json` 的 `type` | 离线 eval | 真机 infer | 说明 |
|-------------------------|-----------|------------|------|
| `act` | ✅ | ✅ | 支持 `--n-action-steps`、`--temporal-ensemble-coeff` |
| `diffusion` | ✅ | ⚠️ | eval 可用；真机 infer 需跳过 ACT 专用 override（见下方说明） |

观测/动作布局须与训练数据集一致（TB6-R5 默认 7 维 state/action + 双相机 RGB 480×640）。

### 基本命令

**Dry-run（推荐第一步）：**

```bash
tb6r5-policy-infer \
  --robot-ip 192.168.11.11 \
  --policy-path model/act/080000/pretrained_model \
  --device cuda \
  --dry-run
```

**ACT 真机：**

```bash
tb6r5-policy-infer \
  --robot-ip 192.168.11.11 \
  --policy-path model/act/080000/pretrained_model \
  --device cuda \
  --fps 15 \
  --joint-step-max-rad 0.03
```

**Diffusion 真机（P05 示例，夹爪为 mm，勿加 `--gripper-normalized`）：**

```bash
tb6r5-policy-infer \
  --robot-ip 192.168.11.11 \
  --policy-path outputs/train/tb6r5_rings_P05/checkpoints/100000/pretrained_model \
  --device cuda \
  --fps 15 \
  --dry-run    # 验证通过后再去掉
```

> **Diffusion 注意：** 当前真机入口会对所有策略调用 ACT 专用 `apply_act_inference_overrides()`（访问 `chunk_size`），Diffusion 模型会报错 `DiffusionConfig has no attribute chunk_size`。离线 eval 不受影响。修复前可暂用 `tb6r5-policy-eval` 验证模型，或在本包 `policy.py` 中对 `type != act` 跳过 override。

也可用模块方式运行：

```bash
python -m tb6r5_policy_infer.cli --robot-ip 192.168.11.11 --policy-path ... --dry-run
```

完整参数：`tb6r5-policy-infer --help`

### 观测 / 动作约定

| 字段 | 形状 | 单位 / 说明 |
|------|------|-------------|
| `observation.state` | 7 | `[q0..q5 rad, gripper]`；夹爪默认 **mm** 反馈 |
| `observation.images.realsense_0/1` | HWC uint8 | 480×640×3 RGB；逻辑名须与训练集一致 |
| `action` | 7 | `[q0..q5 rad, gripper]`；下发前经 `--joint-step-max-rad` 限幅 |

训练时夹爪为 **[0,1] 归一化** 的数据集，加 `--gripper-normalized`（见下文「夹爪单位」）。

### 安全与生命周期

| 参数 | 默认 | 说明 |
|------|------|------|
| `--dry-run` | 关 | 只推理打印，不发 RPC |
| `--fps` | 30 | 控制循环频率；首次真机建议 **10～15** |
| `--joint-step-max-rad` | 0.03 | 策略输出限幅：每步最大关节变化（rad） |
| `--home-joint-deg` | 遥操作 home | 启动 / Ctrl+C 复位姿态（度） |
| `--home-settle-time` | 3 s | 复位后等待 |
| `--no-home-on-start` / `--no-home-on-exit` | 关 | 跳过启动或退出复位 |
| `--print-every` | 0.5 s | 调试打印间隔 |

退出：`Ctrl+C` → 停相机 →（可选）复位 → `arm.disable()`。

### JogAnyJ 运动参数（RPC 层）

策略输出的关节目标经 `--joint-step-max-rad` 限幅后，通过 SubLoop1 `JogAnyJ` 下发。下列参数写入 RPC 命令的 `--joint_vel` / `--joint_acc` / `--joint_dec` / `--zone_ratio`，控制**机器人执行**时的运动学限制（与遥操作 `teleop_tb6r5_hardware.py` 同名参数一致）。

| 参数 | 默认 | 说明 |
|------|------|------|
| `--joint-vel` | 6.0 | `JogAnyJ` 关节速度 |
| `--joint-acc` | 3.0 | `JogAnyJ` 关节加速度 |
| `--joint-dec` | 3.0 | `JogAnyJ` 关节减速度 |
| `--zone-ratio` | 0.05 | `JogAnyJ` 过渡区比例 |

```bash
# 执行更慢、更稳
tb6r5-policy-infer ... \
  --joint-vel 4.0 --joint-acc 2.0 --joint-dec 2.0 \
  --zone-ratio 0.05

# 策略跟得更紧（软件层）+ 机器人跑得快（RPC 层）
tb6r5-policy-infer ... \
  --joint-step-max-rad 0.05 --fps 20 \
  --joint-vel 8.0 --joint-acc 4.0 --joint-dec 4.0
```

**调参区分：**

- 策略「算得慢 / 步幅小」→ 调 `--joint-step-max-rad`、`--fps`、ACT 的 `--n-action-steps`
- 机器人「执行拖沓 / 跟不上目标」→ 调 `--joint-vel` / `--joint-acc` / `--joint-dec`
- 运动不够顺滑 → 适当增大 `--zone-ratio`（与采集时 `--zone-ratio` 保持一致更安全）

实现位置：`runner.py` 传入 `TB6R5Interface` → `xrobotoolkit_teleop/hardware/interface/tb6r5.py` 的 `_format_jog_any_j_cmd()`。

### RPC 与夹爪

| 参数 | 默认 | 说明 |
|------|------|------|
| `--arm-rpc-rate-hz` | 30 | 臂 SubLoop1 下发频率 |
| `--gripper-rpc-rate-hz` | 2 | 夹爪 SubLoop1 下发频率 |
| `--gripper-max-distance` | 70 | 全开距离（mm） |
| `--gripper-min-distance` | 30 | 全合距离（mm） |
| `--gripper-continuous` | 开 | 连续 mm；`--no-gripper-continuous` 为滞回二值模式 |
| `--gripper-cmd-delta` | 0.5 | 夹爪指令变化小于此值（mm）不重发 RPC |

### ACT 部署调参（无需重新训练）

三者互斥注意：**不要**同时开 `--temporal-ensemble-coeff` 与 `--refresh-policy-every-step`。

| 模式 | 参数 | 何时重推理 | 适用 |
|------|------|------------|------|
| **默认队列** | （不传） | 每 `n_action_steps` 步 | 与训练一致 |
| **缩短队列** | `--n-action-steps N` | 每 N 步（1…chunk_size） | 更灵敏 |
| **时间集成** | `--temporal-ensemble-coeff 0.01` | 每步推理 + 融合 | 更平滑（原论文常用 0.01） |
| **每步刷新** | `--refresh-policy-every-step` | 每步 `policy.reset()` | 最灵敏、最慢 |

```bash
# 更灵敏：每 10 步重推理（chunk_size=50 时）
tb6r5-policy-infer ... --n-action-steps 10 --fps 20

# 更平滑：Temporal Ensemble
tb6r5-policy-infer ... --temporal-ensemble-coeff 0.01 --fps 20
```

Diffusion 策略由 checkpoint 内 `n_obs_steps`、`n_action_steps`、`horizon` 控制队列，**不要**传上述 ACT 专用参数。

### 相机

推理时图像写入 `observation.images.<name>`（RGB HWC `uint8`，默认 640×480）。**逻辑名**（如 `realsense_0`）须与训练数据集一致；变的只是如何打开物理设备。

| 模式 | 参数 | 依赖 |
|------|------|------|
| **RealSense（默认）** | 默认 SN 在 `constants.py`；可用 `--camera-serials` 覆盖 | `pyrealsense2` |
| **V4L2 `/dev/video*`** | `--camera-devices` | `opencv-python` |
| **HTTP URL** | `--camera-urls` | `urllib` + `opencv-python` |
| **无相机** | `--no-camera` | 喂黑图，仅测 RPC |

**RealSense — 指定序列号：**

```bash
tb6r5-policy-infer ... \
  --camera-serials 'realsense_0=135522071053,realsense_1=244222075136'
```

查本机 SN：

```bash
rs-enumerate-devices | grep Serial
```

**V4L2：**

```bash
tb6r5-policy-infer ... \
  --camera-devices 'realsense_0=/dev/video0,realsense_1=/dev/video2'
```

指定 `--camera-devices` 时**忽略** `--camera-serials`。

**HTTP 远程相机（RsCameraSensor 等）：**

```bash
tb6r5-policy-infer ... \
  --camera-urls 'realsense_0=http://192.168.2.42:8888/RsCameraSensor/0/0/color,realsense_1=http://192.168.2.42:8888/RsCameraSensor/1/0/color'
```

- URL 响应须为 JPEG/PNG 字节
- 指定 `--camera-urls` 时忽略 `--camera-serials` 与 `--camera-devices`

其它：`--camera-width/height/fps`（默认 640×480×30）、`--show-camera` / `--no-show-camera`、`--camera-preview-fps`。

### 夹爪单位

- **默认（mm）**：`action[6]` / `state[6]` 为 mm（0=闭合，`--gripper-max-distance` 默认 70=全开）
- **归一化训练集**：加 `--gripper-normalized`（obs: `feedback_mm/max`，action: `norm×max` 再下发）

### 机器人 SDK（`.so`，不进 git）

真机 RPC/Topic 依赖厂商二进制，需拷到仓库 `dependencies/`：

| 用途 | 路径 |
|------|------|
| RPC 发指令 | `dependencies/hello_demo_py/rpc_py_all/lib/linux/{x86\|arm}/rpc.so` |
| Topic 读状态 | `dependencies/get_status_py/topic_all_py/lib/{x86\|arm}/topic.so` 等 |

ARM（如 TER30）常用 **Python 3.10**。验证：

```bash
python scripts/hardware/verify_tb6r5_sdk.py --robot-ip 192.168.11.11 --send-test-cmd
```

---

## 离线评估（`tb6r5-policy-eval`）

在已有 LeRobot 数据集上计算 action **MAE**，输出推理速度，并生成对比曲线图。

### 基本用法

```bash
tb6r5-policy-eval \
  --policy-path model/act/080000/pretrained_model \
  --dataset-root data/lerobot/tb6r5_yellow_yogurt_47_v3 \
  --repo-id shanchai/tb6r5_yellow_yogurt_47_v3 \
  --device cuda
```

**Diffusion 示例（P05）：**

```bash
tb6r5-policy-eval \
  --policy-path outputs/train/tb6r5_rings_P05/checkpoints/100000/pretrained_model \
  --dataset-root data/lerobot/tb6r5_rings/P05 \
  --repo-id local/P05 \
  --device cuda \
  --max-samples 50 \
  --stride 30 \
  --warmup-samples 2
```

### 采样与速度参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--max-samples` | 1000 | 最多评估多少个采样点 |
| `--stride` | 5 | 每隔 N 帧取 1 帧（越大越快、统计越稀疏） |
| `--warmup-samples` | 10 | 预热帧数，**不计入**末尾 FPS 统计（但仍消耗时间） |
| `--benchmark-only` | 关 | 只测速度，不算 MAE、不画图 |
| `--bench-samples` | 200 | `--benchmark-only` 模式下计时的推理次数 |
| `--output-dir` | `outputs/eval_act/<dataset>` | 曲线图输出目录 |

采样逻辑：`range(0, n_frames, stride)[:max_samples]`。数据集 30 fps 时 `--stride 30` ≈ 每秒取 1 帧。

**Diffusion 评估较慢**：每帧都会 `policy.reset()` 并做完整去噪（默认 ~100 步），GPU 占满是正常现象。快速抽查建议 `--max-samples 50 --stride 30 --warmup-samples 2`。

**只测推理速度：**

```bash
tb6r5-policy-eval \
  --policy-path outputs/train/tb6r5_rings_P05/checkpoints/100000/pretrained_model \
  --dataset-root data/lerobot/tb6r5_rings/P05 \
  --repo-id local/P05 \
  --device cuda \
  --benchmark-only \
  --bench-samples 30 \
  --warmup-samples 3
```

输出图：`action_comparison.png`、`mae_per_dim.png`、`error_over_time.png`。

---

## 依赖版本（按 lerobot 主版本）

`lerobot` 导入策略时会连带加载 `transformers`，**必须与 `huggingface-hub` 版本配套**。

### lerobot 0.4.x

```bash
pip install "lerobot[transformers-dep]>=0.4.0,<0.5.0"
pip install "transformers>=4.57.1,<5.0.0" "huggingface-hub>=0.34.2,<0.36.0"
pip install -e ./tb6r5_policy_infer --no-deps
```

### lerobot 0.5.x

```bash
pip install "transformers>=5.3.0" "huggingface-hub>=1.16.0,<2.0.0"
pip install -e ./tb6r5_policy_infer
```

### 常见报错

```text
DecodingError: The fields `use_peft` are not valid for ACTConfig
```

**修复：** 升级 lerobot，或同步最新 `tb6r5_policy_infer`（`load_pretrained_config` 会自动忽略未知字段）。

```text
ImportError: cannot import name 'is_offline_mode' from 'huggingface_hub'
```

**0.4 环境修复：**

```bash
pip install "transformers>=4.57.1,<5.0.0" "huggingface-hub>=0.34.2,<0.36.0"
```

---

## 包结构

| 模块 | 说明 |
|------|------|
| `cli` | 真机推理 CLI（`tb6r5-policy-infer`） |
| `eval_cli` | 离线评估 CLI（`tb6r5-policy-eval`） |
| `runner` | 硬件控制循环 |
| `policy` | 模型加载与 ACT 部署参数 |
| `camera` | RealSense、V4L2、HTTP 采集 |
| `gripper` | 夹爪观测与指令辅助 |
| `lerobot_compat` | LeRobot 版本兼容与 config 加载 |

硬件接口通过 `xrobotoolkit_teleop.hardware.interface.tb6r5` 调用；真机部署需同时更新主包与推理包（见「远程机器更新」）。

等价的 legacy 入口：`scripts/hardware/policy_infer_tb6r5_act.py`（推荐统一使用 `tb6r5-policy-infer`）。
