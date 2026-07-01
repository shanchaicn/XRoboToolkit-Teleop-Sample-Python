# tb6r5_policy_infer

独立于 `xrobotoolkit_teleop` 遥操作框架的 TB6-R5 LeRobot ACT 策略推理与离线评估包。

**真机推理数据接口（观测/动作/相机/RPC）**：[docs/ACT_Policy_Infer_Interface.md](../docs/ACT_Policy_Infer_Interface.md)

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
cd /home/ai/pi0.5/XRoboToolkit-Teleop-Sample-Python-main
source /opt/venv/bin/activate   # 或你的 conda/venv

# 1. 推理包 + LeRobot 栈（按上文 lerobot 0.5.x 对齐 transformers）
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

## 真机推理

```bash
tb6r5-policy-infer \
  --robot-ip 192.168.11.11 \
  --policy-path model/act/080000/pretrained_model \
  --dry-run
```

去掉 `--dry-run` 后向机器人下发指令。完整参数见 `tb6r5-policy-infer --help`。

也可用模块方式运行：

```bash
python -m tb6r5_policy_infer.cli --robot-ip 192.168.11.11 --policy-path ... --dry-run
```

### 相机

推理时图像写入 `observation.images.<name>`（RGB HWC `uint8`，默认 640×480）。**逻辑名**（如 `realsense_0`）须与训练数据集一致；变的只是如何打开物理设备。

| 模式 | 参数 | 依赖 |
|------|------|------|
| **RealSense（默认）** | 默认 SN 在 `constants.py`；可用 `--camera-serials` 覆盖 | `pyrealsense2`（随主包 `pip install -e .`） |
| **V4L2 `/dev/video*`** | `--camera-devices` | 仅 `opencv-python`（推理包已声明） |
| **HTTP URL** | `--camera-urls` | 标准库 `urllib` + `opencv-python`（无需本地相机） |
| **无相机** | `--no-camera` | 喂黑图，仅测机械臂/RPC |

**RealSense — 指定序列号：**

```bash
tb6r5-policy-infer ... \
  --camera-serials 'realsense_0=135522071053,realsense_1=327122073649'
```

查本机 SN：

```bash
rs-enumerate-devices | grep Serial
# 或
python -c "import pyrealsense2 as rs; print([d.get_info(rs.camera_info.serial_number) for d in rs.context().query_devices()])"
```

**V4L2 — `/dev/video*` 或数字索引：**

```bash
tb6r5-policy-infer ... \
  --camera-devices 'realsense_0=/dev/video0,realsense_1=/dev/video2'

# 等价写法（0 -> /dev/video0）
tb6r5-policy-infer ... \
  --camera-devices 'realsense_0=0,realsense_1=2'
```

指定 `--camera-devices` 时会**忽略** `--camera-serials`。查设备：

```bash
ls -l /dev/video*
v4l2-ctl --list-devices   # 需安装 v4l-utils
```

**HTTP URL — 远程相机服务（如 RsCameraSensor）：**

适用于相机由另一台机器/服务提供 HTTP 快照，推理机无需接 USB 相机、无需 `pyrealsense2`：

```bash
tb6r5-policy-infer ... \
  --camera-urls 'realsense_0=http://192.168.2.42:8888/RsCameraSensor/0/0/color,realsense_1=http://192.168.2.42:8888/RsCameraSensor/1/0/color'
```

- 每个 URL 通过 **HTTP GET** 轮询，响应体须为 **JPEG/PNG** 图像字节
- 等号左边逻辑名（`realsense_0`）须与训练数据一致；右边为完整 URL
- 指定 `--camera-urls` 时**忽略** `--camera-serials` 与 `--camera-devices`
- 双相机时 sensor 索引通常对应 URL 路径中的 `/0/0/`、`/1/0/` 等

快速验证 URL 是否可解码：

```bash
curl -s -o /tmp/cam0.jpg 'http://192.168.2.42:8888/RsCameraSensor/0/0/color' && file /tmp/cam0.jpg
```

其它相机相关参数：`--camera-width/height/fps`（默认 640×480×30）、`--show-camera` / `--no-show-camera`。

### 夹爪单位

- 默认：`action[6]` / `state[6]` 为 **mm**（0=闭合，`--gripper-max-distance` 默认 70=全开）
- 训练为 **0–1 归一化** 时加 `--gripper-normalized`（obs: `mm/max`，action: `norm×max` 再下发）

### 机器人 SDK（`.so`，不进 git）

真机 RPC/Topic 依赖厂商二进制，需拷到仓库 `dependencies/`（`git pull` 不会带上）：

| 用途 | 路径 |
|------|------|
| RPC 发指令 | `dependencies/hello_demo_py/rpc_py_all/lib/linux/{x86\|arm}/rpc.so` |
| Topic 读状态 | `dependencies/get_status_py/topic_all_py/lib/{x86\|arm}/topic.so` 等 |

ARM（如 TER30）常用 **Python 3.10**。验证：

```bash
python scripts/hardware/verify_tb6r5_sdk.py --robot-ip 192.168.11.11 --send-test-cmd
```

## 依赖版本（按 lerobot 主版本）

`lerobot` 导入策略时会连带加载 `transformers`，**必须与 `huggingface-hub` 版本配套**。

### lerobot 0.4.x（如另一台 TER30JB3 机器）

```bash
pip install "lerobot[transformers-dep]>=0.4.0,<0.5.0"
# 或手动对齐：
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

**原因：** 模型用较新 LeRobot（0.4.3+/0.5）训练，`config.json` 含旧版不认识的字段。

**修复（任选其一）：**

```bash
# 推荐：升到 0.4.4（仍属 0.4 系）
pip install "lerobot==0.4.4"

# 或手动删 config 里多余字段
sed -i '/"use_peft"/d' model/act/080000/pretrained_model/config.json

# 或同步最新 tb6r5_policy_infer（会自动忽略未知字段）
pip install -e ./tb6r5_policy_infer --no-deps
```

```text
ImportError: cannot import name 'is_offline_mode' from 'huggingface_hub'
```

**原因：** 装了 `transformers 5.x`，但 `huggingface-hub` 仍是 0.4 时代的 `<0.36`（lerobot 0.4 要求）。

**0.4 环境修复（降级 transformers，不要升 huggingface-hub 到 1.x）：**

```bash
pip install "transformers>=4.57.1,<5.0.0" "huggingface-hub>=0.34.2,<0.36.0"
```

验证：

```bash
python -c "import transformers, huggingface_hub; print(transformers.__version__, huggingface_hub.__version__)"
# 期望类似：4.57.x  0.35.x
```

## 离线评估

在已有 LeRobot 数据集上计算 action MAE，并生成对比曲线：

```bash
tb6r5-policy-eval \
  --policy-path model/act/080000/pretrained_model \
  --dataset-root data/lerobot/tb6r5_yellow_yogurt_47_v3 \
  --repo-id shanchai/tb6r5_yellow_yogurt_47_v3
```

## 包结构

| 模块 | 说明 |
|------|------|
| `cli` | 真机推理 CLI（`tb6r5-policy-infer`） |
| `eval_cli` | 离线评估 CLI（`tb6r5-policy-eval`） |
| `runner` | 硬件控制循环 |
| `policy` | 模型加载与 ACT 部署参数 |
| `camera` | RealSense、V4L2（`--camera-devices`）或 HTTP（`--camera-urls`）采集 |
| `gripper` | 夹爪观测与指令辅助 |

硬件接口仍通过 `xrobotoolkit_teleop.hardware.interface.tb6r5` 调用，不修改遥操作源码。
