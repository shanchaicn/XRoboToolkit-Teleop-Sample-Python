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

- `observation.state`：7 维 = `[q0..q5, gripper_state]`
  - 前 6 维是机械臂关节角（弧度），真机时从机器人读取
  - 第 7 维 `gripper_state` 采集时是**常数 0.0**（`DEFAULT_GRIPPER_OBSERVATION`），
    推理时也喂同一个常数（数据集统计证实：min=max=mean=0.0）
- `observation.images.realsense_0`：RGB，480×640×3，相机序列号 `135522071053`
- `observation.images.realsense_1`：RGB，480×640×3，相机序列号 `327122073649`

输出动作 `action`：7 维 = `[q0..q5, gripper_norm]`

- 前 6 维是关节目标角（弧度）
- 第 7 维 `gripper_norm` 是**归一化夹爪指令 [0,1]**（数据集统计：min=0, max=1）
- 真机映射：`物理距离(mm) = clip(gripper_norm, 0, 1) × gripper_max_distance`
  - 你训练时用的 `gripper_max_distance = 13`，所以推理默认 `--gripper-max-distance 13`

模型权重已烘焙归一化统计，**推理和评估都无需依赖数据集**（数据集仅用于离线对比 MAE）。

---

## 2. 用 Rerun 可视化本地 LeRobot 数据集

把 HF 数据集下载到本地后，可用 LeRobot 自带的 `lerobot-dataset-viz` 在 **Rerun** 里逐帧查看
相机图像、`observation.state`、`action` 等训练数据。

### 2.1 下载数据集到本地（若尚未下载）

```bash
huggingface-cli download shanchai/tb6r5_yellow_yogurt_47_v3 \
  --repo-type dataset \
  --local-dir data/lerobot/tb6r5_yellow_yogurt_47_v3
```

本地目录需包含 `meta/info.json`、`data/`、`videos/` 等 LeRobot v3 结构。

### 2.2 本机直接打开 Rerun 查看器（推荐）

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

### 2.3 无图形界面 / SSH 远程：先导出 `.rrd` 再本地查看

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

### 2.4 远程机器流式查看（distant 模式）

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

### 2.5 自己转换的本地数据集

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

## 3. 第一步：离线评估（验证模型预测精度）

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

## 4. 第二步：Dry-Run（不发指令，只看预测）

`--dry-run` 不连接机器人、不发任何指令，只打印模型预测，用于上真机前的安全确认。

### 4a. 纯流水线测试（无相机，喂黑图）

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

### 4b. 带真实相机的 Dry-Run（推荐）

接好两台 RealSense 后，喂真实图像，观察预测是否随场景合理变化：

```bash
python scripts/hardware/policy_infer_tb6r5_act.py \
  --robot-ip 192.168.11.11 \
  --policy-path model/act_tb6r5_yellow_yogurt/checkpoints/040000/pretrained_model \
  --device cuda \
  --fps 20 \
  --gripper-max-distance 13 \
  --gripper-interval 25 \
  --dry-run \
  --print-every 0.5
```

打印含义：

- `q_cur`：当前关节角（dry-run 下机器人未连接，恒为 0）
- `q_tgt`：模型预测的关节目标
- `q_cmd`：经每步限速钳制后的实际下发值（`--joint-step-max-rad` 默认 0.03 rad/步）
- `g_norm`：归一化夹爪 [0,1]；`g_dist`：映射后的物理距离 (mm)

确认 `q_tgt` 落在合理关节范围、`g_dist` 在 0–13mm 之间，再上真机。

---

## 5. 第三步：真机运行

**去掉 `--dry-run`** 即开始向机器人下发指令。务必先完成 dry-run 验证，并保证周围安全、急停可及。

```bash
python scripts/hardware/policy_infer_tb6r5_act.py \
  --robot-ip 192.168.11.11 \
  --policy-path model/act_tb6r5_yellow_yogurt/checkpoints/040000/pretrained_model \
  --device cuda \
  --fps 20 \
  --joint-step-max-rad 0.03 \
  --gripper-max-distance 13 \
  --gripper-interval 25
```

真机参数：

- `--robot-ip` / `--rpc-port`：机器人地址（RPC 端口默认 5868）
- `--fps`：控制频率，首次建议先低（如 10）观察后再提高
- `--joint-step-max-rad`：每步关节最大变化量，越小越慢越安全（默认 0.03 rad）
- `--gripper-max-distance`：夹爪物理最大距离，必须与训练一致（你的是 13）
- `--gripper-interval`：`MoveTwoFingersGripper` 的 interval（默认 25）

相机相关（一般用默认即可）：

- `--camera-serials`：覆盖默认序列号映射，格式
  `realsense_0=135522071053,realsense_1=327122073649`
- `--camera-width/height/fps`：默认 640×480×30
- `--no-camera`：跳过相机喂黑图（**仅调试**，真机请勿使用）

按 `Ctrl+C` 停止，脚本会自动停相机线程并 `disable()` 机器人。

---

## 6. 安全清单

- [ ] 先离线评估，确认 MAE 合理
- [ ] 再 dry-run（最好带真实相机），确认 `q_tgt` / `g_dist` 合理
- [ ] 真机首次用低 `--fps`（如 10）+ 默认 `--joint-step-max-rad 0.03`
- [ ] 机器人工作空间清空，急停在手边
- [ ] 相机序列号、`--gripper-max-distance` 与训练数据完全一致

---

## 7. 常见问题

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

---

## 8. 相关文件

- 真机推理：`scripts/hardware/policy_infer_tb6r5_act.py`
- 离线评估：`scripts/misc/eval_act_on_lerobot_tb6r5.py`
- 数据集 Rerun 可视化：`lerobot-dataset-viz`（LeRobot 自带，见第 2 节）
- pkl 曲线图：`scripts/misc/plot_tb6r5_pkl_curves.py`
- pkl → LeRobot v3：`scripts/misc/convert_tb6r5_pkl_to_lerobot_v3.py`
