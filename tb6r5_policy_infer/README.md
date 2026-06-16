# tb6r5_policy_infer

独立于 `xrobotoolkit_teleop` 遥操作框架的 TB6-R5 LeRobot ACT 策略推理与离线评估包。

## 安装

`xrobotoolkit_teleop` 不在 PyPI，不能作为 pip 自动依赖；请在本仓库根目录分步安装：

```bash
# 仅离线评估：只装推理包即可
pip install -e ./tb6r5_policy_infer

# 真机推理：还需主包（TB6-R5 / RealSense 硬件接口）
pip install -e . --no-deps
pip install -e ./tb6r5_policy_infer --no-deps
```

与 `lerobot_robot_tb6r5` 相同，主包用 `--no-deps` 避免拉取 PyPI 上不存在的包。

## 真机推理

```bash
tb6r5-policy-infer \
  --robot-ip 192.168.11.11 \
  --policy-path model/act/080000/pretrained_model \
  --dry-run
```

去掉 `--dry-run` 后向机器人下发指令。完整参数与 `scripts/hardware/policy_infer_tb6r5_act.py` 一致。

也可用模块方式运行：

```bash
python -m tb6r5_policy_infer.cli --robot-ip 192.168.11.11 --policy-path ... --dry-run
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
| `camera` | RealSense 采集 |
| `gripper` | 夹爪观测与指令辅助 |

硬件接口仍通过 `xrobotoolkit_teleop.hardware.interface.tb6r5` 调用，不修改遥操作源码。
