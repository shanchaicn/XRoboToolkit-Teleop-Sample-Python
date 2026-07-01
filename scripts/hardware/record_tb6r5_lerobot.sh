#!/usr/bin/env bash
# TB6-R5 LeRobot 采集（方案 A：teleop_tb6r5_hardware.py）
#
# 任务编号表：data/lerobot/tb6r5_rings/六色圆环抓放任务数据采集计划.md §3
#
# 用法：
#   1. 改下方 TASK_ID（如 P01、S03），或 DATASET + TASK_KEY
#   2. ./scripts/hardware/record_tb6r5_lerobot.sh

set -euo pipefail

# =============================================================================
# ★ 采集任务选择
# =============================================================================
#
# 方式 1（推荐）：命令行覆盖，无需改文件
#   TASK_ID=P01 RESUME=true ./scripts/hardware/record_tb6r5_lerobot.sh
#
# 方式 2：改下方默认值后直接 ./scripts/hardware/record_tb6r5_lerobot.sh
#
# 新建数据集：OVERWRITE=true，RESUME=false
# 续采已有数据集：OVERWRITE=false，RESUME=true
#
_DEFAULT_TASK_ID="P01"
_DEFAULT_DATASET=""
_DEFAULT_TASK_KEY=""
_DEFAULT_OVERWRITE="false"
_DEFAULT_RESUME="false"
_DEFAULT_CUSTOM="false"

TASK_ID="${TASK_ID:-$_DEFAULT_TASK_ID}"
DATASET="${DATASET:-$_DEFAULT_DATASET}"
TASK_KEY="${TASK_KEY:-$_DEFAULT_TASK_KEY}"
OVERWRITE="${OVERWRITE:-$_DEFAULT_OVERWRITE}"
RESUME="${RESUME:-$_DEFAULT_RESUME}"
CUSTOM="${CUSTOM:-$_DEFAULT_CUSTOM}"

normalize_bool() {
  case "${1,,}" in
    true | 1 | yes | on | ture) echo "true" ;;  # ture: common typo
    *) echo "false" ;;
  esac
}

OVERWRITE="$(normalize_bool "${OVERWRITE}")"
RESUME="$(normalize_bool "${RESUME}")"
CUSTOM="$(normalize_bool "${CUSTOM}")"

if [[ "${OVERWRITE}" == "true" && "${RESUME}" == "true" ]]; then
  echo "错误: OVERWRITE 与 RESUME 不能同时为 true" >&2
  exit 1
fi

# 完全自定义路径 / repo / task 时设 CUSTOM=true，并填写下方三项
CUSTOM_ROOT="data/lerobot/TEST"
CUSTOM_REPO_ID="local/TEST"
CUSTOM_TASK="TEST."

# ---------------------------------------------------------------------------
# 机器人 / 遥操作
# ---------------------------------------------------------------------------
ROBOT_IP="192.168.11.11"
TELEOP_MODE="placo_ik"
CONTROL_RATE_HZ=30
SCALE_FACTOR=1.5
ZONE_RATIO=0.00
GRIPPER_MAX_D=70
GRIPPER_MIN_D=30
LOG_FREQ=30

# ---------------------------------------------------------------------------
# 相机（RealSense 序列号）
# 当前可用：244222075136 / 135522071053 / 347622071274
# ---------------------------------------------------------------------------
CAMERA_SERIAL_0="135522071053"
CAMERA_SERIAL_1="244222075136"

LEROBOT_IMAGE_WRITER_PROCESSES=0
LEROBOT_IMAGE_WRITER_THREADS=4
LEROBOT_ENCODER_THREADS=2

# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

normalize_dataset() {
  case "${1}" in
    D1 | d1 | place_on_plate) echo "place_on_plate" ;;
    D2 | d2 | stack) echo "stack" ;;
    *) echo "${1}" ;;
  esac
}

# P01–P06 / S01–S15 → 内部 TASK_KEY；其余原样返回（小写）
normalize_task_key() {
  case "${1^^}" in
    P01) echo "red" ;;
    P02) echo "yellow" ;;
    P03) echo "green" ;;
    P04) echo "cyan" ;;
    P05) echo "blue" ;;
    P06) echo "purple" ;;
    S01) echo "red_yellow" ;;
    S02) echo "red_green" ;;
    S03) echo "red_cyan" ;;
    S04) echo "red_blue" ;;
    S05) echo "red_purple" ;;
    S06) echo "yellow_green" ;;
    S07) echo "yellow_cyan" ;;
    S08) echo "yellow_blue" ;;
    S09) echo "yellow_purple" ;;
    S10) echo "green_cyan" ;;
    S11) echo "green_blue" ;;
    S12) echo "green_purple" ;;
    S13) echo "cyan_blue" ;;
    S14) echo "cyan_purple" ;;
    S15) echo "blue_purple" ;;
    *) echo "${1,,}" ;;
  esac
}

infer_dataset_from_id() {
  case "${1^^}" in
    P01 | P02 | P03 | P04 | P05 | P06) echo "place_on_plate" ;;
    S01 | S02 | S03 | S04 | S05 | S06 | S07 | S08 | S09 | S10 | S11 | S12 | S13 | S14 | S15)
      echo "stack"
      ;;
    *) echo "" ;;
  esac
}

resolve_selection() {
  local raw_id raw_dataset raw_key

  if [[ -n "${TASK_ID}" ]]; then
    raw_id="${TASK_ID}"
    raw_dataset="$(infer_dataset_from_id "${raw_id}")"
    if [[ -z "${raw_dataset}" ]]; then
      echo "错误: 未知 TASK_ID='${TASK_ID}'，请用 P01–P06 或 S01–S15" >&2
      exit 1
    fi
    DATASET="${raw_dataset}"
    TASK_KEY="$(normalize_task_key "${raw_id}")"
    TASK_CODE="${raw_id^^}"
    return
  fi

  if [[ -z "${DATASET}" || -z "${TASK_KEY}" ]]; then
    echo "错误: 请设置 TASK_ID（推荐），或同时设置 DATASET + TASK_KEY" >&2
    exit 1
  fi

  DATASET="$(normalize_dataset "${DATASET}")"
  if [[ "${TASK_KEY^^}" =~ ^(P[0-9]{2}|S[0-9]{2})$ ]]; then
    TASK_CODE="${TASK_KEY^^}"
    inferred="$(infer_dataset_from_id "${TASK_CODE}")"
    if [[ -n "${inferred}" && "${inferred}" != "${DATASET}" ]]; then
      echo "错误: TASK_KEY='${TASK_KEY}' 属于 ${inferred}，与 DATASET='${DATASET}' 不匹配" >&2
      exit 1
    fi
    TASK_KEY="$(normalize_task_key "${TASK_KEY}")"
  else
    TASK_KEY="$(normalize_task_key "${TASK_KEY}")"
    TASK_CODE=""
  fi
}

resolve_dataset_paths() {
  if [[ -z "${TASK_CODE:-}" ]]; then
    echo "错误: TASK_CODE 未设置，请用 TASK_ID=P01 等" >&2
    exit 1
  fi
  # 一个 TASK_ID = 一个独立数据集目录
  LEROBOT_ROOT="data/lerobot/tb6r5_rings/${TASK_CODE}"
  LEROBOT_REPO_ID="local/${TASK_CODE}"
}

resolve_task_string() {
  case "${DATASET}:${TASK_KEY}" in
    place_on_plate:red)
      LEROBOT_TASK="Pick up the red ring and place it on the white plate."
      ;;
    place_on_plate:yellow)
      LEROBOT_TASK="Pick up the yellow ring and place it on the white plate."
      ;;
    place_on_plate:green)
      LEROBOT_TASK="Pick up the green ring and place it on the white plate."
      ;;
    place_on_plate:cyan)
      LEROBOT_TASK="Pick up the cyan ring and place it on the white plate."
      ;;
    place_on_plate:blue)
      LEROBOT_TASK="Pick up the blue ring and place it on the white plate."
      ;;
    place_on_plate:purple)
      LEROBOT_TASK="Pick up the purple ring and place it on the white plate."
      ;;
    stack:red_yellow)
      LEROBOT_TASK="Place the red ring on top of the yellow ring."
      ;;
    stack:red_green)
      LEROBOT_TASK="Place the red ring on top of the green ring."
      ;;
    stack:red_cyan)
      LEROBOT_TASK="Place the red ring on top of the cyan ring."
      ;;
    stack:red_blue)
      LEROBOT_TASK="Place the red ring on top of the blue ring."
      ;;
    stack:red_purple)
      LEROBOT_TASK="Place the red ring on top of the purple ring."
      ;;
    stack:yellow_green)
      LEROBOT_TASK="Place the yellow ring on top of the green ring."
      ;;
    stack:yellow_cyan)
      LEROBOT_TASK="Place the yellow ring on top of the cyan ring."
      ;;
    stack:yellow_blue)
      LEROBOT_TASK="Place the yellow ring on top of the blue ring."
      ;;
    stack:yellow_purple)
      LEROBOT_TASK="Place the yellow ring on top of the purple ring."
      ;;
    stack:green_cyan)
      LEROBOT_TASK="Place the green ring on top of the cyan ring."
      ;;
    stack:green_blue)
      LEROBOT_TASK="Place the green ring on top of the blue ring."
      ;;
    stack:green_purple)
      LEROBOT_TASK="Place the green ring on top of the purple ring."
      ;;
    stack:cyan_blue)
      LEROBOT_TASK="Place the cyan ring on top of the blue ring."
      ;;
    stack:cyan_purple)
      LEROBOT_TASK="Place the cyan ring on top of the purple ring."
      ;;
    stack:blue_purple)
      LEROBOT_TASK="Place the blue ring on top of the purple ring."
      ;;
    *)
      echo "错误: 未知 TASK_KEY='${TASK_KEY}'（DATASET='${DATASET}'）" >&2
      echo "编号见 data/lerobot/tb6r5_rings/六色圆环抓放任务数据采集计划.md §3" >&2
      exit 1
      ;;
  esac
}

task_code_from_key() {
  case "${DATASET}:${TASK_KEY}" in
    place_on_plate:red) echo "P01" ;;
    place_on_plate:yellow) echo "P02" ;;
    place_on_plate:green) echo "P03" ;;
    place_on_plate:cyan) echo "P04" ;;
    place_on_plate:blue) echo "P05" ;;
    place_on_plate:purple) echo "P06" ;;
    stack:red_yellow) echo "S01" ;;
    stack:red_green) echo "S02" ;;
    stack:red_cyan) echo "S03" ;;
    stack:red_blue) echo "S04" ;;
    stack:red_purple) echo "S05" ;;
    stack:yellow_green) echo "S06" ;;
    stack:yellow_cyan) echo "S07" ;;
    stack:yellow_blue) echo "S08" ;;
    stack:yellow_purple) echo "S09" ;;
    stack:green_cyan) echo "S10" ;;
    stack:green_blue) echo "S11" ;;
    stack:green_purple) echo "S12" ;;
    stack:cyan_blue) echo "S13" ;;
    stack:cyan_purple) echo "S14" ;;
    stack:blue_purple) echo "S15" ;;
    *) echo "?" ;;
  esac
}

if [[ "${CUSTOM}" == "true" ]]; then
  LEROBOT_ROOT="${CUSTOM_ROOT}"
  LEROBOT_REPO_ID="${CUSTOM_REPO_ID}"
  LEROBOT_TASK="${CUSTOM_TASK}"
  TASK_CODE="CUSTOM"
  DATASET="custom"
else
  resolve_selection
  resolve_task_string
  if [[ -z "${TASK_CODE:-}" ]]; then
    TASK_CODE="$(task_code_from_key)"
  fi
  resolve_dataset_paths
fi

echo "========================================"
echo "TASK_ID : ${TASK_CODE}"
echo "TYPE    : ${DATASET}"
echo "ROOT    : ${LEROBOT_ROOT}"
echo "REPO_ID : ${LEROBOT_REPO_ID}"
echo "TASK    : ${LEROBOT_TASK}"
echo "MODE    : overwrite=${OVERWRITE} resume=${RESUME}"
echo "========================================"

EXTRA_ARGS=()
if [[ "${OVERWRITE}" == "true" ]]; then
  EXTRA_ARGS+=(--lerobot-overwrite)
fi
if [[ "${RESUME}" == "true" ]]; then
  EXTRA_ARGS+=(--lerobot-resume)
fi

exec python scripts/hardware/teleop_tb6r5_hardware.py \
  --robot-ip "${ROBOT_IP}" \
  --teleop-mode "${TELEOP_MODE}" \
  --control-rate-hz "${CONTROL_RATE_HZ}" \
  --scale-factor "${SCALE_FACTOR}" \
  --zone-ratio "${ZONE_RATIO}" \
  --gripper-max-d "${GRIPPER_MAX_D}" \
  --gripper-min-d "${GRIPPER_MIN_D}" \
  --enable-log-data \
  --enable-camera \
  --camera-serial-dict.realsense-0 "${CAMERA_SERIAL_0}" \
  --camera-serial-dict.realsense-1 "${CAMERA_SERIAL_1}" \
  --enable-lerobot-log \
  --lerobot-root "${LEROBOT_ROOT}" \
  --lerobot-repo-id "${LEROBOT_REPO_ID}" \
  --lerobot-task "${LEROBOT_TASK}" \
  --lerobot-streaming-encoding \
  --lerobot-image-writer-processes "${LEROBOT_IMAGE_WRITER_PROCESSES}" \
  --lerobot-image-writer-threads "${LEROBOT_IMAGE_WRITER_THREADS}" \
  --lerobot-encoder-threads "${LEROBOT_ENCODER_THREADS}" \
  --no-enable-camera-depth \
  --log-freq "${LOG_FREQ}" \
  "${EXTRA_ARGS[@]}"
