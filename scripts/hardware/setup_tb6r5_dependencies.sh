#!/usr/bin/env bash
# 安装 TB6-R5 遥操作依赖的 Topic / RPC SDK（Gitee 固定 commit，与本机原用版本一致）。
#
# 默认安装到仓库根目录 dependencies/：
#   dependencies/get_status_py   ← Topic
#   dependencies/hello_demo_py   ← RPC（teleop_tb6r5 使用）
#
# 用法：
#   ./scripts/hardware/setup_tb6r5_dependencies.sh
#   ./scripts/hardware/setup_tb6r5_dependencies.sh /path/to/dependencies
#   DEST=/data/tb6r5_deps ./scripts/hardware/setup_tb6r5_dependencies.sh
#   FORCE=1 ./scripts/hardware/setup_tb6r5_dependencies.sh   # 已存在则删掉重装
#
# 固定版本（勿随意改，除非明确要升级）：
#   get_status_py  ab1a300  优化subsystem.data消息获取  (2026-06-04)
#   hello_demo_py  27cd9e2  修改ip                      (2026-04-14)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# ---- pinned Gitee remotes / commits (original working versions) ----
GET_STATUS_URL="${GET_STATUS_URL:-https://gitee.com/zhenshi-controller-api-data/get_status_py.git}"
GET_STATUS_COMMIT="${GET_STATUS_COMMIT:-ab1a30059a45a6c80bad5645278701b80e3a6e4f}"

HELLO_DEMO_URL="${HELLO_DEMO_URL:-https://gitee.com/zhenshi-controller-api-data/hello_demo_py.git}"
HELLO_DEMO_COMMIT="${HELLO_DEMO_COMMIT:-27cd9e2fd67356c55fd482dd2821ee58c4c9bdd4}"

DEST="${1:-${DEST:-${REPO_ROOT}/dependencies}}"
FORCE="${FORCE:-0}"

clone_pinned() {
  local name="$1"
  local url="$2"
  local commit="$3"
  local target="${DEST}/${name}"

  if [[ -d "${target}/.git" ]]; then
    local cur
    cur="$(git -C "${target}" rev-parse HEAD 2>/dev/null || true)"
    if [[ "${cur}" == "${commit}" && "${FORCE}" != "1" ]]; then
      echo "[OK] ${name} already at ${commit:0:7} -> ${target}"
      return 0
    fi
    if [[ "${FORCE}" == "1" ]]; then
      echo "[INFO] FORCE=1: removing ${target}"
      rm -rf "${target}"
    else
      echo "[WARN] ${name} exists at ${cur:0:7:-unknown}, expected ${commit:0:7}."
      echo "       Re-run with FORCE=1 to wipe and reinstall, or remove: ${target}"
      return 1
    fi
  elif [[ -e "${target}" ]]; then
    if [[ "${FORCE}" == "1" ]]; then
      echo "[INFO] FORCE=1: removing non-git ${target}"
      rm -rf "${target}"
    else
      echo "[ERROR] ${target} exists but is not a git checkout. Use FORCE=1 or remove it."
      return 1
    fi
  fi

  echo "[INFO] Cloning ${name} @ ${commit:0:7}"
  mkdir -p "${DEST}"
  git clone "${url}" "${target}"
  git -C "${target}" checkout --detach "${commit}"
  echo "[OK] ${name} -> $(git -C "${target}" rev-parse --short HEAD) (${target})"
}

verify_layout() {
  local missing=0
  local topic_so rpc_so
  for rel in \
    "get_status_py/topic_all_py/lib/x86/topic.so" \
    "hello_demo_py/rpc_py_all/lib/linux/x86/rpc.so"
  do
    if [[ ! -f "${DEST}/${rel}" ]]; then
      echo "[WARN] missing: ${DEST}/${rel}"
      missing=1
    fi
  done
  if [[ "${missing}" -eq 0 ]]; then
    echo "[OK] layout check (x86 topic.so + rpc.so present)"
  else
    echo "[WARN] some binaries missing; ARM trees use lib/arm or lib/linux/arm — check vendor docs if needed."
  fi
}

echo "==== TB6-R5 SDK dependencies (pinned) ===="
echo "DEST=${DEST}"
echo "get_status_py  ${GET_STATUS_COMMIT:0:7}  ${GET_STATUS_URL}"
echo "hello_demo_py  ${HELLO_DEMO_COMMIT:0:7}  ${HELLO_DEMO_URL}"
echo

clone_pinned "get_status_py" "${GET_STATUS_URL}" "${GET_STATUS_COMMIT}"
clone_pinned "hello_demo_py" "${HELLO_DEMO_URL}" "${HELLO_DEMO_COMMIT}"
verify_layout

echo
echo "Done. Teleop expects:"
echo "  ${DEST}/get_status_py/topic_all_py/"
echo "  ${DEST}/hello_demo_py/rpc_py_all/"
echo
echo "Verify: python scripts/hardware/verify_tb6r5_sdk.py --topic-only"
echo "        python scripts/hardware/verify_tb6r5_sdk.py --robot-ip <IP>"
