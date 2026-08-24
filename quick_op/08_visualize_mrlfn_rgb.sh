#!/usr/bin/env bash
# 在 test_data_mine 上运行 Pochimireddy 2026 MRLFN，并保存融合部署态的
# RAW/.raw/BIN RGB 对比图。实际的输入解码和可视化流程复用
# 05_visualize_rgb.sh；本脚本只固定 MRLFN checkpoint 与训练态网络配置。
#
# 常用用法：
#   bash quick_op/08_visualize_mrlfn_rgb.sh
#   CUDA_ID=0 RATIO=200 bash quick_op/08_visualize_mrlfn_rgb.sh
#   CHECKPOINT=/path/to/net_g_100000.pth RATIO=100 bash quick_op/08_visualize_mrlfn_rgb.sh
#
# 默认 RATIO=100，位于训练时随机曝光倍率 [100, 300] 的下界。BIN 的默认
# 数字增益仍由 05_visualize_rgb.sh 的 BIN_RATIO=2 控制；可以单独覆盖。
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"

CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/experiments/LED_Pretrain_MRLFN_CVPR26_Paper_Setting/models/net_g_latest.pth}"
NETWORK_OPTIONS="${NETWORK_OPTIONS:-${REPO_ROOT}/options/base/network_g/mrlfn.yaml}"
PARAM_KEY="${PARAM_KEY:-params}"
RATIO="${RATIO:-100}"

CHECKPOINT_BASENAME="$(basename "${CHECKPOINT%.pth}")"
DEPLOY_CHECKPOINT="${DEPLOY_CHECKPOINT:-${REPO_ROOT}/tmp/led_deploy_checkpoints/${CHECKPOINT_BASENAME}.mrlfn.deploy.pth}"
SAVE_PATH="${SAVE_PATH:-${REPO_ROOT}/inference/qualitative/${CHECKPOINT_BASENAME}_mrlfn_test_data_mine}"

export CHECKPOINT NETWORK_OPTIONS PARAM_KEY RATIO DEPLOY_CHECKPOINT SAVE_PATH
exec bash "${SCRIPT_DIR}/05_visualize_rgb.sh"
