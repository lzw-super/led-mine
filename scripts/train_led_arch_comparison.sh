#!/usr/bin/env bash
# Usage: bash scripts/train_led_arch_comparison.sh {dwt|dwt_relu|splitternet|mrlfn|all} [GPU_ID] [extra LED arguments...]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
model="${1:-all}"
gpu="${2:-0}"
if (( $# > 0 )); then shift; fi
if (( $# > 0 )); then shift; fi
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

train_one() {
    local label
    case "$1" in
        dwt) label=LearningDWT_Haar_L3_RepNCB_W32 ;;
        dwt_relu) label=LearningDWT_Haar_L3_RepNCB_W32_ReLU ;;
        splitternet) label=SplitterNet ;;
        mrlfn) label=MRLFN_Paper_S2D_K4_N4_D32 ;;
        *) echo "Unknown model: $1 (expected dwt, dwt_relu, splitternet, mrlfn, all)" >&2; return 2 ;;
    esac
    shift
    "${PYTHON:-/opt/conda/envs/LED-ICCV23/bin/python}" -m led.train -opt "options/LED/pretrain/CVPR20_ELD_${label}.yaml" "$@"
}

if [[ "$model" == all ]]; then
    for architecture in dwt splitternet mrlfn; do
        train_one "$architecture" "$@"
    done
else
    train_one "$model" "$@"
fi
