#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Reuse completed baseline/calibration; new model has its own source hashes/scales.
conda run --no-capture-output -n LED-ICCV23 python quantization/run_baseline.py \
  --models relu_llstem_noact --run-name sid_llstem_noact_20261117 --resume
for config in mixed a16 a8; do
  case "$config" in
    mixed) profile=phase_exp3_pres2d; bits=16; run=qat_llstem_noact_phase_exp3_pres2d ;;
    a16) profile=none; bits=16; run=qat_llstem_noact_w8a16 ;;
    a8) profile=none; bits=8; run=qat_llstem_noact_w8a8 ;;
  esac
  conda run --no-capture-output -n LED-ICCV23 python quantization/train_qat.py \
    --model relu_llstem_noact --ptq-run sid_llstem_noact_20261117 \
    --boundary-profile opt1 --activation-bits "$bits" --mixed-profile "$profile" \
    --run-name "$run" --eval-after-train
done
conda run --no-capture-output -n LED-ICCV23 python quantization/summarize_llstem_noact.py
