#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for profile in phase_exp4 phase_exp5 phase_exp6; do
  conda run --no-capture-output -n LED-ICCV23 python quantization/train_qat.py \
    --model relu --boundary-profile opt1 --activation-bits 16 \
    --mixed-profile "$profile" --run-name "qat_${profile}" --eval-after-train
done
conda run --no-capture-output -n LED-ICCV23 python quantization/summarize_phase_mixed.py \
  --model relu --exp2-run qat_phase_exp2_corrected --exp3-run qat_phase_exp3_corrected
