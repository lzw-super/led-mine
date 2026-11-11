#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for model in prelu relu; do
  for profile in phase_exp1 phase_exp2 phase_exp3; do
    conda run --no-capture-output -n LED-ICCV23 python quantization/train_qat.py \
      --model "$model" --boundary-profile opt1 --activation-bits 16 \
      --mixed-profile "$profile" --run-name "qat_${profile}" --eval-after-train
  done
done
conda run --no-capture-output -n LED-ICCV23 python quantization/summarize_phase_mixed.py
