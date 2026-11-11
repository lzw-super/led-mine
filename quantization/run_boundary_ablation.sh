#!/usr/bin/env bash
# Six independent runs; no warm-start from the already-trained QAT model.
set -euo pipefail
cd "$(dirname "$0")/.."
for model in prelu relu; do
  for profile in opt1 opt2 opt12; do
    conda run --no-capture-output -n LED-ICCV23 python quantization/train_qat.py \
      --model "$model" --boundary-profile "$profile" \
      --run-name "qat_ablation_${profile}" --eval-after-train
  done
done
conda run --no-capture-output -n LED-ICCV23 python quantization/summarize_ablation.py
