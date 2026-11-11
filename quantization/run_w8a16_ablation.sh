#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for model in prelu relu; do
  conda run --no-capture-output -n LED-ICCV23 python quantization/train_qat.py \
    --model "$model" --boundary-profile opt1 --activation-bits 16 \
    --run-name qat_ablation_opt1_w8a16 --eval-after-train
done
conda run --no-capture-output -n LED-ICCV23 python quantization/summarize_activation_bits.py
