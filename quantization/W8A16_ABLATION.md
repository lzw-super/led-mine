# QAT + opt1：W8A8 / W8A16 激活位宽对照

## 设计

以已完成的 `qat_ablation_opt1` 为 W8A8 对照；新运行名为 `qat_ablation_opt1_w8a16`。
PReLU、ReLU 分别独立训练。正式训练由用户执行，不覆盖原始结果。

|项目|已有 W8A8 + opt1|新增 W8A16 + opt1|
|---|---|---|
|学习卷积权重|每输出通道 signed INT8 [-127,127]|相同|
|激活|逐张量 signed INT8 [-127,127]|逐张量 signed INT16 [-32767,32767]|
|激活边界|38 个|相同 38 个|
|Conv → activation|仅激活后量化|相同，改用 A16|
|S2D/D2S|继承输入尺度|相同，无额外量化|
|Haar|opt1 的原有边界|相同，不开启 opt2|
|Bias|INT32，尺度为输入尺度 × 权重尺度|相同规则，以新的 A16 输入尺度编码|
|PReLU α|FP32 可训练|相同|
|教师/损失|原始 FP32；L1 + 0.1 输出 L1 蒸馏|相同|

所有激活量化边界，包括 RAW 输入、Haar、中间特征、残差/拼接输出、最终输出，都使用所选位宽；不是仅提升输入输出精度。
ReLU 输出依旧使用有符号对称量化，不引入 unsigned 量化。
为兼容已有 artifact，boundary_policy 的历史原因字符串仍可能写 A8；实际位宽由 artifact 顶层 `activation_bits` 决定。
加载器会恢复实际位宽；旧 artifact 缺少该字段时按 A8 加载。

## 初始化与公平性

新组从与 W8A8 opt1 相同的原始 FP32 权重开始，沿用相同 PTQ 校准文件。
不从训练好的 W8A8 opt1 checkpoint 追加训练，以免多出 20k 步训练造成混淆。
激活步长初始化为 `s16 = s8 × 127/32767`，因此初始截断范围保持不变，只提高量化分辨率。
保留的权重量化尺度不变。激活 LSQ 梯度归一化随 qmax 更新为 `1/sqrt(N × 32767)`；权重仍使用 127。
训练过程中尺度可学习，所以最终截断范围可能与 A8 不同。

默认：20,000 步、batch 4、crop 256、seed 2026、权重 LR 1e-5、尺度 LR 1e-3，最后 20% 步冻结尺度。
沿用相同合成噪声训练数据和逐步 RNG。学习率、数据、蒸馏权重不针对 A16 单独调参。
本实验回答“相同训练流程下提高激活位宽的效果”，不能把全部指标差异直接解释为某一层的误差。

## 手动启动

在工程根目录顺序运行两个模型，训练完各自自动测试全部 93 对 SID 图像并汇总：

```bash
bash quantization/run_w8a16_ablation.sh
```

也可以分别执行：

```bash
conda run --no-capture-output -n LED-ICCV23 python quantization/train_qat.py \
  --model prelu --boundary-profile opt1 --activation-bits 16 \
  --run-name qat_ablation_opt1_w8a16 --eval-after-train

conda run --no-capture-output -n LED-ICCV23 python quantization/train_qat.py \
  --model relu --boundary-profile opt1 --activation-bits 16 \
  --run-name qat_ablation_opt1_w8a16 --eval-after-train
```

中断恢复（以 PReLU 为例，其余训练参数需与开始时一致）：

```bash
conda run --no-capture-output -n LED-ICCV23 python quantization/train_qat.py \
  --model prelu --boundary-profile opt1 --activation-bits 16 \
  --run-name qat_ablation_opt1_w8a16 \
  --resume quantization/artifacts/qat_ablation_opt1_w8a16/prelu/qat_latest.pth \
  --eval-after-train
```

`--resume` 校验位宽，不能用它把 A8 checkpoint 改为 A16。
批处理脚本遇到已存在的运行会停止；恢复请用上面的单组命令。

## 文件位置和测试

- 训练日志、checkpoint、run.json：`quantization/artifacts/qat_ablation_opt1_w8a16/<prelu或relu>/`
- A16 导出文件：上述目录下的 **`qat_w8a16_qdq.pth`**。
- 测试指标：对应原 experiment 下的 `evaluation/qat_ablation_opt1_w8a16/qat/`。
- `summary.csv` 汇总 100/250/300 倍和全体 PSNR/SSIM；`per_image.csv` 为逐图指标。
- `comparison.md` 仍比较 FP32/PTQ；A8 opt1 对照由单独汇总脚本生成。

仅执行/恢复测试：

```bash
conda run --no-capture-output -n LED-ICCV23 python quantization/evaluate_qat.py \
  --artifact quantization/artifacts/qat_ablation_opt1_w8a16/prelu/qat_w8a16_qdq.pth --resume
```

汇总位宽对照：

```bash
conda run --no-capture-output -n LED-ICCV23 python quantization/summarize_activation_bits.py
```

输出 `quantization/runs/activation_bits_summary.md`；同时检查源权重、数据、训练设置及边界一致性。
缺失的 A16 结果显示“待完成”，不以冒烟测试代替正式结果。

## 解读与硬件范围

已有 A8 opt1：PReLU 37.022260 dB / 0.90662998；ReLU 36.850287 dB / 0.90563387。
若 A16 显著缩小与 FP32 的差距，支持“激活量化是重要限制”的判断；如果收益仍小，需要进一步检查权重量化和训练过程。
A16 仍含权重量化、bias 编码、不同训练轨迹，不能保证恢复 FP32 指标。

这是浮点内核 Q/DQ 仿真，不是 FP16 混合精度训练，也不是整型 FPGA 位精确仿真。
激活存储位数为 A8 的两倍；实际 BRAM、带宽、DSP 和吞吐变化取决于硬件实现。
整数乘法/累加器溢出、重定标乘数及移位误差尚未模拟；FPGA 应重新核算累加器范围。
PReLU α 和固定 Haar 常量的处理沿用此前例外。

验证覆盖 A16 饱和范围、STE/尺度梯度、初始范围保持、权重量化尺度不变、opt1 尺度别名、两模型训练更新与导出重载一致性。
回归测试：

```bash
conda run --no-capture-output -n LED-ICCV23 python -m unittest \
  quantization.test_static_ptq quantization.test_qat \
  quantization.test_boundary_ablation quantization.test_w8a16
```
