# ReLU：以实验 3 为基线的 head / S2D 位宽对照

仅启动 ReLU。为避免与已有实验冲突，本轮用户提出的实验 1/2/3 分别命名为 `phase_exp4/5/6`。
参照组为正确补跑的 `qat_phase_exp3_corrected`，不是旧的错误配置 `qat_phase_exp3`。

|位置|参照实验 3|本轮 1：exp4|本轮 2：exp5|本轮 3：exp6|
|---|---|---|---|---|
|Phase 1、RAW|A16|A16|A16|A16|
|主分支 S2D 输出 conv2d_3|A8|A8|A16|A16|
|stem、5 个 block 输出|A8|A8|A8|A8|
|noisy S2D 输出|继承 RAW A16|相同|相同|相同|
|head 输出|A16|A8|A16|A8|
|cat_4、fusion 输出|A16|A16|A16|A16|
|最终 D2S|继承 fusion A16|相同|相同|相同|
|独立量化边界数|39|39|38|38|

“head 激活”指 head 卷积输出张量；不新增 ReLU，也不改变网络结构。
head 输入在所有组中仍为最后一个 block 的 A8 输出。exp4/6 的 head 先经历 A8 舍入，即使随后 concat 输出 A16，也不能恢复已经丢失的分辨率。
exp5/6 的主分支 S2D 纯重排继承 Phase 1 输出的 A16 编码，不引入独立尺度或额外舍入；stem 输入因此为 A16，其 bias 仍按实际输入尺度与权重尺度编码。
学习卷积权重保持逐通道 W8；激活逐张量、有符号对称。维持 opt1，不开启 opt2。

## 公平初始化

“以实验 3 为基础”指配置消融，不是从实验 3 追加训练。新组从同一原始 FP32 checkpoint 和 PTQ 截断范围初始化，使用相同 20k 步预算、seed 2026、batch 4、crop 256、合成数据和 0.1 输出蒸馏权重。
因此可与实验 3、全 A16 直接比较同预算 QAT 结果。没有新增学习率或损失调参。

## 执行

工程根目录运行，仅顺序训练 ReLU 的三组，每组完成自动测试 93 对图像：

```bash
bash quantization/run_relu_phase456.sh
```

逐组命令：

```bash
# 本轮实验 1：head 输出改为 A8
conda run --no-capture-output -n LED-ICCV23 python quantization/train_qat.py \
  --model relu --boundary-profile opt1 --activation-bits 16 \
  --mixed-profile phase_exp4 --run-name qat_phase_exp4 --eval-after-train

# 本轮实验 2：主分支 S2D 输出恢复 A16
conda run --no-capture-output -n LED-ICCV23 python quantization/train_qat.py \
  --model relu --boundary-profile opt1 --activation-bits 16 \
  --mixed-profile phase_exp5 --run-name qat_phase_exp5 --eval-after-train

# 本轮实验 3：上述两项组合
conda run --no-capture-output -n LED-ICCV23 python quantization/train_qat.py \
  --model relu --boundary-profile opt1 --activation-bits 16 \
  --mixed-profile phase_exp6 --run-name qat_phase_exp6 --eval-after-train
```

务必同时匹配 `--mixed-profile` 和 `--run-name`，运行名本身不决定位宽；批量脚本用同一个变量生成两者，避免配置错配。

恢复 exp4 示例（恢复其他组时同步修改 profile、run-name、checkpoint 路径）：

```bash
conda run --no-capture-output -n LED-ICCV23 python quantization/train_qat.py \
  --model relu --boundary-profile opt1 --activation-bits 16 \
  --mixed-profile phase_exp4 --run-name qat_phase_exp4 \
  --resume quantization/artifacts/qat_phase_exp4/relu/qat_latest.pth --eval-after-train
```

不要用旧实验 3 的 checkpoint 作为 --resume。批处理遇到已有运行会停止，防止覆盖。

## 输出与汇总

- 权重/日志：`quantization/artifacts/qat_phase_exp<N>/relu/`，N 为 4、5、6。
- 导出：上述目录下 `qat_w8_mixed_qdq.pth`。
- 测试：`experiments/LED_Pretrain_CVPR20_ELD_LearningDWT_Haar_L3_RepNCB_W32_ReLU/evaluation/qat_phase_exp<N>/qat/`。
- 汇总：`quantization/runs/phase_mixed_summary.md`，包含 FP32、全 A8、全 A16、实验 1–6，各倍率 PSNR/SSIM。

```bash
conda run --no-capture-output -n LED-ICCV23 python quantization/summarize_phase_mixed.py \
  --model relu --exp2-run qat_phase_exp2_corrected --exp3-run qat_phase_exp3_corrected
```

单独测试 exp4：

```bash
conda run --no-capture-output -n LED-ICCV23 python quantization/evaluate_qat.py \
  --artifact quantization/artifacts/qat_phase_exp4/relu/qat_w8_mixed_qdq.pth --resume
```

## 结果应如何解释

- exp4 对实验 3：在主分支入口 A8 条件下，检验 head 输出 A8 的代价。
- exp5 对实验 3：在 head 输出 A16 条件下，检验保留主分支入口 A16 的收益。
- exp6 对 exp5：在入口 A16 条件下，检验 head 输出 A8 的代价。
- exp6 对 exp4：在 head 输出 A8 条件下，检验入口 A16 的收益。

这是一个 2×2 配置对照；两项改动的效果可能相互作用，不能直接将单项 PSNR 差值相加预测组合结果。
正式结果需训练完成后获得。当前仍为 Q/DQ 仿真，未测量 FPGA 资源或吞吐。

验证包含逐节点位宽、入口尺度继承、饱和范围初始化、反向梯度、导出/重载一致性及原有配置回归。正式训练由用户手动执行。
