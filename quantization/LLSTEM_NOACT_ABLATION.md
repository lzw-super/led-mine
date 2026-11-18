# ReLU_LLStemNoAct：新模型四组精度对照

## 模型与比较范围

目标 checkpoint：
`experiments/LED_Pretrain_CVPR20_ELD_LearningDWT_Haar_L3_RepNCB_W32_ReLU_LLStemNoAct/models/net_g_latest.pth`

简称 `--model relu_llstem_noact`。配置 `dwt_ll_stem_activation: none` 将 LL stem 后的激活替换为 Identity，其余结构沿用固定 Haar L3 / RepNCB / ReLU。
所有新组使用这个新 checkpoint 和它自己的校准范围，不能借用旧 ReLU 模型的 PTQ 尺度或 FP32 指标。

四个主对照：原始 FP32、修改后的实验 3 QAT、全 W8A16 QAT、全 W8A8 QAT。
三组 QAT 均采用 opt1：卷积和 ReLU 之间不重复舍入，S2D/D2S 纯重排继承编码，Identity 继承输入编码。
LL stem 不再有激活，因此它自身的卷积输出仍需量化，Identity 不增加第二次量化。

## 修改后的实验 3：A16 → A8 → S2D

profile 为 **`phase_exp3_pres2d`**，区别于旧 `phase_exp3`（S2D 后降为 A8）。
实际 FX 执行链为：

```text
最后一层 IDWT (conv_transpose2d_2)
    → Phase 1 输出 A16
    → phase2_input_a8：显式 A16→A8 量化
    → S2D (conv2d_3)：只重排，继承 phase2_input_a8 的 A8 位宽和尺度
    → stem：W8×A8 输入，输出 A8
    → 5 个 RepNCB：W8×A8，输出 A8
    → head：W8×A8 输入，输出 A16
    → concat：输出 A16（noisy S2D 分支继承 RAW A16）
    → fusion：W8×A16 输入，输出 A16
    → D2S：继承 fusion A16
```

`phase2_input_a8` 是参数为空的 Identity 图节点，实际量化由 QAT/静态解释器执行；它不是新增的网络激活函数。
Phase 1 原来的 A16 输出量化保留，然后再转换为 A8，并非直接将 Phase 1 输出节点改成 A8。
部署时这个 A8 转换可在 S2D 前执行；图中不要求必须把中间 A16/A8 完整张量都写入外存。

|位置|修改后实验 3|全 W8A16|全 W8A8|
|---|---|---|---|
|RAW、Phase 1 全部激活|A16|A16|A8|
|Phase 1 输出后的显式转换|A16→A8|无|无|
|主分支 S2D、stem、5 个 block 输出|A8|A16|A8|
|noisy skip S2D 输出|继承 RAW A16|继承 RAW A16|继承 RAW A8|
|head、concat、fusion、最终输出|A16|A16|A8|
|独立激活量化边界数|39|38|38|

所有学习卷积权重均为逐输出通道 signed INT8；激活均为逐张量对称 signed A8/A16。
在相同尺度下，逐张量量化与纯 S2D 排列在数学上可交换；因此仅移动量化位置本身不应被期待带来精度提升。此次修改明确落实所需硬件执行顺序。测试已检查新旧位置在同初始编码下输出一致。

## 初始化和训练预算

FP32 基线先评估。使用该模型专属 32 张训练帧 × 8 个 256² crop 的 absmax 校准，保存到：
`quantization/artifacts/sid_llstem_noact_20261117/relu_llstem_noact_w8a8.pth`。
同时保留全边界静态 PTQ 测试，作为初始化参考；它不是本轮 opt1 QAT 全 W8A8 主对照。

三组 QAT 从同一个原始 FP32 模型初始化，不从旧实验 3 或全 A16 QAT 权重追加训练。
默认 20,000 步、batch 4、crop 256、seed 2026；沿用相同合成噪声数据、原始 FP32 教师、L1 + 0.1 输出 L1 蒸馏和尺度冻结设置。
每个节点依据目标位宽调整初始步长，保持校准截断范围；新 phase2_input_a8 使用 Phase 1 输出的校准范围。

## 手动执行三组 QAT

正式训练由用户执行。在工程根目录运行：

```bash
bash quantization/run_llstem_noact.sh
```

脚本先以 --resume 检查/补齐新模型 FP32、校准和 PTQ 基线；已完成的会跳过。随后依次训练修改后实验 3、全 A16、全 A8，分别自动测试并汇总。
遇到已存在的 QAT 运行会停止，防止覆盖，恢复训练请使用单组命令。

单组命令：

```bash
# 修改后实验 3：S2D 前降为 A8
conda run --no-capture-output -n LED-ICCV23 python quantization/train_qat.py \
  --model relu_llstem_noact --ptq-run sid_llstem_noact_20261117 \
  --boundary-profile opt1 --activation-bits 16 --mixed-profile phase_exp3_pres2d \
  --run-name qat_llstem_noact_phase_exp3_pres2d --eval-after-train

# 整体 W8A16
conda run --no-capture-output -n LED-ICCV23 python quantization/train_qat.py \
  --model relu_llstem_noact --ptq-run sid_llstem_noact_20261117 \
  --boundary-profile opt1 --activation-bits 16 --mixed-profile none \
  --run-name qat_llstem_noact_w8a16 --eval-after-train

# 整体 W8A8
conda run --no-capture-output -n LED-ICCV23 python quantization/train_qat.py \
  --model relu_llstem_noact --ptq-run sid_llstem_noact_20261117 \
  --boundary-profile opt1 --activation-bits 8 --mixed-profile none \
  --run-name qat_llstem_noact_w8a8 --eval-after-train
```

恢复混合组：

```bash
conda run --no-capture-output -n LED-ICCV23 python quantization/train_qat.py \
  --model relu_llstem_noact --ptq-run sid_llstem_noact_20261117 \
  --boundary-profile opt1 --activation-bits 16 --mixed-profile phase_exp3_pres2d \
  --run-name qat_llstem_noact_phase_exp3_pres2d \
  --resume quantization/artifacts/qat_llstem_noact_phase_exp3_pres2d/relu_llstem_noact/qat_latest.pth \
  --eval-after-train
```

其他组恢复时保持各自命令不变，加上对应目录的 --resume checkpoint。

## 保存与对比

- FP32：新 experiment 下 `evaluation/sid_llstem_noact_20261117/fp32/`。
- 全边界 PTQ：同上 `ptq/`，仅作附加参考。
- QAT 训练：`quantization/artifacts/<run-name>/relu_llstem_noact/`。
- 混合模型导出：`qat_w8_mixed_qdq.pth`；全精度组为 `qat_w8a16_qdq.pth` / `qat_w8a8_qdq.pth`。
- QAT 测试：新 experiment 下 `evaluation/<run-name>/qat/`。
- 每组含 summary.csv、per_image.csv、metrics.json、comparison.md（相对新模型 FP32/PTQ）。
- 四组统一汇总：`quantization/runs/llstem_noact_summary.md`。

```bash
conda run --no-capture-output -n LED-ICCV23 python quantization/summarize_llstem_noact.py
```

汇总校验新模型 checkpoint、数据清单、训练设置和 mixed profile，缺失结果标为待完成。

单独恢复混合组测试：

```bash
conda run --no-capture-output -n LED-ICCV23 python quantization/evaluate_qat.py \
  --artifact quantization/artifacts/qat_llstem_noact_phase_exp3_pres2d/relu_llstem_noact/qat_w8_mixed_qdq.pth --resume
```

## 验证和范围

验证新模型无 PReLU、LL stem 后为 Identity、插入的 A8 节点位于 Phase 1 A16 输出与 S2D 之间、S2D 尺度继承、各组梯度、训练更新、导出/重载一致性，以及旧模型 A8/A16/混合配置回归。

当前为浮点内核的 Q/DQ 仿真，不是整数 FPGA 位精确实现；不模拟固定点重定标和累加器溢出。
比较 FP32 与 QAT 时需注意 QAT 额外经历 20k 步训练和蒸馏。正式训练尚未执行，不预判量化指标。

## 当前已完成的 FP32 基线

93 对图像完整评估（LED 校正 RAW）：

|倍率|PSNR dB|SSIM|
|---|---:|---:|
|100|40.401309|0.94961183|
|250|38.434593|0.93016116|
|300|35.972421|0.91691436|
|总体|38.121246|0.93124107|

当前三组 QAT 仅执行了各 3 步、batch 4、256×256 的流程检查，导出检查误差均为 0；不作为正式指标。
模型与逐节点量化审计保存在 `quantization/artifacts/sid_llstem_noact_20261117/llstem_noact_quantization_audit.json`，同目录包含三组 FX 图。
