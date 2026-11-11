# Phase 1 / Phase 2 混合精度：三个对照实验

三组均采用 opt1（卷积与激活融合，纯重排继承编码），不启用 opt2；学习卷积权重始终为逐通道 signed INT8。
激活逐张量 signed symmetric，A8 为 [-127,127]，A16 为 [-32767,32767]。

## 精确定义

|位置|实验 1 `phase_exp1`|实验 2 `phase_exp2`|实验 3 `phase_exp3`|
|---|---|---|---|
|共享 RAW 输入 `raw`|A16|A16|A16|
|Phase 1 所有激活、最终 IDWT 输出 `conv_transpose2d_2`|A16|A16|A16|
|Refiner 主分支 S2D 输出 `conv2d_3`|A8|A8|A8|
|stem 输出及 RepNCB×5 激活输出|A8|A8|A8|
|noisy skip 的 S2D 输出 `conv2d_4`|A8|A16|A16|
|head 输出|A8|A8|A16|
|拼接 `cat_4` 输出|A8|A8|A16|
|fusion 输出|A8|A8|A16|
|最终 D2S 输出|继承 fusion A8|继承 fusion A8|继承 fusion A16|
|独立激活量化边界数|40|39|39|

共享 raw 必须保持 A16，才能保证 Phase 1 全程 A16。实验 1 的 noisy 分支在自身 S2D 输出处才转 A8，不会把 Phase 1 的输入也降位。
主分支的 Phase 1 输出保留 A16，进入 Phase 2 后在 S2D 输出处转 A8。
这两处是必要的跨精度转换，区别于此前被 opt1 删除的同精度重复重排舍入。

实验 2 的 noisy S2D 输出直接继承 raw 的 A16 尺度，不重新学习一个相同位宽的 S2D 尺度；但是 `cat_4` 仍将拼接结果量化为 A8，所以 noisy 信息在 fusion 入口前仍降为 A8。
该实验检验的是跳过 noisy 分支提前 A8 舍入的作用，并不等于 fusion 全程使用 A16。
实验 3 才让 head、cat、fusion 以及最终输出均使用 A16。

实验 3 的 head **输入**来自最后一个 A8 block，因此 head 仍为 W8×A8 乘法、A16 输出；fusion 输入来自 A16 cat，因此 fusion 为 W8×A16 乘法。
“head 及以后输出 A16”不意味着自动把 head 的输入也提升为 A16，这与用户指定的输出位宽一致。
所有卷积累加器保持原 Q/DQ 宽精度仿真，未改为 8/16 位累加。

## 公平训练

每组从各自原始 FP32 checkpoint 开始；沿用相同 PTQ 校准范围，各节点按目标 qmax 初始化步长。
不从已完成的 A16 checkpoint 追加训练，以保持与现有全 A8、全 A16 各 20k 步实验相同预算。
默认 20,000 步、batch 4、crop 256、seed 2026、相同合成噪声数据与原始 FP32 教师。
损失仍为 L1(GT) + 0.1×L1(teacher)，最后 20% 步冻结尺度；不改变学习率等设置。
训练日志和 artifact 保存 `mixed_profile`、完整 `activation_bit_map`、尺度别名和边界策略，评估自动加载。
混合 artifact 的 `activation_bits=16` 是默认位宽，**每个节点的实际位宽以 `activation_bit_map` 为准**。
加载器同时检查配置和逐节点位宽表；旧版 uniform artifact 保持兼容。

## 手动执行

在工程根目录执行六组训练（两个模型×三个配置），每组完成后自动全量测试：

```bash
bash quantization/run_phase_mixed.sh
```

单独执行 PReLU 的三组：

```bash
conda run --no-capture-output -n LED-ICCV23 python quantization/train_qat.py \
  --model prelu --boundary-profile opt1 --activation-bits 16 \
  --mixed-profile phase_exp1 --run-name qat_phase_exp1 --eval-after-train

conda run --no-capture-output -n LED-ICCV23 python quantization/train_qat.py \
  --model prelu --boundary-profile opt1 --activation-bits 16 \
  --mixed-profile phase_exp2 --run-name qat_phase_exp2 --eval-after-train

conda run --no-capture-output -n LED-ICCV23 python quantization/train_qat.py \
  --model prelu --boundary-profile opt1 --activation-bits 16 \
  --mixed-profile phase_exp3 --run-name qat_phase_exp3 --eval-after-train
```

ReLU 将 `--model prelu` 改为 `--model relu`。三组使用不同运行名；两种模型分别保存在各自子目录。
恢复训练示例：

```bash
conda run --no-capture-output -n LED-ICCV23 python quantization/train_qat.py \
  --model prelu --boundary-profile opt1 --activation-bits 16 \
  --mixed-profile phase_exp1 --run-name qat_phase_exp1 \
  --resume quantization/artifacts/qat_phase_exp1/prelu/qat_latest.pth --eval-after-train
```

恢复时位宽、mixed profile 和原训练设置必须一致；批量脚本遇到已有运行会停止，不会覆盖原结果。

## 输出和比较

- 训练：`quantization/artifacts/qat_phase_exp<N>/<prelu或relu>/`。
- 混合导出文件：`qat_w8_mixed_qdq.pth`。
- 测试：对应 experiment 下 `evaluation/qat_phase_exp<N>/qat/`，含 summary.csv、per_image.csv、metrics.json 和相对 FP32/PTQ 的比较。
- 跨组汇总：`quantization/runs/phase_mixed_summary.md`，包含 FP32、全 A8 opt1、全 A16 opt1、三组混合精度，各倍率及总体指标，并计算相对全 A16 的差值。

```bash
conda run --no-capture-output -n LED-ICCV23 python quantization/summarize_phase_mixed.py
```

单独测试或恢复中断的测试：

```bash
conda run --no-capture-output -n LED-ICCV23 python quantization/evaluate_qat.py \
  --artifact quantization/artifacts/qat_phase_exp1/prelu/qat_w8_mixed_qdq.pth --resume
```

## 验证与限制

测试覆盖两个真实模型×三个配置的逐节点位宽、Phase 1 A16、S2D 转换/继承、最终输出继承、初始截断范围、尺度梯度、训练更新及导出重载一致性，同时回归已有 A8/A16 流程。
正式训练由用户执行，不以冒烟检查的结果作为精度结论。

PReLU α 仍为 FP32，固定 Haar 运算沿用原处理；当前为 Q/DQ 仿真，不模拟 FPGA 的整数累加溢出、定点乘数和移位舍入。
