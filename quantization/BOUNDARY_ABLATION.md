# LearningDWT 量化边界对照实验（PReLU / ReLU）

## 实验目的与分组

本实验改变 Q/DQ 的放置位置，分别评估消除中间舍入与 Haar 分带处理的收益。
不修改原始网络结构、FP32 checkpoint 或已完成的 PTQ / QAT baseline。
训练和评估仍是浮点内核上的量化仿真，不是 FPGA 位精确执行。

每个模型有以下四组，共八组；原始 QAT 已完成，默认只需新训练六组：

|组别 `--boundary-profile`|优化 1|优化 2|独立激活量化边界数|默认 run-name|
|---|---|---|---:|---|
|baseline|关闭|关闭|53|qat_w8a8_v1|
|opt1|开启|关闭|38|qat_ablation_opt1|
|opt2|关闭|开启|47|qat_ablation_opt2|
|opt12|开启|开启|32|qat_ablation_opt12|

### 优化 1：卷积与激活融合、纯重排继承尺度

- 对只有一个消费者、且消费者是 ReLU/PReLU 的学习卷积，删除卷积输出的 A8 舍入。
  原路径 `W8 Conv → Q8 → activation → Q8` 改为 `W8 Conv 宽累加结果 → activation → Q8`。
  共删除 12 处边界；无激活的 stem/head/fusion 等卷积仍保留输出量化。
- 两处 S2D 和一处 D2S 不再重新量化，输出直接继承输入 scale，无独立可训练 scale。
  涉及 preliminary 的 S2D、noisy skip 的 S2D、最终 D2S；包括 CFA 顺序调整。
  重排后进入学习卷积的 bias INT32 编码也使用继承的输入尺度。
- 此处 W8A8 指学习卷积的输入/权重，以及融合算子的存储输出；不要求卷积累加器为 8 位。
  PReLU α 仍为可训练 FP32 参数，此次不引入 α 的定点误差。

### 优化 2：Haar 分带量化与逆 Haar 宽精度求和

- 三层正向 Haar：删除完整 16 通道结果的统一量化，再对 split 后的 LL（4 通道）与 HF（12 通道）分别 A8 量化。
  HF 的 12 通道仍共享一个尺度，未细分 LH/HL/HH；不是逐通道激活量化。
- 三层逆 Haar：删除 LL/HF 拼接结果的统一 A8 量化，保留两个分支原有编码；按各自尺度参与逆变换，最后对逆 Haar 输出 A8 量化。
- 学习卷积前的 LL concat/fusion 和 refiner concat/fusion 仍做统一 A8 量化；HF residual add 输出仍量化。
- FPGA 必须在逆 Haar 中支持分支尺度对齐及宽精度求和，不能把不同尺度的整数直接相加。
  仿真用反量化浮点表达这一操作；尚未模拟定点乘数、移位舍入或累加器溢出。
  因而本组是新的量化边界方案，需要相应硬件数据通路支持，不能当作现有 baseline 硬件上的无条件免费收益。

## 公平比较设置

所有新组从对应原始 `net_g_latest.pth` 的融合后 FP32 权重开始训练；不从已有 20k 步 QAT checkpoint 继续训练。
保留的激活尺度与每通道权重尺度使用同一份 `sid_w8a8_20261031` PTQ artifact 初始化。
移除的量化节点不再拥有尺度参数；S2D/D2S 别名随输入尺度更新。
不重新校准，避免同时改变校准过程。故这是从同一初始化出发、相同训练预算的边界消融实验，并非重新优化校准后的 PTQ 对比。

默认训练参数与原始 QAT 相同：20,000 步、batch 4、256×256 crop、seed 2026、权重 LR 1e-5、尺度 LR 1e-3、余弦衰减、最后 20% 步数冻结尺度。
数据采用原来的合成噪声流程，仅允许 SID train 场景；每步重新设置 RNG，使各组获得相同训练样本与噪声。
损失仍为 `L1(student, GT) + 0.1 × L1(student, frozen FP32 teacher)`，教师为同一模型的原始 FP32 网络。
均采用 signed symmetric [-127,127]、权重逐输出通道、激活逐张量；没有混合 A16、没有 unsigned ReLU、没有新增损失。

已有 QAT baseline：

|模型|FP32 PSNR / SSIM|原始 QAT PSNR / SSIM|
|---|---|---|
|PReLU|38.272141 / 0.93165700|37.006187 / 0.90615722|
|ReLU|38.049337 / 0.93055644|36.735729 / 0.90510460|

数值采用现有 LED 校正后的 RAW 指标、完整 93 对图像。新组正式结果需要训练后获得。

## 执行命令

在工程根目录运行，顺序训练六组，每组训练完成后自动全量评估：

```bash
bash quantization/run_boundary_ablation.sh
```

只运行一组，例如 PReLU + 优化 1：

```bash
conda run --no-capture-output -n LED-ICCV23 python quantization/train_qat.py \
  --model prelu --boundary-profile opt1 \
  --run-name qat_ablation_opt1 --eval-after-train
```

`--model` 可取 `prelu/relu`；`--boundary-profile` 可取 `opt1/opt2/opt12`。
保持各组其余超参数一致。运行名按配置区分，模型各自位于子目录，不会互相覆盖。
若仅想先试最可能受益的组合，可先分别执行两模型的 `opt12`。

中断后恢复某一组（其余参数需与该组原设置一致）：

```bash
conda run --no-capture-output -n LED-ICCV23 python quantization/train_qat.py \
  --model prelu --boundary-profile opt1 --run-name qat_ablation_opt1 \
  --resume quantization/artifacts/qat_ablation_opt1/prelu/qat_latest.pth \
  --eval-after-train
```

批处理脚本遇到已存在的运行或失败会停止，防止意外覆盖；中断后请逐组恢复或执行尚未运行的组。
若训练完成但评估中断，可以单独恢复评估，不必再次训练：

```bash
conda run --no-capture-output -n LED-ICCV23 python quantization/evaluate_qat.py \
  --artifact quantization/artifacts/qat_ablation_opt1/prelu/qat_w8a8_qdq.pth --resume
```

## 保存与汇总

- 训练权重/日志/导出：`quantization/artifacts/qat_ablation_<profile>/<model>/`。
- `run.json` 中记录 profile、保留边界、移除原因、尺度别名、源文件哈希及训练设置。
- 导出 artifact 同样保存完整 boundary policy，加载时自动恢复并校验；旧版无 policy 的 artifact 按 baseline 兼容加载。
- 指标：各原始 experiment 目录下 `evaluation/qat_ablation_<profile>/qat/`，含 PSNR/SSIM、逐图记录及相对 FP32/PTQ 的比较。
- 训练完成后汇总新组与原始 QAT 的差异：

```bash
conda run --no-capture-output -n LED-ICCV23 python quantization/summarize_ablation.py
```

默认输出 `quantization/runs/boundary_ablation_summary.md`；缺失/未完成组显示“待完成”，不会编造指标。
汇总会检查权重与数据集来源一致。自定义运行名前缀时使用 `--prefix`，原始 QAT 名称使用 `--qat-baseline-run`。
既有 `.gitignore` 已覆盖 artifacts、runs 和 experiments；新增启动脚本的白名单，覆盖仓库原有 `*.sh` 忽略规则。

## 已完成验证与范围

测试覆盖两个实际模型的全部四种 profile：边界数量、尺度继承、原始 baseline 前向兼容、尺度反向梯度、一次优化更新，以及多种幅度输入下的 QAT/导出/重新加载一致性。
另有六组各两步真实合成训练数据的 GPU 冒烟检查，只验证流程，不作为精度结论。
测试命令：

```bash
conda run --no-capture-output -n LED-ICCV23 python -m unittest \
  quantization.test_static_ptq quantization.test_qat quantization.test_boundary_ablation
```

本次不执行正式训练或根据测试集选择超参数。完整实验后再比较 opt1、opt2 与 opt12，判断各项独立收益及组合是否存在相互作用。
