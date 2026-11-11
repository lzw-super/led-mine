# LED 静态 W8A8 PTQ baseline

本目录评估五个实验的 `models/net_g_latest.pth`，并为 LearningDWT PReLU / ReLU 两版生成静态 PTQ 精度基线。模型原始权重不修改。

## 评估口径

- 工程默认 LED SID Sony 测试配置：`options/base/dataset/test/SID_SonyA7S2_val_split.yaml`。文件名沿用 `valid_*`，即本仓库训练/测试入口共用的评估划分；不是额外新建的独立测试集。
- 100/250/300 倍率分别 29/29/35 对，共 **93 对**。文本末行无换行，不应以 `wc -l` 当作样本数。
- 使用本地 `Sony_npy` 的完整 packed RAW，尺寸 4×1424×2128；不做 tile/crop 推理、不启用 AMP/TF32。
- RAW PSNR/SSIM，边界裁剪 2 像素，逐图计算后平均；`all` 按图数加权，并非倍率等权。
- 主指标沿用 LED `IlluminanceCorrect`：预测先 clamp 到 [0,1]，排除 GT=1 的位置估计全局缩放，缩放后不再次裁剪。这是 GT 辅助评估步骤，不是部署前处理。
- 同时提供 uncorrected 指标：原模型输出，不施加亮度校正或输出裁剪。输入/GT 的归一化、曝光比和截断仍沿用 `PairedRAWDataset`。
- PSNR 严格沿用 LED tensor 版本的 `MSE+1e-8`。SSIM 使用仓库官方 float64 NumPy/OpenCV 实现，不转 uint8；脚本启动时与官方 tensor 版本比较通过后才评估。
- 原始 LED RepNR 网络先 `generalize()` 平均虚拟相机分支再部署融合，匹配仓库验证行为。其余模型使用各自保存配置，strict-load、eval、deploy。

## PTQ 定义与边界

只量化两个 LearningDWT 版本；先融合 RepNCB，再使用静态 mobile Haar 图，并在启动时对照部署图检查数值一致性。

- 卷积权重：每输出通道 absmax；激活：每张量 absmax；对称均匀量化；zero-point=0，码域 **[-127,127]**；round-to-nearest ties-to-even。ReLU 输出也保持 signed symmetric，不改用 unsigned。
- 训练集真实 noisy RAW 校准，32 张按曝光比轮转抽样的图像，每张 8 个 256×256、8 像素对齐 crop，种子 2026；冻结尺度后评估。自动检查校准训练场景与测试场景不相交。
- 校准观察浮点图的 tensor 节点输出，随后冻结；不迭代优化，不使用测试图校准。逐张量包含输入、Conv、激活、各 split 子张量、Add、Concat、固定 Haar 和空间重排算子的输出，共 53 个边界。常量与 tuple 本身不量化。
- Concat 后统一尺度；这是一种明确且保守的独立节点 Q/DQ baseline，未做算子融合消除重复舍入。输入/输出、Haar 和残差接口也不保留 A16。
- 可学习 Conv 权重保存为 INT8；bias 按 `input_scale*weight_scale[channel]` 取整保存 INT32，同时检查溢出。
- **PReLU 斜率和固定 Haar/重排常量保留 FP32**，数据激活仍量化。固定 ±0.5/0/1 是精确常量；PReLU 是明确记录的浮点算子例外，不能称为全部参数或全部算子都 INT8。
- **执行形式是 FP32 卷积/算子 + 显式 Q/DQ 的量化精度仿真**，不使用真正整数卷积累加或整数 multiplier/shift。它建立量化误差 baseline，不是 bit-exact FPGA 模型，也不测 INT8 加速比。

`static_ptq.py` 用 FX Interpreter 执行上述语义，冻结后的尺度不会随测试图变化。导出的 `.pth` 同时保存 dequantized 图状态、整数卷积参数、activation scales、FX 图文本和校准清单。`load_artifact()` 可重新装载并验证图一致性；重建时须使用同一模型配置和源码。

## 执行

在仓库根目录，使用已安装 LED 依赖的环境：

```bash
/opt/conda/envs/LED-ICCV23/bin/python -m unittest quantization.test_static_ptq -v
/opt/conda/envs/LED-ICCV23/bin/python quantization/run_baseline.py --models prelu relu led mrlfn splitter --modes fp32 --resume
/opt/conda/envs/LED-ICCV23/bin/python quantization/run_baseline.py --models prelu relu --modes ptq --resume
/opt/conda/envs/LED-ICCV23/bin/python quantization/summarize_baseline.py
```

默认 run-name 是 `sid_w8a8_20261031`。`--resume` 用于同配置中断恢复；修改校准规模、量化逻辑或源码时必须使用新的 run-name，不应复用旧量化文件。`--max-images` 仅供试跑，结果状态为 `smoke_complete`，不能混入完整报告。

## 产物

- 每个实验：`evaluation/<run-name>/fp32/` 下保存 `README.md`、`metrics.json`、`per_image.csv`、`summary.csv`。
- 两个 LearningDWT 实验另有 `ptq/` 同类指标、`quantization_scales.json`，以及同级 `ptq_vs_fp32.json`。
- `quantization/artifacts/<run-name>/`：两份 PTQ checkpoint、校准清单（在 checkpoint 元数据内）、resolved YAML、FX 图、对比 CSV 和源码哈希清单。
- `quantization/BASELINE_RESULTS.md`：全部完成后生成的五模型比较及两版 PTQ 损失表。损失定义为 FP32−PTQ，正数表示下降。
- `quantization/runs/`：过程日志。

`.gitignore` 忽略 artifacts/runs；实现脚本、测试与说明文档保留版本管理。实验目录沿用工程已有忽略规则。原始 experiment 权重不被覆盖。

## QAT 后续实验

两版 LearningDWT 的训练、断点恢复与训练后自动测试见 [QAT 使用说明](QAT_README.md)。入口为 `train_qat.py` 和 `evaluate_qat.py`；完整训练由用户启动，原 PTQ baseline 保留不变。

## 量化边界消融实验

参见 [BOUNDARY_ABLATION.md](BOUNDARY_ABLATION.md)：保留原始 baseline，提供 opt1（Conv/激活融合、重排继承尺度）、opt2（Haar 分带/逆变换边界）以及 opt12 组合的独立训练与汇总命令。
