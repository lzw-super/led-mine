# LearningDWT PReLU / ReLU：W8A8 QAT 训练与测试

此脚本延续已有静态 PTQ baseline 的量化边界和评估口径。完整训练由你执行；目前只完成了少量 step 的工程验证，没有宣称完整 QAT 后 PSNR/SSIM 已提升。

## 1. 直接运行两版训练，结束后自动测试

在 LED-mine 仓库根目录运行。使用已验证的 `LED-ICCV23` 环境，无需安装额外量化框架。

```bash
# PReLU：训练 20000 step，完成后自动评估 93 对 SID 测试图
/opt/conda/envs/LED-ICCV23/bin/python quantization/train_qat.py \
  --model prelu --run-name qat_w8a8_v1 --eval-after-train

# ReLU：使用相同超参数，输出保存到独立的 relu 子目录
/opt/conda/envs/LED-ICCV23/bin/python quantization/train_qat.py \
  --model relu --run-name qat_w8a8_v1 --eval-after-train
```

建议按顺序执行两条命令。去掉 `--eval-after-train` 则只训练、导出，之后用第 4 节命令单独测试。已有同名训练会拒绝覆盖，应使用 `--resume` 或更换 `--run-name`。

## 2. 训练方法与默认配置

| 项目 | 默认值 |
|---|---|
| 浮点主权重初始化 | 对应实验 `models/net_g_latest.pth`，融合 RepNCB 后的部署图 |
| 初始量化尺度 | `quantization/artifacts/sid_w8a8_20261031/{prelu,relu}_w8a8.pth` |
| 权重 | W8，每输出通道对称均匀量化 |
| 激活 | A8，每张量对称均匀量化；含 RAW、Haar、Add、Concat、最终输出等全部 53 个 PTQ 边界 |
| 码域/零点 | `[-127,127]`，zero-point=0；ReLU 后同样用 signed symmetric |
| 量化训练 | round 使用 STE，LSQ 风格缩放步长梯度；正尺度表示为初始尺度 × exp(可训练参数) |
| 尺度更新 | 前 80% step 学习，最后 20% 冻结，推理尺度固定 |
| 步数 / batch / crop | 20000 / 4 / 256×256 packed RAW |
| 优化器 | Adam；主权重 LR=1e-5，尺度 LR=1e-3，余弦衰减至初始值的 10% |
| 损失 | `L1(QAT输出, GT) + 0.1 × L1(QAT输出, 同版FP32教师输出)` |
| 梯度保护 | 全局范数上限 1；非有限 loss/梯度立即报错 |
| 保存/记录 | 每 1000 step 保存 checkpoint，每 50 step 记录训练 CSV |
| 精度 | FP32 训练、关闭 AMP/TF32；启用 cuDNN deterministic |

这里学习的是量化**步长**，不会把量化类型改成非均匀、非对称或逐通道激活。初始化保留既有 absmax PTQ 的尺度，不使用测试集重新校准、分位数搜索或位宽回退。训练结束后直接导出学到的尺度，不重新调用 PTQ `freeze()` 覆盖它们。

bias 前向模拟 INT32（尺度为输入激活尺度 × 权重尺度），保持 STE 以更新浮点主 bias。PReLU 的 alpha 仍是可训练 FP32 参数；固定 Haar / 重排常量保留精确值，和上一轮 PTQ 的例外相同。当前导出仍是 **Q/DQ 精度仿真**，不是全整数 FPGA 位精确运行时。

上一轮 RAW 输入步长约 0.0813，可能导致信息损失。QAT 可以学习更合适的范围并使权重适应量化，但已被输入量化丢失的信息不能仅靠后续层保证恢复；因此是否提升以及提升多少，必须由完整训练后的统一测试确认。

## 3. 数据：默认复用原实验噪声生成方式

默认 `--data synthetic`：

1. 读取 `Sony_train_long_patches` 的 GT 补丁。
2. 按 `datasets/txtfiles/SID/SonyA7S2/train.txt` 中的训练场景白名单筛选。
3. 使用各自 experiment 的 `noise_g.pth`，调用 LED 原有 `VirtualNoisyPairGenerator`，随机选虚拟相机，曝光倍率 100–300，噪声类型 `ptrqc`。
4. 输入保持负值、上限沿用生成器裁剪至 1；GT 归一化流程沿用原工程。随机裁片尺寸必须是 8 的倍数。

本次检查到补丁目录混有非训练名单场景。筛选后实际使用 **1288 个补丁、161 个训练场景**，排除 560 个非训练名单补丁，并断言与测试场景无交集。不会删除或修改任何数据文件。此筛选约束本次 QAT，不追溯改变原始 FP32 权重的训练历史。

可选 `--data paired` 使用官方 `train.txt` 中的真实 noisy/GT 配对进行 QAT；同样排除测试场景。该选择会改变微调数据分布，应使用不同 run-name 并与 synthetic 路线区分记录，不建议第一次同时改变多个设置。

训练过程不计算测试 PSNR/SSIM、不根据测试集选 best checkpoint，也不使用 GT 辅助亮度校正计算训练损失。最终以固定训练 step 的模型进行一次完整评估。每个 step 重新以 `seed+step` 设置随机种子，便于中断恢复采样；跨硬件/软件版本不承诺逐 bit 可复现。

## 4. 单独测试与结果路径

训练结束后可以在同一工程的数据/测试机器上运行：

```bash
/opt/conda/envs/LED-ICCV23/bin/python quantization/evaluate_qat.py \
  --artifact quantization/artifacts/qat_w8a8_v1/prelu/qat_w8a8_qdq.pth --resume

/opt/conda/envs/LED-ICCV23/bin/python quantization/evaluate_qat.py \
  --artifact quantization/artifacts/qat_w8a8_v1/relu/qat_w8a8_qdq.pth --resume
```

自动测试和上述命令一致：默认完整测试 29/29/35 对，即 93 对 SID Sony 图像，RAW 域、整图推理、crop_border=2、关闭 AMP/TF32，使用原 LED 亮度校正。另存未做亮度校正的 PSNR/SSIM，PSNR 的 `MSE+1e-8` 及 SSIM 实现与上一轮完全一致。

输出分别在：

```text
experiments/LED_Pretrain_CVPR20_ELD_LearningDWT_Haar_L3_RepNCB_W32/
  evaluation/qat_w8a8_v1/qat/

experiments/LED_Pretrain_CVPR20_ELD_LearningDWT_Haar_L3_RepNCB_W32_ReLU/
  evaluation/qat_w8a8_v1/qat/
```

每个目录包含 `README.md`、`metrics.json`、`per_image.csv`、`summary.csv`，完整评估后另生成 `comparison.json` / `comparison.md`。增益定义为 **QAT − FP32/PTQ**，正数表示提升。比较前会核对源 checkpoint 哈希与测试清单，防止混用不同基线。

测试时需保留对应原始 FP32 权重、实验配置、上一轮 baseline 报告和 SID NPZ 数据；迁移到其他测试机器时需要一起提供这些依赖。`--max-images-per-ratio 1` 仅用于检查流程，会写入带 `_smoke` 的独立评估目录，不生成正式的全测试集增益报告。

## 5. 断点恢复与调整参数

PReLU 默认训练的恢复示例：

```bash
/opt/conda/envs/LED-ICCV23/bin/python quantization/train_qat.py \
  --model prelu --run-name qat_w8a8_v1 \
  --resume quantization/artifacts/qat_w8a8_v1/prelu/qat_latest.pth \
  --eval-after-train
```

ReLU 把两个 `prelu` 改为 `relu` 即可。恢复时必须保持原有 `--steps`、数据方式、batch、crop、学习率等训练设置；若初次用了自定义参数，恢复时也传入相同参数。checkpoint 保存主权重、可学习尺度和 Adam 状态；早期 checkpoint 恢复会去掉训练 CSV 中比该 checkpoint 更新的记录。

常用可选参数：

```text
--steps 20000             总训练步数（不是新增步数）
--batch-size 4            显存不足时从新 run 改为 2 或 1
--crop-size 256           packed RAW，须为 8 的倍数
--lr 1e-5                主权重学习率
--scale-lr 1e-3          学习尺度的参数学习率
--freeze-scales-at 0.8   训练进度达到 80% 时冻结尺度
--distill-weight 0.1     设为 0 可关闭同版 FP32 教师蒸馏
--data synthetic         或 paired；切换时使用新 run-name
--ptq-run sid_w8a8_20261031
--device cuda:0
```

训练产物在 `quantization/artifacts/<run-name>/<prelu|relu>/`：

- `run.json`：设置、数据筛选、源权重和代码哈希。
- `training.csv`：loss、主权重/尺度 LR、RAW 输入步长、冻结状态等。
- `qat_step_*.pth` / `qat_latest.pth`：用于恢复训练。
- `qat_w8a8_qdq.pth`：训练结束的测试/导出模型，包含 INT8 卷积权重、INT32 bias 和最终静态尺度。
- `export_check.json`：训练图与导出 Q/DQ 图的一致性检查。

已有 `.gitignore` 会忽略 artifacts/runs，脚本和本说明保留版本管理；不会覆盖原始 experiment 的模型或 PTQ baseline。

## 6. 本次验证范围

- 梯度、裁剪 STE、尺度梯度、权重更新、尺度冻结、导出与重载的单元测试。
- PReLU / ReLU 各 3 step 的真实训练数据与原噪声生成器试跑。
- 默认 batch=4、crop=256 的短程显存/运行验证。
- 断点恢复与连续训练参数的数值一致性检查。
- 两版各取每倍率 1 张图，验证测试入口及指标文件写入；这些不是完整 QAT 质量结果。

运行单元测试：

```bash
/opt/conda/envs/LED-ICCV23/bin/python -m unittest quantization.test_qat -v
```

正式训练完成后，再查看 `comparison.md` 判断是否优于之前的 PReLU PTQ 33.9587 dB / 0.806752、ReLU PTQ 33.0723 dB / 0.767462。

## 量化边界消融实验

参见 [BOUNDARY_ABLATION.md](BOUNDARY_ABLATION.md)：保留原始 baseline，提供 opt1（Conv/激活融合、重排继承尺度）、opt2（Haar 分带/逆变换边界）以及 opt12 组合的独立训练与汇总命令。

W8A16 激活位宽对照见 [W8A16_ABLATION.md](W8A16_ABLATION.md)，基于 opt1，训练参数 `--activation-bits 16`。
