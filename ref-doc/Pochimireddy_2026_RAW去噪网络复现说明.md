# Pochimireddy 等（CVPR 2026）RAW 去噪网络复现说明

## 1. 实现范围

本实现复现论文 *Efficient Real-Time Raw-to-Raw Denoising for Extreme Low-Light Ultra HD Video on Mobile Devices* 的单帧基础网络、训练态结构重参数化模块，以及 RAW 重建/色差联合损失，并接入本仓库现有的 SID Sony A7S2 数据管线。

按照当前任务的训练和数据特点，以下部分不实现：

- 网络内部的 Space-to-Depth（S2D）与 Depth-to-Space（D2S）；
- 双帧输入和多帧卷积；
- 知识蒸馏、INT16 量化、分辨率重构及移动端打包部署；
- 论文自建的合成视频/实拍视频数据生成流程。

SID 数据加载器已经将 Bayer 图像打包成四通道 patch。这个既有的 `[R, G1, B, G2]` 数据表示仍然保留，但网络内部不再额外执行论文的 S2D/D2S。

## 2. 结构对应关系

论文图 6 的单帧结构在本实现中的映射如下：

| 论文结构 | 当前实现 | 说明 |
| --- | --- | --- |
| S2D | 省略 | 直接输入四通道 SID packed RAW |
| 浅层 3×3 Conv | `MRLFN.shallow_conv` | 4 通道映射到特征深度 `d` |
| `N` 个 mRLFB | `MRLFN.blocks` | 默认采用论文 Model A：`N=4, d=32` |
| 深层 3×3 Conv | `MRLFN.deep_fusion` | 聚合级联块输出 |
| 浅层 1×1 Conv | `MRLFN.shallow_fusion` | 保留浅层纹理信息 |
| Concat + 1×1 Conv | `MRLFN.output_conv` | 输出四通道 clean RAW |
| D2S | 省略 | 输出与输入 patch 同尺寸 |

单个 mRLFB 严格保留图 6 的两个块内跳连。令块输入为 `x`，三个重参数卷积的连续输出为 `f3`，则：

```text
f3  = RepConv3(RepConv2(RepConv1(x)))
mid = f3 + x
out = Conv1x1(mid) + x
```

这里没有额外添加输入 RAW 到最终 clean RAW 的全局残差；论文图中可见的残差连接位于 mRLFB 和特征融合路径内。

## 3. 结构重参数化

训练时，每个 mRLFB 中原本的 `3×3 Conv + ReLU` 使用论文图 7 的多分支模块替代：

```text
y = ReLU(x + Conv1x1_b(Conv3x3(x + Conv1x1_a(x))))
```

推理前将其解析融合为：

```text
y = ReLU(Conv3x3_fused(x))
```

设两个 1×1 权重分别为 `Wa`、`Wb`，中间 3×3 权重为 `W3`，则先把局部跳连写成 `A = I + Wa`，再进行卷积核复合，最后在中心位置加入全局恒等核：

```text
Kmid   = W3 ○ A
Kfused = Wb ○ Kmid + I_center
bfused = Wb · b3 + bb
```

其中 `○` 表示连续卷积的核合成。为了使零填充边界也能严格等价，`Conv1x1_a` 不使用 bias；位于 3×3 之后的 bias 可以精确融合。测试会同时检查边界像素，当前 float32 前向的最大误差处于 `1e-6` 量级。

网络提供两种转换接口：

- `network.deploy()`：返回已融合的副本，不破坏训练中的网络及优化器状态；
- `network.switch_to_deploy()`：原地删除训练分支并替换为单个 3×3 Conv。

`RAWImageDenoisingModel.save()` 已扩展为识别所有提供 `deploy()` 的网络。训练 checkpoint 会同时保存：

- `params`：训练态多分支权重，可继续训练；
- `params_deploy`：推理态单 3×3 权重，不能直接恢复训练分支。

加载 `params_deploy` 时必须使用 `network_g.deploy: true`，以保证模型结构和 state dict 的键一致。

## 4. Loss

论文的最终目标为：

```text
L = 0.6 * Lraw + 0.4 * Lchromatic
```

`Lraw` 是预测 packed RAW 与 GT 的 L1。色差项先根据 SID 的 `[R, G1, B, G2]` 顺序计算：

```text
Gavg = (G1 + G2) / 2
DBG  = B - Gavg
DRG  = R - Gavg
Lchromatic = L1(DBG_pred, DBG_gt) + L1(DRG_pred, DRG_gt)
```

实现类为 `RawReconstructionChromaticLoss`。为与仓库现有 loss 的数值尺度和 patch 尺寸解耦，`reduction: mean` 会先对每个样本的空间/通道元素取均值，再对 batch 取均值；两个色差分量分别取空间均值后相加。这样 256 与 1024 patch 不会仅因像素数变化而改变梯度尺度。

## 5. LED 多虚拟相机 + SID clean RAW 训练配置

训练入口：

```bash
python led/train.py -opt options/LED/pretrain/MRLFN_CVPR20_ELD_Setting.yaml
```

该配置参考 `options/LED/pretrain/CVPR20_ELD_Setting.yaml`，使用的不是 SID short/long 成对训练方式，而是 LED 的虚拟相机预训练方式：

```text
SID long-exposure clean RAW patch
        ↓ RAWGTDataset
随机曝光 ratio（100–300）
        ↓ VirtualNoisyPairGenerator
从 IC0–IC4 随机选择一台虚拟相机，在线生成 ptrqc noisy RAW
        ↓ MRLFN
预测 clean RAW，并计算重建/色差联合 loss
```

相关数据与噪声设置如下：

- clean 数据：`datasets/ICCV23-LED/Sony_train_long_patches`；
- 数据类型：`RAWGTDataset`，读取 SID 长曝光 clean RAW；
- patch：数据准备阶段已生成 `4×512×512` packed-RAW patch，训练时不再随机裁剪；
- ratio：每个样本在 `[100,300]` 内随机采样；
- 虚拟相机数量：5，对应 `IC0`–`IC4`；
- 噪声类型：`ptrqc`，包含 photon shot、Tukey-Lambda read、row、quantization 和 color-bias noise；
- 虚拟相机参数：从 `VirtualNoisyPairGenerator_ELD_ptrqc_5VirtualCameras.pth` 加载；
- 每个 iteration 随机选择一个 `camera_id`，同一 batch 使用该虚拟相机生成 noisy/clean pair；
- MRLFN 本身没有 RepNR 的相机专属 CSA 分支，因此不同虚拟相机共同更新同一套 MRLFN 权重。

网络、训练和 loss 配置如下：

- 输入/输出：4 通道 packed RAW；
- 特征深度：`d=32`；
- mRLFB 数量：`N=4`；
- patch：`512×512` packed RAW；
- batch size：1；
- optimizer：Adam，初始学习率 `1e-4`；
- scheduler：沿用 LED CVPR20 配置的 `HandieLR`，在 128800/231840 iteration 将学习率调整为 `5e-5/1e-5`；
- loss 权重：`wr=0.6, wc=0.4`；
- 总迭代数：257600。

训练启动时会通过 THOP 统计 `1×4×512×512` 输入：默认 `N=4,d=32` 的训练态为 151,908 参数、39.561 GMACs；融合部署态为 126,948 参数、33.118 GMACs（66.236 GFLOPs）。

论文原始实验使用 256×256 patch、batch size 16 和 cosine schedule；当前配置为了接入 LED 已准备好的 SID clean patch 与虚拟相机流程，采用 LED 的 512×512、batch size 1、HandieLR 和 257600 iteration。这些是明确的数据/训练适配，不是论文原始超参数。

如需减小显存，只改变 batch size 没有意义，因为默认已经是 1；应重新准备较小的 SID clean patch，或者给 `RAWGTDataset` 指定小于已有 patch 的 `crop_size`。例如在 512 patch 内随机裁出 256 patch：

```bash
python led/train.py \
  -opt options/LED/pretrain/MRLFN_CVPR20_ELD_Setting.yaml \
  --force_yml datasets:train:crop_size=256
```

论文 Model B 可通过把 `network_g:feature_channels` 改为 16 获得；当前默认不做教师蒸馏，只训练对应宽度的单模型。

## 6. 不同 N/d 配置的模型规模与计算量

### 6.1 论文配置与本节扩展配置

论文明确给出了以下两种主干宽度配置：

| 论文名称 | N | d | 论文中的其他处理 | 本复现中的对应关系 |
| --- | ---: | ---: | --- | --- |
| Model A / 单帧 Model A* | 4 | 32 | `k=4`，并作为蒸馏 teacher | 默认训练配置；本复现省略网络内部 S2D/D2S |
| Model B / 单帧 Model B* | 4 | 16 | `k=4`，由 Model A 蒸馏得到 | 可直接把 `feature_channels` 改为 16，但当前不执行蒸馏 |
| GT 生成大模型 | 16 | 未说明 | 用于消除 burst average 后的残余噪声 | 论文没有报告 `d`，不能唯一确定其参数量和计算量 |

Model AQ、Model BRQ 不会改变基础 `N/d` 的含义，但涉及量化或分辨率重构，不属于本次统计。为了展示深度与宽度的缩放规律，下面额外扫描 `N∈{2,4,8,16}`、`d∈{16,32}`。除 `N=4,d=16/32` 外，其余组合均是当前实现上的工程扩展，不应写成论文报告过的模型。

### 6.2 统计口径与解析公式

为便于与论文报告的 patch 尺寸比较，本节参数扫描统一采用 `1×4×256×256` packed RAW；该 patch 对应 `512×512` 个传感器 Bayer 样本。新建的 LED 虚拟相机训练配置实际使用 512×512 packed patch，其卷积 MACs/GFLOPs 是下表同一 `N/d` 项的 4 倍，参数量不变。统计约定如下：

- 参数量包含卷积 weight 和 bias；
- MACs 只统计 `Conv2d` 乘加，与仓库 THOP 日志口径一致，不计 ReLU、逐元素加法和 Concat；
- `1 MAC = 2 FLOPs`，因此表中的 GFLOPs 是 GMACs 的两倍；
- 训练态包含图 7 的三个卷积分支，部署态已将每个重参数模块融合为单个 3×3 Conv；
- FP32 权重体积按 `部署参数量 × 4 Byte / 2^20` 计算，不包含框架、激活、优化器和临时 buffer；
- `MAC/RAW pixel` 的分母是 `512×512` 个传感器样本，而非 `256×256` 个 packed 像素；
- 理论最长感受野只考虑最长的 3×3 卷积路径。当前没有 S2D，故为 `(6N+5)×(6N+5)` packed 像素。

令 packed patch 高宽为 `H,W`，在当前四通道输入/输出、所有卷积 stride=1 的实现下，可以得到：

```text
P_train  = (34N + 10)d² + (7N + 47)d + 4
P_deploy = (28N + 10)d² + (4N + 47)d + 4

MAC_train(H,W)  = HW[(34N + 10)d² + 44d]
MAC_deploy(H,W) = HW[(28N + 10)d² + 44d]
```

这些公式已使用逐层 `Conv2d` forward hook 对下列全部配置进行核对，解析结果和实际执行图的卷积 MAC 计数完全一致。

### 6.3 参数量比较

| 配置来源 | N | d | 重参数模块数 `3N` | 训练态参数 | 部署态参数 | 参数下降 | 部署 FP32 权重 | 最长感受野（packed） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 扩展配置 | 2 | 16 | 6 | 20,948 | 17,780 | 15.1% | 0.068 MiB | 17×17 |
| 扩展配置 | 2 | 32 | 6 | 81,828 | 69,348 | 15.3% | 0.265 MiB | 17×17 |
| 论文 Model B* 同结构 | 4 | 16 | 12 | 38,580 | 32,244 | 16.4% | 0.123 MiB | 29×29 |
| 论文 Model A* 同结构（默认） | 4 | 32 | 12 | 151,908 | 126,948 | 16.4% | 0.484 MiB | 29×29 |
| 扩展配置 | 8 | 16 | 24 | 73,844 | 61,172 | 17.2% | 0.233 MiB | 53×53 |
| 扩展配置 | 8 | 32 | 24 | 292,068 | 242,148 | 17.1% | 0.924 MiB | 53×53 |
| 扩展配置（非论文 GT 模型定值） | 16 | 16 | 48 | 144,372 | 119,028 | 17.6% | 0.454 MiB | 101×101 |
| 扩展配置（非论文 GT 模型定值） | 16 | 32 | 48 | 572,388 | 472,548 | 17.4% | 1.803 MiB | 101×101 |

### 6.4 256×256 packed patch 计算量比较

| 配置来源 | N | d | 训练态 GMACs | 部署态 GMACs | 部署态 GFLOPs | MACs 下降 | 部署态 MAC/RAW pixel |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 扩展配置 | 2 | 16 | 1.355 | 1.153 | 2.307 | 14.9% | 4,400 |
| 扩展配置 | 2 | 32 | 5.327 | 4.521 | 9.043 | 15.1% | 17,248 |
| 论文 Model B* 同结构 | 4 | 16 | 2.496 | 2.093 | 4.186 | 16.1% | 7,984 |
| 论文 Model A* 同结构（默认） | 4 | 32 | 9.890 | 8.280 | 16.559 | 16.3% | 31,584 |
| 扩展配置 | 8 | 16 | 4.777 | 3.972 | 7.944 | 16.9% | 15,152 |
| 扩展配置 | 8 | 32 | 19.017 | 15.796 | 31.591 | 16.9% | 60,256 |
| 扩展配置（非论文 GT 模型定值） | 16 | 16 | 9.341 | 7.730 | 15.460 | 17.2% | 29,488 |
| 扩展配置（非论文 GT 模型定值） | 16 | 32 | 37.271 | 30.828 | 61.656 | 17.3% | 117,600 |

### 6.5 SID 整帧计算量外推

SID Sony A7S2 的完整网络输入为 `1×4×1424×2128` packed RAW，对应 `2848×4256` 传感器 RAW。因为当前网络全程保持空间分辨率，卷积 MACs 与 `H×W` 成正比。对论文两个主干配置的部署态外推如下：

| 对应结构 | 部署态参数 | 256² patch | SID 整帧 GMACs | SID 整帧 GFLOPs | 理想 30 fps 算术吞吐需求 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Model B*：`N=4,d=16` | 32,244 | 2.093 GMACs | 96.775 | 193.550 | 5.806 TFLOP/s |
| Model A*：`N=4,d=32` | 126,948 | 8.280 GMACs | 382.832 | 765.665 | 22.970 TFLOP/s |

这里的整帧结果是当前“省略 S2D/D2S”版本的理论卷积量，不能与论文采用 `k=4` S2D、分辨率重构、FP16/INT16 和特定 NPU 后报告的手机延迟直接比较。它也不包含 tiled inference 的 overlap 重复计算；若使用 crop/merge，实际 MACs 还要乘以脚本报告的 `processed_pixel_ratio`。

### 6.6 配置选择结论

- `d` 是主要计算量杠杆。固定 `N=4` 时，从 `d=16` 增至 `d=32`，部署参数由 32,244 增至 126,948（约 3.94 倍），patch MACs 由 2.093 增至 8.280 GMACs（约 3.96 倍），符合卷积复杂度近似随 `d²` 增长的规律。
- `N` 基本呈线性增长。固定 `d=32` 时，从 `N=4` 增至 `N=8`，部署参数和 MACs 分别约增至 1.91 倍。
- 重参数化在这些配置上减少约 15%–17% 的参数和 MACs，更重要的收益是把每个训练模块的 `1×1→3×3→1×1` 与两次 skip 变成单个 3×3 Conv，从而减少中间特征读写和算子调度；真实延迟收益不能只由 MACs 下降比例推断。
- 若优先验证训练流程或显存/吞吐受限，建议先用 `N=4,d=16`；若优先贴近论文 Model A 的容量，则使用默认 `N=4,d=32`。`N=16` 更适合研究大感受野或教师/GT 生成模型，不适合作为本次无 S2D 的整帧实时默认值。
- 参数表不等于训练显存表。训练态还会保存多分支激活和反向梯度，实际峰值显存需在目标 batch、混合精度设置和 GPU 上实测。

## 7. 部署态验证

训练保存的 `params_deploy` 可用部署态配置验证：

```bash
python led/test.py \
  -opt options/SID/MRLFN_SonyA7S2_deploy.yaml \
  --force_yml path:pretrain_network_g=experiments/LED_Pretrain_MRLFN_CVPR20_ELD_Setting/models/net_g_latest.pth
```

部署配置只包含 SID 验证集，不会把训练集误当成测试集遍历；其 `param_key_g` 已设为 `params_deploy`。

## 8. 文件清单

- `led/archs/mrlfn_arch.py`：MRLFN、mRLFB、可重参数 3×3 模块；
- `led/losses/mrlfn_loss.py`：RAW 重建与色差联合 loss；
- `options/base/network_g/mrlfn.yaml`：网络基础配置；
- `options/LED/pretrain/MRLFN_CVPR20_ELD_Setting.yaml`：SID clean RAW + 5 台虚拟相机的 LED 预训练配置；
- `options/SID/SID_SonyA7S2_CVPR26_MRLFN_Setting.yaml`：可选的真实 SID short/long 成对训练配置；
- `options/SID/MRLFN_SonyA7S2_deploy.yaml`：融合权重验证配置；
- `tests/test_mrlfn.py`：结构、loss、严格加载与融合等价性测试。

## 9. 已知差异与后续工作

当前结果应被理解为论文基础网络在 SID 单图 patch 训练上的结构复现，而非论文完整 UHD 视频移动端指标复现。若后续目标转向端侧实测，还需要补充 S2D/D2S、双帧输入、论文的分辨率重构、蒸馏、量化校准以及对应 NPU 算子评测；这些步骤会改变延迟/功耗，但不影响本次实现的单帧基础网络和 3×3 融合逻辑。
