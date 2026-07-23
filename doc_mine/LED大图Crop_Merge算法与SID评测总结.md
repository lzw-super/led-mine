# LED 大图 Crop–Merge 算法与 SID 评测总结

## 1. 研究背景

图像去噪模型通常需要处理高分辨率输入。直接将整幅图像送入网络虽然实现简单，
但会产生较大的显存和硬件缓存需求，不利于资源受限的 GPU、FPGA 或 ASIC 部署。

为降低单次推理的存储压力，本次为 LED RAW 去噪模型实现了 crop–merge 流式处理：

1. 将大图切分成多个较小的 patch；
2. 相邻 patch 保留重叠区域，为卷积提供足够的上下文；
3. 软件实现中逐 patch 循环推理，硬件实现中可按相同顺序流式处理；
4. 对 patch 输出的边缘或重叠区域做额外处理；
5. 将各 patch 的可靠输出重新拼接为完整图像。

使用的模型权重为：

```text
experiments/LED_Pretrain_CVPR20_ELD_Setting/models/net_g_latest.pth
```

评测数据为 SID Sony A7S2 验证集，共 93 张 packed RAW 图像，包括 100×、250×
和 300× 三种曝光倍率。

## 2. 卷积边缘效应

如果直接对图像进行无重叠切块，每个内部 patch 的边缘都会被网络误认为是整幅
图像的边界。卷积 padding 产生的错误会随着网络层数增加向 patch 内部传播，
最终表现为：

- patch 接缝处亮度或颜色不连续；
- 输出出现规则网格；
- 分块推理结果与整图推理结果存在 PSNR 偏差；
- 网络越深、感受野越大，需要保留的上下文越多。

LED 当前使用的 U-Net 包含四次 2× 下采样，整体下采样倍数为 16。因此，除了
保留重叠区域，还必须使 patch 起点和尺寸与 16 对齐，否则 MaxPool 和反卷积的
采样相位会与整图推理不同。

## 3. 实现的重叠融合方式

本次实现支持三种方法。

### 3.1 Uniform 平均融合

相邻 patch 的完整输出在重叠区域直接求平均：

```text
output = sum(patch_output) / overlap_count
```

该方式实现简单，但 patch 边缘受到 padding 污染时，错误边缘仍然会进入最终
输出，只是被平均后稍微减弱。

### 3.2 Hann 加权融合

对 patch 边缘使用 raised-cosine/Hann 权重。靠近 patch 中心的输出权重大，
靠近内部边缘的输出权重小，所有覆盖同一像素的权重经过归一化后再求和。

Hann 融合通常比硬平均更平滑，适用于无法提供足够上下文的情况，但它仍然会
保留少量受边缘效应影响的输出。

### 3.3 Overlap-Crop

最终选用 overlap-crop，也可称为 halo exchange：

- 相邻输入 patch 保留重叠区域；
- 重叠部分作为网络计算所需的上下文 halo；
- 网络处理完成后，丢弃内部 patch 边缘的不可靠输出；
- 每个 patch 只向最终图像写入可靠中心区域；
- 整幅图像最外侧仍保持原始网络的边界处理方式。

该方法不是对错误边缘做平滑，而是直接阻止错误边缘进入最终图像。当 halo
覆盖网络感受野时，理论上可以得到与整图推理一致的输出。

## 4. 最终参数

对当前 LED RepNR U-Net，推荐配置为：

```text
tile_size = 512
overlap   = 192
halo      = 96
core      = 320
blend     = crop
alignment = 16
```

其中：

```text
halo = overlap / 2 = 96
core = tile_size - overlap = 320
```

内部 patch 的输入大小最大为 512×512，真正写入最终输出的是 320×320 中心
区域，四周各 96 像素仅用于提供上下文。

采用 96 像素 halo 的原因如下：

1. 当前四层 U-Net 的感受野分析表明，stride-16 对齐后，96 像素可以覆盖被
   保留输出所需的边界上下文；
2. 96、192、320 和 512 都是 16 的整数倍，可保持下采样相位一致；
3. SID 实测中，overlap=192 已达到与整图输出近乎严格一致的结果；
4. 继续增加到 overlap=256 不能带来有意义的精度提升，只会增加重复计算。

对于不是 16 倍数的输入，算法先按照 LED U-Net 相同的 reflect 方式进行全局
对称 padding，分块合并后再裁剪回原始尺寸，避免最后一个 patch 破坏相位。

## 5. 不同策略对比

在 SID 的 15 张样本上，对不同重叠和融合策略进行了扫描：

| 512 patch 策略 | 平均 PSNR 损失 | 输出相对整图 PSNR |
| --- | ---: | ---: |
| Crop，overlap=128，halo=64 | 0.019285 dB | 68.34 dB |
| Hann，overlap=128 | 0.016514 dB | 70.02 dB |
| Crop，overlap=192，halo=96 | 2.60e-8 dB | 168.63 dB |
| Hann，overlap=192 | 0.003307 dB | 75.77 dB |

由结果可见：

- overlap 不足时，Hann 融合通常比直接裁边或硬平均更平滑；
- overlap=192 时，Hann 仍会引入少量边缘误差；
- overlap-crop 192 已将算法误差降低到浮点误差量级；
- 对当前模型，保证足够 halo 后裁掉边缘，是 PSNR 损失最小的方法。

因此最终采用 `512 + overlap-crop 192`。

## 6. SID 完整验证结果

PSNR 在 linear packed RAW 空间计算，保持 LED 原验证配置：

- 全图合并完成后再执行 illumination correction；
- metric crop border 为 2；
- RepNR 权重先平均五个虚拟相机 alignment 分支，再融合成普通 U-Net 推理；
- 整图和 crop–merge 使用相同模型、权重和评测流程。

| SID 倍率 | 图像数 | 整图 PSNR | Crop–Merge PSNR | PSNR 损失 |
| --- | ---: | ---: | ---: | ---: |
| 100× | 29 | 42.008605 dB | 42.008605 dB | 1.62e-8 dB |
| 250× | 29 | 39.242763 dB | 39.242763 dB | 1.39e-8 dB |
| 300× | 35 | 36.696812 dB | 36.696812 dB | 8.66e-9 dB |
| 总体 | 93 | 39.147076 dB | 39.147076 dB | 1.26e-8 dB |

Crop–merge 输出与整图输出之间的直接误差为：

| 指标 | 结果 |
| --- | ---: |
| 输出相对整图 PSNR | 168.62 dB |
| 平均绝对误差 MAE | 9.12e-10 |
| 最大绝对误差 | 4.77e-7 |

因此，在当前模型和 FP32 推理条件下，推荐的 crop–merge 输出可以视为与整图
输出等价。

## 7. 显存与速度

测试硬件为 NVIDIA GeForce RTX 2080 Ti：

| 推理方式 | 峰值 CUDA 显存 | 平均耗时 |
| --- | ---: | ---: |
| 整图推理 | 4203.5 MiB | 0.302 s/张 |
| 512/192 Overlap-Crop | 390.6 MiB | 0.839 s/张 |

Crop–merge 带来的变化为：

- 峰值显存降低约 90.7%；
- 峰值显存约为整图推理的 1/10.8；
- 平均耗时增加至约 2.77 倍；
- 内部区域理论重复计算量约为：

```text
(tile_size / core_size)^2
= (512 / 320)^2
= 2.56
```

时间增加主要来自重叠区域的重复卷积和逐 patch 调度。硬件实现时可以通过
DMA 与计算流水重叠、双缓冲、片上缓存复用等方式降低调度开销。

## 8. 软件实现

核心模块：

```text
led/utils/tiled_inference.py
```

支持：

- `crop`：带 halo 的可靠中心裁剪拼接；
- `hann`：Hann 加权 overlap-add；
- `uniform`：重叠区域平均；
- 非 16 倍数输入的全局对称 padding；
- CPU 保存完整输入和输出，仅将单个 patch 流式送入 GPU；
- CUDA AMP；
- 任意保持输入输出空间尺寸不变的图像恢复网络。

SID 评测脚本：

```text
scripts/evaluate_crop_merge_sid.py
```

评测脚本会输出：

```text
results/crop_merge_sid/details.csv
results/crop_merge_sid/summary.csv
results/crop_merge_sid/report.json
```

其中 `details.csv` 保存逐图结果，`summary.csv` 保存各倍率及总体平均结果。

## 9. 使用方法

### 9.1 运行完整 SID 对比

```bash
conda activate LED-ICCV23

python scripts/evaluate_crop_merge_sid.py \
  --dataset-root /path/to/SID/Sony_npy \
  --tile-size 512 \
  --overlaps 192 \
  --methods crop \
  --device cuda:0
```

如果数据位于本次实验使用的共享目录，可以执行：

```bash
python scripts/evaluate_crop_merge_sid.py \
  --dataset-root /home/shared_files/dataset/SID/Sony_npy \
  --tile-size 512 \
  --overlaps 192 \
  --methods crop \
  --device cuda:0
```

### 9.2 扫描多种重叠策略

```bash
python scripts/evaluate_crop_merge_sid.py \
  --dataset-root /path/to/SID/Sony_npy \
  --max-images-per-ratio 5 \
  --tile-size 512 \
  --overlaps 128 192 256 \
  --methods crop hann uniform
```

### 9.3 在其他代码中调用

```python
from led.utils.tiled_inference import tiled_inference

output = tiled_inference(
    model.eval(),
    lq,                         # BCHW packed RAW，可保留在 CPU
    tile_size=512,
    overlap=192,
    blend='crop',
    alignment=16,
    inference_device='cuda:0',
    accumulation_device='cpu',
)
```

完整输出在 CPU 中合并，GPU 同一时刻只需保存模型和一个 patch 的中间特征，
符合硬件流式处理的基本思路。

## 10. 硬件部署建议

可以将软件循环映射为如下硬件流程：

```text
外部存储读取 patch
        ↓
输入双缓冲
        ↓
LED U-Net 流式计算
        ↓
丢弃内部边缘 halo
        ↓
将 320×320 core 写回目标地址
        ↓
读取下一个 patch
```

需要重点注意：

1. patch 起点必须保持 16 对齐；
2. 内部 patch 需要上下左右 96 像素 halo；
3. 图像真实外边界不能像内部边界一样直接裁掉；
4. illumination correction 应在全图完成后统一计算；
5. 如果硬件保存不了 512×512 patch，可以降低 `tile_size`，但应继续保留
   192 overlap，否则 PSNR 会下降；
6. tile 变小会显著增加重复计算，例如 `tile_size=384、overlap=192` 时，
   core 只有 192×192；
7. 换用 NAFNet、Restormer 或其他网络时，必须重新分析感受野和下采样倍数，
   不能直接复用当前的 96 像素 halo。

## 11. 最终结论

`512×512 patch + 192 overlap + 96 halo + overlap-crop + 16 像素对齐`
是当前 LED U-Net 权重下兼顾精度与显存的推荐方案。

它将峰值 GPU 显存从约 4.20 GiB 降低到约 0.39 GiB，同时在 SID 93 张验证图
上的平均 PSNR 损失仅为 `1.26e-8 dB`，基本保持整图推理精度。其代价是约
2.77 倍运行时间和约 2.56 倍理论重复计算量，适合作为资源受限硬件上的
流式去噪实现基础。
