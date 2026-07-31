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
experiments/LED_Pretrain_CVPR20_ELD_Setting_archived_20260726_230625/models/net_g_latest.pth
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
| 整图推理 | 4203.5 MiB | 0.320 s/张 |
| 512/192 Overlap-Crop | 390.6 MiB | 0.723 s/张 |

Crop–merge 带来的变化为：

- 峰值显存降低约 90.7%；
- 峰值显存约为整图推理的 1/10.8；
- 平均纯推理耗时增加至约 2.26 倍；
- 考虑边缘 patch 的实际重复计算量为 2.373 倍；
- 无限大图内部区域的渐近重复计算量约为：

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
results/crop_merge_sid_raw_full/details.csv
results/crop_merge_sid_raw_full/summary.csv
results/crop_merge_sid_raw_full/report.json
```

其中 `details.csv` 保存逐图结果，`summary.csv` 保存各倍率及总体平均结果。

## 9. 使用方法

### 9.1 运行完整 SID 对比

```bash
conda activate LED-ICCV23

python scripts/evaluate_crop_merge_sid.py \
  --dataset-root /path/to/SID/Sony \
  --input-format raw \
  --tile-sizes 512 768 1024 \
  --overlaps 192 \
  --methods crop \
  --device cuda:0
```

如果数据位于本次实验使用的共享目录，可以执行：

```bash
python scripts/evaluate_crop_merge_sid.py \
  --dataset-root /home/shared_files/dataset/SID/Sony \
  --input-format raw \
  --tile-sizes 512 768 1024 \
  --overlaps 192 \
  --methods crop \
  --device cuda:0
```

### 9.2 扫描多种重叠策略

```bash
python scripts/evaluate_crop_merge_sid.py \
  --dataset-root /path/to/SID/Sony \
  --input-format raw \
  --max-images-per-ratio 5 \
  --tile-size 512 \
  --scan-overlap-methods \
  --device cuda:0 \
  --output-dir results/crop_merge_sid_raw_method_overlap_15
```

`--scan-overlap-methods` 会忽略手工传入的 `--overlaps` 和 `--methods`，固定执行
如下 9 种组合：

```text
overlap ∈ {128, 160, 192}
method  ∈ {crop, hann, uniform}
```

如果需要其他组合，仍可显式使用 `--overlaps ... --methods ...`。例如
`--overlaps 96 224 --methods crop hann` 会执行 4 组配置。

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
2.26 倍纯推理时间和 2.373 倍实际计算量，适合作为资源受限硬件上的
流式去噪实现基础。

## 12. 可能的硬件数据流架构与 3×3 策略扫描

### 12.1 公共前端与约束

三种 merge 方法可以共用 RAW 读取、tile 调度和 LED U-Net 计算单元：

```text
Short RAW/DDR
      ↓
packed RAW 解包与归一化
      ↓
16 像素对齐的 tile 地址发生器
      ↓
DMA ping-pong 输入缓冲
      ↓
LED U-Net
      ↓
crop 直接写回，或 Hann/uniform overlap-add
      ↓
4 通道 packed RAW 输出
```

关键约束如下：

1. `tile_size`、`overlap`、patch 起点和 patch 边长都必须满足网络的 16 像素
   下采样对齐要求；
2. 片上只需常驻模型工作集和当前/下一 patch，输入可通过双缓冲与计算重叠；
3. 当前软件的 crop 边界 patch 可以小于 512，并按实际尺寸计算。因此下表的
   crop 计算量是“支持可变边长”的结果。如果硬件只能运行固定 512×512，
   应对边界做 padding 和 valid mask；此时 35 个 patch 全部按 512×512
   计算，实际计算倍率会高于软件统计；
4. 本文 PSNR 评测使用的 illumination correction 依赖 Long RAW 真值，
   它只是评测协议，不属于真实相机部署链路。用于复现实验时必须在完整输出
   merge 后统一执行，不能逐 patch 执行。

### 12.2 Crop：无跨 tile 累加状态的流式路径

Crop 方法把 `overlap/2` 作为 halo，只保留网络输出的可靠 core。不同 tile
的 output box 互不重叠，所以每个最终像素只写一次：

```text
for tile in tile_schedule:
    DMA-in  input patch（core + 内部 halo）
    patch_output = LED_U-Net(input patch)
    valid_core = discard_internal_halo(patch_output)
    DMA-out valid_core → final_output[output_box]   # 直接写，无读改写
```

真实图像外边界必须保留整图网络原有的 padding 行为，只裁掉人为产生的内部
tile 边界。以 `tile_size=512` 为例：

| Overlap | 单侧 Halo | 标称 Core | Patch 数 | 软件实际计算倍率 |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 64 | 384×384 | 24 | 1.652× |
| 160 | 80 | 352×352 | 35 | 1.996× |
| 192 | 96 | 320×320 | 35 | 2.373× |

这里 `overlap=160` 虽然单块重复区域小于 192，但 `1424×2128` 不能被
352×352 core 整齐覆盖，因此会生成很窄的末尾边界块，patch 数仍达到 35。
硬件调度器应显式携带 `valid_height/valid_width`，避免把 padding 区域当作
有效输出。

### 12.3 Hann/Uniform：带全图累加状态的路径

Hann 和 Uniform 均保留完整 patch 输出，重叠区域必须归一化累加：

```text
DDR 常驻:
    output_sum[4, H, W]
    weight_sum[1, H, W]

for tile in tile_schedule:
    DMA-in  input patch
    patch_output = LED_U-Net(input patch)
    DMA-in  当前区域的 output_sum 和 weight_sum
    output_sum += patch_output * weight
    weight_sum += weight
    DMA-out 更新后的 output_sum 和 weight_sum

final_output = output_sum / weight_sum
```

Uniform 的 `weight=1`，省去窗函数乘法，但仍有全图读—改—写和最终除法；
Hann 使用可分离二维窗 `Wy × Wx`，系数可由 ROM/LUT 或一维发生器提供，
接缝通常比硬平均平滑，但需要额外乘法。若不能在片上保存覆盖一条完整
overlap stripe 的累加值，两者都需要频繁访问外存，严格意义上不再是
“tile 完成后即可永久释放状态”的真流式结构。

对 `4×1424×2128` 的 float32 packed RAW：

- 一张 4 通道输出图为 46.24 MiB；
- 一张 1 通道 `weight_sum` 为 11.56 MiB；
- Hann/Uniform 至少需要 57.80 MiB 的 `output_sum + weight_sum` 累加空间，
  还未计输入、最终输出和重复的外存读写；
- Crop 只需要最终输出地址，不需要 `weight_sum`，且 output box 可直接写入。

三种方法的 CUDA 峰值都约为 390.6 MiB，是因为 GPU 同时只计算一个最大
512×512 patch；该数字没有反映 Hann/Uniform 放在 CPU/DDR 的全图累加
buffer 和额外带宽。因此硬件选择不能只比较 CUDA 峰值。

### 12.4 Overlap × Method 原始大图扫描

测试直接读取 SID 原始 Short/Long ARW，每个倍率取 5 对，共 15 对，
网络输入均为 `4×1424×2128`。计算量按实际处理的 packed 像素和
`209152 MAC/pixel` 估算；时间是 RTX 2080 Ti 上的纯推理平均时间。

| Method | Overlap | Patch 数 | 计算倍率 | 估算 GMAC | 时间 | 相对整图目标 PSNR 变化 | 输出相对整图 PSNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| crop | 128 | 24 | 1.652× | 1046.71 | 0.515 s | -0.019285 dB | 68.34 dB |
| crop | 160 | 35 | 1.996× | 1264.95 | 0.634 s | +0.007994 dB | 72.07 dB |
| crop | 192 | 35 | 2.373× | 1503.75 | 0.709 s | -2.60e-8 dB | 168.63 dB |
| Hann | 128 | 24 | 2.076× | 1315.87 | 0.647 s | -0.016514 dB | 70.02 dB |
| Hann | 160 | 24 | 2.076× | 1315.87 | 0.642 s | -0.011215 dB | 73.55 dB |
| Hann | 192 | 28 | 2.422× | 1535.18 | 0.749 s | -0.003307 dB | 75.77 dB |
| Uniform | 128 | 24 | 2.076× | 1315.87 | 0.626 s | -0.053792 dB | 60.33 dB |
| Uniform | 160 | 24 | 2.076× | 1315.87 | 0.632 s | -0.060553 dB | 60.26 dB |
| Uniform | 192 | 28 | 2.422× | 1535.18 | 0.736 s | -0.043728 dB | 60.33 dB |

需要区分两个指标：

- “相对整图目标 PSNR 变化”比较 tiled/full 各自对 Long RAW 真值的 PSNR；
  `crop/160` 的 `+0.007994 dB` 是 tile 误差与模型误差偶然抵消，不能证明
  它比整图推理更忠实；
- “输出相对整图 PSNR”直接比较 tiled 输出与 full 输出，是衡量 crop–merge
  数值保真度的主要指标。`crop/192` 达到 168.63 dB，平均绝对误差仅
  `9.27e-10`，明显优于其他 8 组。

Hann/Uniform 在 overlap=128 与 160 时都是 24 个 512×512 patch、2.076×
计算量，这是由滑窗起点和末窗锚定到图像末端的离散几何决定的；二者虽然
patch 数相同，实际相邻窗口的重叠宽度仍不同。overlap 增至 192 后才增加到
28 个 patch。

### 12.5 硬件方案选择

综合数值精度、计算量和数据流复杂度：

1. **最终推荐：`crop/192`。** 它是唯一与整图输出达到浮点误差量级一致的
   配置，且不需要跨 tile 累加 buffer，适合作为硬件基准实现；
2. **性能折中：`crop/160`。** 相比 crop/192，实际计算量减少约 15.9%，
   纯推理时间减少约 10.6%，但输出相对整图 PSNR 降至 72.07 dB；只有系统
   能接受该数值偏差时才建议采用；
3. **Hann 是泛化备选。** 当其他网络无法准确确定可裁弃的 halo 时，Hann
   可以平滑重叠边界，但当前 LED 上精度、计算量和外存流量均不如 crop/192；
4. **Uniform 不推荐用于当前权重。** 它的算术最简单，但直接平均了受
   padding 影响的 patch 边缘，本次 3×3 扫描中三组误差都最大。

本次扫描结果保存在：

```text
results/crop_merge_sid_raw_method_overlap_15/summary.csv
results/crop_merge_sid_raw_method_overlap_15/details.csv
results/crop_merge_sid_raw_method_overlap_15/report.json
```

## 13. 原始 Short/Long 大图验证

### 13.1 数据源澄清

本轮进一步绕过 NPZ 缓存，直接使用以下 SID 原始文件对：

```text
/home/shared_files/dataset/SID/Sony/short/*.ARW
/home/shared_files/dataset/SID/Sony/long/*.ARW
```

验证列表仍使用：

```text
datasets/txtfiles/SID/SonyA7S2/valid_100.txt
datasets/txtfiles/SID/SonyA7S2/valid_250.txt
datasets/txtfiles/SID/SonyA7S2/valid_300.txt
```

需要特别说明：之前评测使用的 `Sony_npy` 也不是 512×512 的预裁剪 patch，
而是原始 ARW 的无损整图缓存。抽查同一组 Short/Long 数据时，原始 ARW 与
NPZ 的 packed RAW 尺寸、归一化像素值、白平衡和 CCM 完全一致。

每张原始 Sony A7S2 图像的尺寸为：

```text
传感器 Bayer RAW：2848×4256
网络 packed RAW：4×1424×2128
```

其中 512×512 是 crop–merge 的网络输入 tile，而不是待测源图的尺寸。
`512/192 crop` 在一张原始大图上实际产生 35 个 patch，因此确实执行了多块
流式推理。

### 13.2 原始 RAW 降噪性能

本轮在 93 对原始 Short/Long ARW 上重新完成完整评测。Noisy input 和模型
输出均采用相同的全图 illumination correction 与 2 像素 metric crop：

| SID 倍率 | 图像数 | Noisy Input PSNR | 整图输出 PSNR | 降噪增益 | 512/192 Crop PSNR |
| --- | ---: | ---: | ---: | ---: | ---: |
| 100× | 29 | 30.201851 dB | 42.008605 dB | 11.806754 dB | 42.008605 dB |
| 250× | 29 | 26.613879 dB | 39.242763 dB | 12.628885 dB | 39.242763 dB |
| 300× | 35 | 25.357967 dB | 36.696812 dB | 11.338845 dB | 36.696812 dB |
| 总体 | 93 | 27.260054 dB | 39.147076 dB | 11.887023 dB | 39.147076 dB |

从原始 Short/Long 数据对可见，模型总体带来约 11.89 dB 的 PSNR 提升；
crop–merge 没有损失这一降噪增益。

### 13.3 Tile 大小、块数和资源代价

固定 `overlap=192、halo=96、alignment=16`，比较三种都确实发生分块的
tile 大小。MACs 由 THOP 对部署态 U-Net 统计后，按各 patch 的真实空间尺寸
累加；1 MAC 按 2 FLOPs 计算。

| 方案 | 块数 | 实际计算倍率 | 估算 GMACs | 纯推理时间 | 含 RAW 解码时间 | 峰值显存 | 显存降低 | PSNR 损失 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 整图 | 1 | 1.000× | 633.79 | 0.320 s | 0.576 s | 4203.5 MiB | — | — |
| Crop 512/192 | 35 | 2.373× | 1503.75 | 0.723 s | 0.978 s | 390.6 MiB | 90.71% | 1.26e-8 dB |
| Crop 768/192 | 12 | 1.613× | 1022.51 | 0.491 s | 0.746 s | 842.4 MiB | 79.96% | -1.11e-10 dB |
| Crop 1024/192 | 6 | 1.340× | 849.03 | 0.404 s | 0.659 s | 1338.2 MiB | 68.16% | -1.00e-8 dB |

负的 PSNR 损失表示极小的正向浮点波动，不代表实际质量提升。三种 crop 配置
相对整图输出的 PSNR 分别为 168.62、175.99 和 166.76 dB，均属于浮点误差
量级。

原始 Short/Long ARW 的平均解码与归一化耗时约为 0.255 s/对。纯推理时间
不包含 RAW 解码，端到端时间则包含同一对 Short/Long 文件的加载和处理。

这一结果揭示了可调的硬件权衡：

- 512 tile 显存最低，适合片上 SRAM/显存严格受限的设备，但需处理 35 块；
- 768 tile 是较均衡的方案，仍降低约 80% 显存，计算量为整图的 1.61 倍；
- 1024 tile 只需 6 块，速度接近整图，但峰值显存提高到约 1.31 GiB；
- 三者保持相同的 96 像素 halo，因此降噪精度都与整图一致。

### 13.4 Overlap 有效性

为验证重叠区域确实必要，在原始 ARW 的 15 对样本上固定 tile=512，扫描
不同 overlap：

| Overlap | Halo | 块数 | 实际计算倍率 | 估算 GMACs | 平均 PSNR 损失 | 输出相对整图 PSNR |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 15 | 1.000× | 633.79 | 0.084512 dB | 56.35 dB |
| 64 | 32 | 20 | 1.271× | 805.77 | -0.019796 dB | 59.75 dB |
| 128 | 64 | 24 | 1.652× | 1046.71 | 0.019285 dB | 68.34 dB |
| 192 | 96 | 35 | 2.373× | 1503.75 | 2.60e-8 dB | 168.63 dB |

Overlap=64 偶尔会因边界误差碰巧提高相对 GT 的平均 PSNR，但其输出相对整图
只有 59.75 dB，说明它改变了网络原本的计算结果，不能视为稳定的降噪改进。
判断 crop–merge 是否正确，除了比较 GT PSNR，还必须同时检查分块输出相对
整图输出的 PSNR、MAE 和最大误差。

无重叠时的平均损失为 0.0845 dB；随着 halo 增大，输出逐渐接近整图；
只有 96 像素 halo 将误差稳定降低到浮点量级。这从原始大图上验证了
overlap-crop 的必要性和有效性。

### 13.5 原始 RAW 复现命令

```bash
conda activate LED-ICCV23

python scripts/evaluate_crop_merge_sid.py \
  --dataset-root /home/shared_files/dataset/SID/Sony \
  --input-format raw \
  --tile-sizes 512 768 1024 \
  --overlaps 192 \
  --methods crop \
  --device cuda:0 \
  --output-dir results/crop_merge_sid_raw_full
```

完整实验报告位于：

```text
results/crop_merge_sid_raw_full/details.csv
results/crop_merge_sid_raw_full/summary.csv
results/crop_merge_sid_raw_full/report.json
```

## 14. GPU Batch 并行 Tile 模拟

### 14.1 为什么需要先裁成规则 ROI

第 13 节的流式 crop 实现为了保留完整原图，会在图像边界产生不同尺寸的
patch，无法直接沿 batch 维堆叠。本轮改用三种方法完全相同的规则滑窗：

```text
tile_size = 512
overlap   = 192
stride    = 512 - 192 = 320
```

要让每个 patch 都严格为 512×512，裁剪后的每条边必须满足：

```text
length = 512 + n × 320
```

原始 packed RAW 为 `4×1424×2128`，保持网络 16 像素下采样相位后，取
如下 ROI：

```text
高度：1424 → 1152，裁掉 top=128、bottom=144
宽度：2128 → 2112，裁掉 left=0、right=16
网格：3 行 × 6 列 = 18 个 512×512 tile
保留原 packed RAW 像素：80.29%
```

三种 merge 方法由此共享相同的 18 个网络输入、相同的 986.90 GMAC 和
1.939× 重复计算量，避免了原来 crop 边界块较小而 Hann/Uniform 始终计算
完整块造成的不公平比较。需要注意，本节 PSNR 都在裁剪后的 Short/Long ROI
上计算，只用于研究 GPU batch 并行与 merge，不应与第 13 节的完整 93 张
全图指标直接混合。

### 14.2 GPU 数据流

实现位于 `batched_tiled_inference()`，其数据流为：

```text
规则 ROI
  ↓
提取 18 个 512×512 tile
  ↓
按 tile_batch_size 堆成 [Btile, 4, 512, 512]
  ↓
一次或多次 LED U-Net 前向
  ↓
输出保留在 GPU
  ↓
GPU crop 写回，或 GPU Hann/Uniform overlap-add
  ↓
完整 4×1152×2112 GPU 输出
```

`tile_batch_size=0` 表示把全部 18 个 tile 一次送入网络；正数则分批执行。
这能模拟多个 tile 同时进入并行计算阵列，但 GPU 的 cuDNN batch 卷积并不
等价于真实硬件上复制 18 套独立 U-Net 流水线，因此结果主要用于估算并行度、
显存和吞吐变化。

### 14.3 Batch 大小扫描

在 RTX 2080 Ti 上直接读取 SID 原始 Short/Long ARW，每个倍率取 5 对，
共 15 对。这里“一张图的时间”已经包含该图 18 个 tile 的提取、全部网络
前向和 GPU merge，并不是单个 tile 的耗时；“15 对 GPU 总时间”是 15 张
图的上述完整处理时间求和。下表使用数值最准确的 crop 方法展示结果：

| Tile Batch | 每图网络启动次数 | 15 对 GPU 总时间 | 平均每图完整时间 | 15 对端到端总时间 | 相对 Batch=1 加速 | 峰值显存 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 18 | 3.2320 s | 0.2155 s | 6.9244 s | 1.000× | 431.7 MiB |
| 2 | 9 | 3.2168 s | 0.2145 s | 6.9092 s | 1.005× | 797.6 MiB |
| 4 | 5 | 3.2251 s | 0.2150 s | 6.9175 s | 1.002× | 1526.7 MiB |
| 8 | 3 | 3.2674 s | 0.2178 s | 6.9598 s | 0.989× | 2986.7 MiB |
| 18 | 1 | 3.5122 s | 0.2341 s | 7.2046 s | 0.920× | 6529.6 MiB |

端到端总时间按“每张 RAW 读取与预处理时间 + 该配置 GPU 推理时间”累加，
不包含 PSNR 计算和 CSV 写入。15 对图共用的 RAW 读取与预处理总时间为
3.6924 s。同一 ROI 的整图推理在 15 对图上的 GPU 总时间为 1.8119 s，
平均每图 0.1208 s，含 RAW 读取的端到端总时间为 5.5043 s，峰值显存为
3388.9 MiB。Batch=1
分块推理仍需做 1.939× 计算，所以耗时为整图的约 1.79 倍，但显存仅约
0.42 GiB。Batch=18 虽然把网络启动次数降到 1，却比 Batch=1 慢约 8.0%，
峰值显存增至约 6.38 GiB，甚至超过裁剪 ROI 的整图推理。

这说明 512×512 单 tile 已能较充分占用当前 GPU，卷积计算而不是 Python
循环或 kernel launch 是主要开销。增加 batch 只改变 cuDNN 的并行映射并
保存更多中间激活，没有减少总 MAC。当前 GPU 的实际折中点为 Batch=2 或 4；
若目标只是最大程度模拟“一次并行处理全部 tile”，可用 Batch=18，但不应
预期它自动加速。

### 14.4 三种 Merge 方法

下表固定 `tile_batch_size=18`，即每张图的 18 个 tile 一次前向，并在 GPU
上拼接：

| Method | ROI 输出 PSNR | 相对 ROI 整图 PSNR 变化 | 输出相对整图 PSNR | MAE vs Full | 15 对 GPU 总时间 | 15 对端到端总时间 | 峰值显存 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Crop | 39.649942 dB | +1.10e-7 dB | 156.38 dB | 9.83e-9 | 3.5122 s | 7.2046 s | 6529.6 MiB |
| Hann | 39.647679 dB | -0.002264 dB | 75.82 dB | 7.79e-5 | 3.5470 s | 7.2394 s | 6529.6 MiB |
| Uniform | 39.605360 dB | -0.044582 dB | 60.73 dB | 4.13e-4 | 3.5377 s | 7.2301 s | 6529.6 MiB |

结论与流式版本一致：

1. Crop/192 的 96 像素 halo 已覆盖当前 LED U-Net 的有效边缘影响。
   Batch=1/2/4/8 时输出与 ROI 整图逐像素一致；Batch=18 因 cuDNN 对不同
   batch 形状采用的浮点计算次序不同，仍只产生 `9.83e-9` 的平均误差；
2. Hann 平滑了边界输出，但没有删除受 padding 影响的像素，因此相对整图
   PSNR 只有 75.82 dB；
3. Uniform 直接平均可靠中心与不可靠边缘，误差最大，平均目标 PSNR 下降
   0.0446 dB；
4. Batch=18 时网络激活占主导，三种方法的 CUDA 峰值相同。Batch=1 时
   Crop 为 431.7 MiB，Hann/Uniform 约 442.9 MiB，后两者额外保存 GPU
   `weight_sum`。

因此，若目标是 GPU 上模拟硬件并行，建议：

```text
精度优先：crop + overlap 192
GPU 性能/显存折中：tile_batch_size 2 或 4
最大并行形态模拟：tile_batch_size 18，但显存约 6.53 GiB 且不会加速
```

### 14.5 复现命令

```bash
conda activate LED-ICCV23

python scripts/evaluate_gpu_batched_crop_merge_sid.py \
  --dataset-root /home/shared_files/dataset/SID/Sony \
  --input-format raw \
  --ratios 100 250 300 \
  --max-images-per-ratio 5 \
  --tile-size 512 \
  --overlap 192 \
  --methods crop hann uniform \
  --tile-batch-sizes 1 2 4 8 0 \
  --device cuda:0 \
  --output-dir results/crop_merge_sid_gpu_batch_15
```

其中最后一个 batch 参数 `0` 会自动解析为当前图像的全部 18 个 tile。结果
保存在：

```text
results/crop_merge_sid_gpu_batch_15/details.csv
results/crop_merge_sid_gpu_batch_15/summary.csv
results/crop_merge_sid_gpu_batch_15/report.json
```
