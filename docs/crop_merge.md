# LED 大图 crop–merge 推理与 SID 评测

## 结论

对当前仓库的 RepNR U-Net 和
`experiments/LED_Pretrain_CVPR20_ELD_Setting/models/net_g_latest.pth`，推荐参数是：

```text
tile_size = 512
overlap   = 192
blend     = crop
alignment = 16
```

这里的 `crop` 不是直接无重叠切块，而是 overlap-crop（halo exchange）：
相邻输入 patch 重叠 192 个 packed-RAW 像素，每个内部边缘保留 96 像素
上下文；网络输出后丢弃可能受 patch 边界影响的 96 像素 halo，只拼接可靠的
320×320 中心区域。

在 SID Sony 的 93 张验证图上，该方式与整图输出只有浮点计算次序导致的误差：

| SID 倍率 | 图像数 | 整图 PSNR | crop–merge PSNR | PSNR 损失 |
| --- | ---: | ---: | ---: | ---: |
| 100× | 29 | 42.008605 dB | 42.008605 dB | 1.62e-8 dB |
| 250× | 29 | 39.242763 dB | 39.242763 dB | 1.39e-8 dB |
| 300× | 35 | 36.696812 dB | 36.696812 dB | 8.66e-9 dB |
| 总体 | 93 | 39.147076 dB | 39.147076 dB | 1.26e-8 dB |

总体输出相对整图输出的 PSNR 是 168.62 dB，MAE 是 9.12e-10，最大绝对误差
是 4.77e-7。PSNR 在 linear packed RAW 上计算，使用与 LED 验证配置一致的
全图 illumination correction 和 2 像素 metric crop。

RTX 2080 Ti 上的平均统计如下：

| 方式 | 峰值 CUDA 显存 | 每张耗时 |
| --- | ---: | ---: |
| 整图 | 4203.5 MiB | 0.302 s |
| 512/192 overlap-crop | 390.6 MiB | 0.839 s |

分块方式将峰值显存降低约 90.7%（约 10.8 倍），代价是约 2.77 倍耗时。
内区的理论重复计算量约为 `(512 / 320)^2 = 2.56` 倍。

完整逐图结果在 `results/crop_merge_sid/details.csv`，汇总结果在
`results/crop_merge_sid/summary.csv` 和 `report.json`。

## 为什么选择 overlap-crop

U-Net 每个输出像素依赖周围一片输入。若直接切块，内部 patch 边缘会被网络
当成图像边缘，卷积 padding 产生的错误会向内传播，形成网格接缝。

本实现采用以下措施：

1. **输入重叠，输出裁边。** 内部 patch 的边缘输出不参与合并，因此不会把
   伪边界带到最终图像。整幅图的最外边缘仍保留原网络的边界行为。
2. **96 像素 halo。** 当前四层 U-Net 的有效感受野分析及 SID 实测表明，
   在 stride-16 对齐的 core 边界上，96 像素足以覆盖被保留输出的上下文。
3. **所有坐标按 16 对齐。** 网络有四次 2× 下采样。若 patch 起点不与全图
   的 pooling phase 对齐，即使 overlap 足够也可能与整图结果不同。
4. **先全局对称 padding。** 非 16 倍数的输入先按 U-Net 相同的 reflect
   规则对称补齐，合并后再裁回原尺寸，末尾 patch 不会破坏相位。
5. **全图合并后再做 illumination correction。** 若每个 patch 单独校正，
   每块会得到不同亮度系数，接缝和 PSNR 偏差都会增大。

实现同时提供完整 patch 输出的 Hann 加权融合和平均融合，便于其他网络使用。
但对当前 U-Net，裁掉不可靠边缘比“平滑错误边缘”更准确。15 张 SID 策略扫描：

| 512 patch 策略 | 平均 PSNR 损失 | 输出相对整图 PSNR |
| --- | ---: | ---: |
| crop, overlap 128（halo 64） | 0.019285 dB | 68.34 dB |
| Hann, overlap 128 | 0.016514 dB | 70.02 dB |
| crop, overlap 192（halo 96） | 2.60e-8 dB | 168.63 dB |
| Hann, overlap 192 | 0.003307 dB | 75.77 dB |

当 overlap 不足时，Hann 通常比硬平均平滑，且在这组扫描中也比 128-overlap
的 crop 更接近整图；但只要 halo 覆盖感受野，overlap-crop 可以直接消除内部
patch 边界误差，所以最终选择 192-overlap crop。

## 使用

先激活项目环境，然后运行完整 SID 对比：

```bash
conda activate LED-ICCV23
python scripts/evaluate_crop_merge_sid.py \
  --dataset-root /path/to/SID/Sony_npy \
  --tile-size 512 \
  --overlaps 192 \
  --methods crop \
  --device cuda:0
```

权重默认就是本次指定的 `net_g_latest.pth`。脚本会按照原验证逻辑，先平均
RepNR 的五个虚拟相机 alignment 分支，再融合成普通 U-Net 做推理。

快速扫描多种策略：

```bash
python scripts/evaluate_crop_merge_sid.py \
  --dataset-root /path/to/SID/Sony \
  --input-format raw \
  --max-images-per-ratio 5 \
  --tile-size 512 \
  --scan-overlap-methods \
  --device cuda:0
```

该预设会扫描 `overlap={128,160,192}` 与
`method={crop,hann,uniform}` 的全部 9 种组合。其他组合仍可通过
`--overlaps ... --methods ...` 显式指定。

在其他推理代码中直接调用：

```python
from led.utils.tiled_inference import tiled_inference

# lq 保持在 CPU；每次只把一个 patch 流式送入 GPU，完整输出也在 CPU 合并。
output = tiled_inference(
    model.eval(),
    lq,                         # BCHW packed RAW
    tile_size=512,
    overlap=192,
    blend='crop',
    alignment=16,
    inference_device='cuda:0',
    accumulation_device='cpu',
)
```

如果硬件仍放不下 512 patch，可以在保持 `overlap=192` 的前提下降低
`tile_size`，但 core 会变小、重复计算比例会明显增加。换用其他网络时，
应根据它的下采样倍数修改 `alignment`，并重新扫描 overlap；不要直接假设
96 像素 halo 对所有架构都足够。

## 直接读取原始 SID Short/Long RAW

评测脚本可绕过 NPZ 缓存，直接解码验证列表中的原始 ARW 对：

```bash
python scripts/evaluate_crop_merge_sid.py \
  --dataset-root /path/to/SID/Sony \
  --input-format raw \
  --tile-sizes 512 768 1024 \
  --overlaps 192 \
  --methods crop \
  --device cuda:0 \
  --output-dir results/crop_merge_sid_raw_full
```

SID Sony A7S2 的网络输入是 `4×1424×2128` packed RAW，对应原始 Bayer
尺寸 `2848×4256`。原先使用的 `Sony_npy` 是同一原始 RAW 的无损整图缓存，
不是 512×512 预裁剪数据；512 是推理 tile 大小。对于一张完整 SID 图：

| 配置 | Patch 数 | 实际计算倍率 | 估算 GMACs | 平均推理时间 | 峰值显存 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 整图 | 1 | 1.000× | 633.79 | 0.320 s | 4203.5 MiB |
| Crop 512/192 | 35 | 2.373× | 1503.75 | 0.723 s | 390.6 MiB |
| Crop 768/192 | 12 | 1.613× | 1022.51 | 0.491 s | 842.4 MiB |
| Crop 1024/192 | 6 | 1.340× | 849.03 | 0.404 s | 1338.2 MiB |

93 对原始 Short/Long ARW 的 noisy-input PSNR 为 27.2601 dB，模型整图输出为
39.1471 dB，平均降噪增益为 11.8870 dB。三种 crop 配置都保持 96 像素
halo，输出 PSNR 均为 39.1471 dB，与整图差异处于浮点误差量级。

脚本输出中还包含逐图的 RAW 解码时间、端到端时间、patch 尺寸分布、实际处理
像素数、THOP GMAC/GFLOPs、吞吐率、显存降幅以及分块输出相对整图的误差。
