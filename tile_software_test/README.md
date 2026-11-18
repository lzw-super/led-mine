# LLStemNoAct：FPGA 分块推理实验设计

分析日期：2026-11-18。代码基线：`b7cf3c434f53107e8c0713a0477359c31b33b49d`。
目标实验：`experiments/LED_Pretrain_CVPR20_ELD_LearningDWT_Haar_L3_RepNCB_W32_ReLU_LLStemNoAct`。
本目录包含分析、统计、验证和软件实测材料，不修改训练或正式推理流程。

量化模型手动测试：[QAT crop 测试命令与口径](QAT_CROP_TEST.md)（原三组及追加七组，命令均由用户手动执行）。

最新联合结果：[13 组 FP32 crop 精度与时间对比](FP32_CROP_COMPARISON_13.md)，包含新增 tile=88/96、单边 halo=20/24。

原九组实测入口：[FP32 小 tile crop 精度与时间结果](FP32_CROP_RESULTS.md)；tile=80/88/96、单边 halo=8/10/16，共 9 组。

## 1. 结论与尺寸约定

- 所有 tile / halo 尺寸默认指网络输入的 **packed RAW 像素**，张量为 `[N,4,H,W]`，不是 RGB 或 Bayer 单通道像素。Bayer 尺寸为 `2H × 2W`。
- 定义输入 tile 最大边长 `T`、单侧 halo `h`、保留的输出 core 边长 `C=T-2h`。相邻内部输入 tile 重叠 `2h`，步长为 `C`。
- 对当前固定 Haar L3 网络，起点、core 和 halo 均按 **8 packed 像素**对齐时，架构保证的最小单侧 halo 为 **56**，对应 Bayer 112；重叠为 112 packed。
- 建议初始基线 **T=512,h=64,C=384**；资源更紧时选 `256/56/144` 或 `256/64/128`；精确边界最小方案为 `512/56/400`。尚无指定 FPGA 型号、片上 RAM、数据位宽、DDR 带宽，因此这些是测量候选，不能视为已完成资源收敛。
- 拼接采用 **overlap-crop：丢弃内部 halo，只写 core**。每个输出像素写一次，无需权重图、浮点累加或融合除法。

## 2. 数据统计口径

见 [dataset_summary.md](dataset_summary.md)、[analysis.json](analysis.json) 和逐文件 [image_inventory.csv](image_inventory.csv)。统计扫描当前数据链接内全部 NPZ 的 `im.npy` 头部，不解压整幅数组；验证列表逐对检查输入/GT 是否存在、尺寸是否相同。

当前实验训练输入来自 `Sony_train_long_patches`，验证来自 `Sony_npy` 中 `valid_100/250/300.txt` 指定的输入。额外统计 `Sony_npy` 全库及 train 配对列表用于区分“本实验实际使用”和“磁盘可用数据”。训练列表不是当前 RAWGT 训练实际使用的整图输入。Sony 原始 ARW 与 NPZ 是同源存储，不重复算作两套输入。Bayer 分辨率由 packed shape 推导，未重新解析 ARW 传感器完整含黑边尺寸；也不泛化到未配置的 ELD/Nikon 等数据集。

配置依据：实验 YAML；`options/base/dataset/pretrain/SID_raw_gt.yaml`、`options/base/dataset/test/SID_SonyA7S2_val_split.yaml`。两者 `crop_size: null`，网络直接接收存储尺寸。`model_complexity.input_shape=512×512` 仅为复杂度估计输入，不是验证图像分辨率。

## 3. 按实际计算图推导 halo

代码依据（相对仓库根目录）：

| 文件/模块 | 当前有效路径 |
|---|---|
| `led/archs/sid_reference_arch.py` | 输入输出通道置换，不改变空间支持 |
| `led/archs/sid_reference/learning_wt/learning_dwt.py` | `_static_subband_forward`，固定 Haar 三级分解与重建 |
| 同文件 `LLRestorationCNN` | LL3 stem 3×3 + **3** 个 RepNCB + head 3×3，共 **5** 层空间卷积；concat 1×1 |
| `led/archs/sid_reference/models/haar_hf_cnn.py` | 每层 HF 独立一个 RepNCB + residual |
| `led/archs/sid_reference/models/learning_dwt_repncb.py` | Refiner S2D2 → stem 3×3 → **5** 个 RepNCB → head 3×3 → concat 1×1 → D2S2，共 **7** 层空间卷积 |

RepNCB 的多分支是并行求和，每个分支最多只有一个空间 3×3；1×1 不增加空间范围，部署融合后也是一个 3×3。不能把五个分支计成五层。LL 的 `depth=5` 包含 stem/head，Refiner 的 `num_blocks=5` 不包含 stem/head。去掉 stem 激活、ReLU、逐元素加法、固定系数缩放都不扩大支持。

这里 `threshold_mode=hf_cnn` 直接处理各层子带，虽然配置写了 `context=atlas`，实际不走 atlas 阈值 CNN。也没有图像均值、方差或 attention 之类全局依赖。`normalize_bands` 并非图像统计归一化，当前 HF CNN 路径直接使用有符号未归一化系数。

### 3.1 一维精确依赖区间

二维沿两个轴分别推导。固定 Haar 核长 2、步长 2，没有重叠；三级分析后第 `k` 个 LL3 系数依赖原输入 `[8k,8k+7]`。LL3 五个 3×3 将系数范围扩大至 `[k-5,k+5]`；逆 Haar 不跨系数块重叠。

因此 wavelet preliminary 输出位置 `p` 经 LL 最长路径依赖：

```
D_wavelet(p) = [8 floor(p/8)-40, 8 floor(p/8)+47]
```

区间宽度 88。HF 在 level j 的独立 3×3 路径只覆盖三个 `2^j` 大小的块，宽度分别为 6、12、24，被 LL 路径覆盖。HF 各层不是串联的三个 CNN，不能累计为额外三个粗尺度卷积。

Refiner 的 S2D2 和七层 3×3 使最终位置 `q` 依赖 preliminary：

```
a(q) = 2 floor(q/2)-14
b(q) = 2 floor(q/2)+15
D_final(q) = [8 floor(a(q)/8)-40, 8 floor(b(q)/8)+47]
```

| q mod 8 | 最左相对距离 | 最右相对距离 | 一维支持宽度 |
|---:|---:|---:|---:|
| 0 | 56 | 55 | 112 |
| 1 | 57 | 54 | 112 |
| 2 | 58 | 61 | 120 |
| 3 | 59 | 60 | 120 |
| 4 | 60 | 59 | 120 |
| 5 | 61 | 58 | 120 |
| 6 | 54 | 57 | 112 |
| 7 | 55 | 56 | 112 |

最大单点偏移为 **61**，最大包围支持为 **120×120**，但系统仅对 8 像素平移相位一致，不能简单用 `(RF-1)/2` 当成 tile halo。

### 3.2 为什么最小对齐 halo 是 56，而不是 64

取 core 为 `[s,e)`，`s,e` 均为 8 的倍数。支持区间端点随 q 单调不减：

```
min D_final(core) = s-56
max D_final(core) = e+55
需要的输入区间 = [s-56,e+56)
```

core 内距左边界更远的像素虽然可向左看 61 像素，但已经向内移动了 5 像素，所以不会越过 `s-56`。故 **56 是该对齐布局的精确最小值**。48 会漏掉上下文；64 是额外冗余的工程选择，也是若希望采用保守的逐点最大偏移 61 后向 8 对齐的结果。偏离 8 像素相位时，单纯增加 halo 不能修复 Haar 分块相位变化。

此结论是对任意参数值的结构支持保证，不能用特定图像的 ReLU 关闭或远端权重很小来缩小它。量化若新增跨 tile 统计或改变算子语义，须重新分析。

## 4. tile / halo 候选与 FPGA 内存

详细候选和整幅图的边缘截断开销见 [tile_candidates.md](tile_candidates.md)。以 halo=56、64 为主，48 作不足 halo 对照；可增加 0、32 作接缝压力测试、80 作冗余对照。`T<=2h` 无有效 core，应排除。

`A` 表示每个激活的字节数。仅存一张 packed 输入需要 `4HWA`；输入和输出同时驻留翻倍。Refiner 的单张 W32 特征为 `32×H/2×W/2×A = 8HWA`。tile 对应分别为 `4T²A` 和 `8T²A`。两个特征 ping-pong buffer 为 `16T²A`，还未计入跳连、preliminary、Haar/HF 保留量、权重、累加器、DMA 双缓冲和布局对齐。

表中 FP16 仅是“激活按 2 bytes 存储”的容量模型，不代表现有 PyTorch 模型全链 FP16；原始 wavelet forward 在 AMP 下仍强制 FP32。INT8/INT16 也需独立量化验证。不要把单张特征容量误报为峰值片上内存。

FPGA 优先：RepNCB 部署折叠；一次处理一块；输入/输出大帧留在 DDR 或主机，core DMA 按行写回；中间张量尽量片上复用。必要时使用行缓冲流水线（3×3 通常需要两行历史和窗口寄存器），但还要安排多尺度子带及 skip 的寿命。整图流式设计也可能节省内存，分块不是必须整图缓存的唯一替代。tile 越小虽节省缓冲，但 halo 重算和 DDR 流量越大，最终用实测吞吐和资源选择。

## 5. 拼接方法与边界语义

### 5.1 首选完整图像、可变边缘 tile 的 overlap-crop

保持全图 packed 原点不变。对每个轴令 `s=0,C,2C,...`，`e=min(s+C,H)`，取 `a=max(0,s-h)`、`b=min(H,e+h)`，二维做笛卡尔组合：

```python
for y in range(0, H, C):
    y1 = min(y+C, H)
    iy0, iy1 = max(0, y-h), min(H, y1+h)
    for x in range(0, W, C):
        x1 = min(x+C, W)
        ix0, ix1 = max(0, x-h), min(W, x1+h)
        tile_output = model(image[..., iy0:iy1, ix0:ix1])
        output[..., y:y1, x:x1] = tile_output[
            ..., y-iy0:y1-iy0, x-ix0:x1-ix0]
```

内部 core 和输入起点均按 8 对齐；最右/下末块长度允许不是 8 的倍数，因为它与整图共享真实边界及同一余数。仅在全局真实边界截断 halo，不在内部补伪造上下文。core 覆盖检查应确保所有像素覆盖次数恰好为 1。

网络的真实边界语义：wavelet 在右/下 replicate 到 8 倍数，重建后裁回原尺寸；随后 refiner 单独在右/下 replicate 到 2 倍数，最后裁回。各 CNN 的空间卷积在各自特征边界采用零 padding。不能一次在整图外补足 56 或 64 后将结果裁回，这会改变多层卷积的真实图像边界行为。

固定形状 FPGA 可以分配最大 T 缓冲，但必须传入有效宽高，并在每层按有效边界实现与参考网络一致的 padding/裁切/掩码；不能直接让伪补齐像素参与全部计算。初版建议验证支持有效尺寸的边缘路径，后续再实现固定形状优化。

### 5.2 与仓库现有工具衔接

`led/utils/tiled_inference.py` 已有 crop/hann/uniform。整幅尺寸能被 8 整除时，可采用 `tile_size=T, overlap=2h, alignment=8, blend='crop'`；h=56 不符合默认 alignment=16，须显式传 8。

非 8 倍数时，现有 `tiled_inference` 会先对全图 **对称 reflect** padding，这与上述网络右下 replicate、wavelet 裁回后再 refiner 的顺序不同。因此该通用函数不能直接作为任意尺寸的严格等价实现；应采用上述原尺寸裁片方法。即使只改为一次右下 replicate，最后 refiner 的边界仍可能改变。

不要用 `crop_to_batchable_tiles` 作为完整图像实验的默认入口：它会裁掉图像边缘。不要使用 `tile_batch_size=0` 将全部 tile 一次送入设备，这会削弱峰值内存收益。FPGA 默认串行单 tile，后续按资源评估双缓冲。

### 5.3 可选平滑融合

不足 halo 的探索可将相邻输出用 separable Hann 或 uniform 权重归一化：`sum(w_i*y_i)/sum(w_i)`，外边缘权重保持非零。它可减轻接缝但不能恢复遗漏上下文或保证等价，并增加累加缓冲和除法。足够 halo 的 crop 已能保证数学等价，无需再混合包含污染边缘的完整 tile 输出。RGB ISP、去马赛克若也要分块，需要独立计算其 halo；首版先拼完整 packed RAW 再做 ISP。

## 6. 验证阶段与验收

1. **结构与基线**：固定 checkpoint、代码版本、输入归一化和 dtype；模型 eval。分别比较原网络整图/原网络 tile、deploy 整图/deploy tile，避免将 RepNCB 融合误差混入分块误差。
2. **小图正确性**：随机图、常数图、脉冲、横竖边缘与棋盘；覆盖八种位置相位、四角、最后一行/列、尺寸小于 T、H/W 非 8 倍数。精确方案 FP32 初始门槛 `atol=1e-6,rtol=1e-5`，同时记录 max/mean/p99 绝对误差及接缝 ±8 带内误差。对新后端如需调整浮点门槛，须根据整图后端差异给出依据。
3. **真实数据**：先固定各 ratio 代表图，再全部 100/250/300 验证集；统计 RAW PSNR/SSIM 和 RGB 指标、接缝图、全局边界图。先在 illumination correction 之前比较原始网络输出；若沿用原验证的全图 illumination corrector，只能拼接后执行同一次全局校正，不能逐 tile 拟合。
4. **候选扫描**：固定 h=64 扫 T=256/384/512/768/1024，再固定适合资源的 T 比较 h=48/56/64/80。固定 T 会使 h 改变 core 和 tile 数；比较时同时报告这些变化。h=48 仅质量/成本消融，不作为理论等价方案。
5. **硬件阶段**：固定校准集和量化尺度，比较同一整数算子图整图与 tile。无全局统计、同累加顺序/舍入饱和的定点图期望逐 bit 相同；FP16/混合精度另订误差与质量预算。报告片上峰值 RAM、DDR 读写字节、带宽、首帧/稳定帧时延、FPS、功耗、DSP/LUT/BRAM/URAM、时钟、有效尺寸路径耗时。GPU allocated memory 不能代替 FPGA RAM。

建议结果记录列：checkpoint/hash、backend、dtype、图像 H/W、T/h/C、tile 数、处理像素比、merge、边界策略、max/mean/p99 error、seam error、RAW/RGB PSNR/SSIM、总耗时及 DMA/compute 分解、峰值内存。

### 已执行的验证（不是未来实验结果）

- `analyze.py` 使用精确整数区间遍历 core=8..1024（步长 8）和 h=0..80（步长 8），断言最小可行 h=56；完整分辨率清单见统计附件。
- `verify_tiles.py` 用真实 `net_g_255000.pth`，严格加载，无 profile buffer 删除，CPU FP32，随机 seed=2026；原网络及 deploy 各自测试 192×256、197×259，core=64，h=48/56/64，共 12 组。
- h=56/64 的最大误差均为 `4.76837158203125e-7`，全部通过上述 allclose；h=48 四组全部不通过，最大误差约 `9.26e-5` 至 `1.98e-4`。详见 [numerical_check.json](numerical_check.json)，含权重 SHA256。
- 本节记录最初的理论验证。后续真实数据软件评估见 [FP32_CROP_RESULTS.md](FP32_CROP_RESULTS.md)；尚未进行 FPGA 综合或实机测量，不宣称某个 tile 已是硬件最优。

## 7. 复现

在仓库根目录：

```bash
python tile_software_test/analyze.py
python tile_software_test/render_tables.py
/opt/conda/envs/LED-ICCV23/bin/python tile_software_test/verify_tiles.py
```

统计脚本只需 Python 标准库。数值脚本使用现有 LED-ICCV23 环境；默认环境缺少 `einops`，无法正常导入完整 led 包。无需更改现有环境即可复现。
