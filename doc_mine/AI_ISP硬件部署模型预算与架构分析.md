# 面向 FPGA / ASIC 与手机部署的 AI-ISP RAW 去噪预算与架构分析

> 调研日期：2026-07-14
> 适用范围：Sensor RAW 输入，仅将去噪模块神经网络化，其余环节采用经典 ISP，最终输出可视 RGB/YUV。
> 结论性质：文中的论文/厂商数字标为“公开实测或规格”；由其推导出的预算标为“工程建议”，不是行业统一强制标准。

## 1. 先给结论

不存在一个适用于所有手机、FPGA 和 ASIC 的固定“参数量上限”或“FLOPs 上限”。同样参数量的两个模型，延迟可能相差几个数量级。对高分辨率 RAW 去噪，优先级通常是：

1. 每个传感器 RAW 像素需要多少 MAC；
2. 中间特征是否能留在片上，还是要频繁读写 DDR；
3. 算子能否完整映射到 NPU、FPGA 或 ASIC，而不回退到 CPU/GPU；
4. RAW 动态范围、权重/激活/累加器位宽；
5. 参数量与权重 SRAM；
6. 最终才是模型文件大小本身。

针对本项目，建议采用下面两套预算，不要混为一谈。

### 1.1 自研 FPGA / ASIC 的第一版目标

| 指标 | 理想起步值 | 可接受上探值 | 说明 |
|---|---:|---:|---|
| 参数量 | 50K–500K | 500K–1M | 只有权重仍可片上驻留、计算受控时才考虑 1–2M |
| 复杂度 | 500–2,000 MAC/RAW-pixel | 2,000–5,000 MAC/RAW-pixel | 超过 5,000 应优先重构网络，而不是先堆算力 |
| 权重 | W8 QAT | 首尾层 W10/W12 或混合精度 | RAW 恢复不能直接套用分类模型的 INT8 经验 |
| 激活 | A8/A10 | 敏感层 A12/A16 | RAW 输入输出保持 10–16 bit |
| 累加器 | INT32 优先 | 至少约 24 bit 并做溢出证明 | 需与训练 fake-quant 位精确一致 |
| 片上 SRAM/BRAM/URAM | 2–4 MB 作为首版规划量级 | 随器件调整 | 包含权重、行缓存、tile、FIFO 和部分和 |

对应的第一阶段里程碑应先定为 **FHD 30 fps**；验证质量、量化和数据流后，再扩展到 4K30。对 4K30 自研流式硬件，约 1 TOPS 是合理主目标，约 2.5 TOPS 已是高质量上探区间。这里统一规定：

> **1 MAC = 1 次乘加 = 2 operations/FLOPs**。
> 一些论文或工具把 1 MAC 记成 1 FLOP，因此所有比较必须先统一口径。

### 1.2 映射到现有手机 GPU/NPU 的实用预算

这是利用手机已有加速器，不是把自研 FPGA 直接“装进手机”。

| 使用场景 | 建议参数量 | 建议计算量 | Stretch goal | 主要风险 |
|---|---:|---:|---:|---|
| FHD30 连续预览/视频 | 0.1–1M，旗舰可到 2M | 5–20 GMAC/frame | 20–30 GMAC/frame | 算子回退、特征带宽、持续功耗 |
| FHD60 | 0.1–1M | 5–15 GMAC/frame | 约 20 GMAC/frame | AI 段最好控制在 8–12 ms |
| 4K30 连续视频 | 0.1–1M，谨慎到 2M | 20–50 GMAC/frame | 不建议超过约 100 GMAC/frame | 峰值 TOPS 看似够，带宽和热通常先到瓶颈 |
| 12MP 单张拍照 | 0.1–2M | 30–80 GMAC/frame | 取决于能否接受 0.2–0.8 s | 拍照可短时冲高，但不能外推到连续视频 |

以上是基于公开真机结果整理出的工程区间，而不是硬限制。最终上限只能通过目标手机、目标 delegate/runtime 和完整模型真机测量确定。

### 1.3 对当前仓库模型的直接判断

当前 deployed U-Net 约 **7.760M 参数**。按仓库当前 THOP 口径，在 `[1,4,256,256]` packed RAW 输入下约 **13.707 GMAC**；该 packed 输入对应 `512×512` 个传感器 RAW 像素，因此保守复杂度约为：

```text
13.707 GMAC / (512 × 512 RAW pixels)
= 52,288 MAC/RAW-pixel
```

逐层只计算非零卷积/反卷积乘加时，下界为 **12.096 GMAC**，即 **46,144 MAC/RAW-pixel**。两者的差异来自 stride-2 `ConvTranspose2d` 的统计：THOP 按稠密输出网格计数；跳过插零无效乘法的硬件实现会更接近后者。实际 compiler/RTL 应单独核对。因此，当前网络可写成约 **46–52K MAC/RAW-pixel**，本文整帧表采用与仓库日志一致、较保守的 THOP 数字。

即便采用较低的非零 MAC 下界，它仍比首版硬件建议的 500–2,000 MAC/RAW-pixel 高约 **23–92 倍**，比上探线 5,000 高约 **9.2 倍**。结论是：

> 当前 U-Net 适合作为训练基线或 teacher，不适合不经重构就作为手机 4K30 / FPGA 流式去噪核心。

它在 512×512 packed 训练 patch 上的数字看似不大，但扩展到整帧后会迅速增长：

| 模型 | 参数量 | MAC/RAW-pixel | FHD 单帧 | 4K 单帧 | 12MP 单帧 |
|---|---:|---:|---:|---:|---:|
| 当前 U-Net32 | 7.760M | 52,288 | 108.4 GMAC | 433.7 GMAC | 627.5 GMAC |
| 当前 NAFNet32 配置 | 7.600M | 33,482 | 69.4 GMAC | 277.7 GMAC | 401.8 GMAC |

U-Net 若按“跳过反卷积插零”的非零 MAC 下界计算，则 FHD、4K、12MP 分别约为 95.7、382.7、553.7 GMAC/frame；这不改变其不适合首版实时硬件的结论。

若要求 30 fps，并按 1 MAC = 2 ops：

| 模型 | FHD30 理想算术吞吐 | 4K30 理想算术吞吐 |
|---|---:|---:|
| 当前 U-Net32 | 6.51 TOPS | 26.02 TOPS |
| 当前 NAFNet32 | 4.17 TOPS | 16.66 TOPS |

这还没有计入归一化、拼接、激活、padding、DMA、量化/反量化和利用率损失。NAFNet 的 MAC 较低，但包含 LayerNorm、全局池化、门控和 PixelShuffle 等结构，实际硬件延迟未必比规则卷积网络更好。

---

## 2. 为什么不能只问参数量和 FLOPs

### 2.1 参数量主要回答“权重放在哪里”

权重容量为：

```text
weight_bytes = parameter_count × weight_bits / 8
```

| 参数量 | FP32 | FP16 / INT16 | INT8 |
|---:|---:|---:|---:|
| 0.1M | 0.4 MB | 0.2 MB | 0.1 MB |
| 0.5M | 2.0 MB | 1.0 MB | 0.5 MB |
| 1M | 4.0 MB | 2.0 MB | 1.0 MB |
| 2M | 8.0 MB | 4.0 MB | 2.0 MB |
| 当前 7.760M | 31.0 MB | 15.5 MB | 7.76 MB |

以常见 FPGA 原型平台为例，AMD Kria K26 官方给出的总片上 SRAM 为 26.6 Mb，约 3.33 MB。因此当前模型即便做 INT8，权重也不能全部留在该器件片上；而 0.5–1M INT8 权重才更容易给行缓存和 tile SRAM 留出空间。[AMD Kria K26 Product Brief](https://www.amd.com/content/dam/amd/en/documents/products/som/kria/k26/k26-product-brief.pdf)

### 2.2 MAC 决定计算阵列和动态能耗

卷积层 MAC 为：

```text
MAC = Hout × Wout × Cout × (Kh × Kw × Cin / groups)
```

对全卷积图像恢复网络，参数量与分辨率无关，但 MAC 基本随像素数线性增加。一个 512×512 patch 上可接受的网络，直接跑 12MP 可能增加几十倍计算。

推荐同时报告：

```text
MAC/RAW-pixel = total_MAC / number_of_sensor_RAW_pixels
MAC/frame     = MAC/RAW-pixel × width × height
MAC/s         = MAC/frame × fps
ops/s         = 2 × MAC/s          # 本文口径
```

### 2.3 激活往往比权重更致命

MicroISP 的 TFLite FP32 模型只有约 158 KB，但论文报告运行时 RAM 约为：

- Full HD：90 MB；
- 12MP：475 MB；
- 32MP：975 MB。

这说明高分辨率 ISP 的瓶颈经常是特征图，而不是模型文件。[MicroISP](https://arxiv.org/abs/2211.06770)

当前 U-Net 的四级 encoder skip，仅按必须保留的 INT8 特征图估算，其下界约为：

```text
skip_bytes ≈ RAW_width × RAW_height × (8 + 4 + 2 + 1)
           = 15 bytes/RAW-pixel
```

所以仅 skip 下界就是：

| 分辨率 | INT8 skip 下界 |
|---|---:|
| FHD | 31.1 MB |
| 4K | 124.4 MB |
| 12MP | 180 MB |

这里还没有加入 decoder 当前特征、拼接后的双通道张量、双缓冲、工作区和对齐开销。

### 2.4 算子兼容性可以压倒 FLOPs 优势

移动端 MFDNet 的实测中，某些模型虽 MAC 很低，却因 LayerNorm 或不兼容算子回退而极慢；论文报告 NAFNet 在一款设备上出现超过 1 秒的延迟。规则 3×3/1×1 Conv、ReLU/LReLU、固定下采样通常更可靠。[MFDNet](https://arxiv.org/abs/2211.04687)

Mobile AI 2021 也显示，在同一 Dimensity 1000+ APU 上，13.3 MB 的 tuned U-Net 可以比某些几十 KB 模型更快。模型尺寸不能替代真机 profiling。[Mobile AI 2021 Learned ISP](https://openaccess.thecvf.com/content/CVPR2021W/MAI/papers/Ignatov_Learned_Smartphone_ISP_on_Mobile_NPUs_With_Deep_Learning_Mobile_CVPRW_2021_paper.pdf)

---

## 3. 分辨率、帧率与每像素预算

下面给出一张可直接用于架构设计的换算表。

| 复杂度 | FHD30 | 4K30 | 12MP30 |
|---:|---:|---:|---:|
| 500 MAC/pixel | 31.1 GMAC/s = 0.062 TOPS | 124.4 GMAC/s = 0.249 TOPS | 180 GMAC/s = 0.360 TOPS |
| 1,000 MAC/pixel | 62.2 GMAC/s = 0.124 TOPS | 248.8 GMAC/s = 0.498 TOPS | 360 GMAC/s = 0.720 TOPS |
| 2,000 MAC/pixel | 124.4 GMAC/s = 0.249 TOPS | 497.7 GMAC/s = 0.995 TOPS | 720 GMAC/s = 1.440 TOPS |
| 5,000 MAC/pixel | 311.0 GMAC/s = 0.622 TOPS | 1.244 TMAC/s = 2.488 TOPS | 1.8 TMAC/s = 3.600 TOPS |

注意表中的 TOPS 是理想算术需求，不含利用率损失。若使用通用 NPU/DPU，应按目标网络实测有效利用率做余量，不能按厂商峰值的 100% 规划。

以 12MP 单张照片为例，若允许 100 ms：

- 1,000 MAC/pixel：12 GMAC/frame，需要约 0.24 TOPS；
- 2,000 MAC/pixel：24 GMAC/frame，需要约 0.48 TOPS；
- 5,000 MAC/pixel：60 GMAC/frame，需要约 1.20 TOPS。

因此单张拍照和 12MP30 连续视频是完全不同的设计问题，不能只写“支持 12MP”而不注明帧率/延迟。

---

## 4. 公开工作给出的可落地量级

### 4.1 与本项目最接近的 RAW 去噪

| 工作 | 任务与平台 | 参数/权重 | 复杂度 | 实测或结果 | 解读 |
|---|---|---:|---:|---|---|
| PMRID, ECCV 2020 | 手机 RAW 去噪，Snapdragon 855 GPU | 官方代码统计约 1.032M 参数 | 约 3.6 GMAC/RAW-MP | 约 70.7 ms/MP | 传感器噪声标定与 k-Sigma 让小模型替代更大的通用 U-Net |
| Feature-Align, WACVW 2022 | RAW 去噪 | Table 5 为 239,304 参数，INT8 约 239 KB | 2.6688 GMAC/RAW-MP | DND RAW PSNR 48.28 dB | 与本项目推荐的 2,000–5,000 MAC/pixel 高度一致 |
| Samsung RAW-to-RAW, CVPR 2026 | 手机 UHD 极暗光视频，ISP 前端 RAW 去噪 | 轻量 mRLFB，S2D，蒸馏、重参数化和量化 | 论文重点报告设备延迟/电流 | Model B 约 19.3 ms/frame；明确目标 <33 ms | 位置和系统形态与“AI 去噪 + 经典 ISP”最相似 |
| LiteDenoiseNet, CVPRW 2026 | 手机真实图像去噪，非 RAW | 1.96M | FHD 14.13 GMAC | 天玑 9500 34.0 ms；骁龙 8 Elite 46.1 ms | 1–2M 可部署，但不代表跨平台稳达 FHD30 |
| RawIR, ICIP 2024 | RAW 恢复 | 约 1.5M | 248×248×4 时约 12.3 GMAC | 质量优先 | 折算每 RAW-MP 很高，不适合作为实时视频预算 |

来源：

- [PMRID：Practical Deep Raw Image Denoising on Mobile Devices](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510001.pdf)
- [PMRID 官方代码](https://github.com/MegEngine/PMRID)
- [Feature-Align Network with Knowledge Distillation](https://openaccess.thecvf.com/content/WACV2022W/WACI/papers/Young_Feature-Align_Network_With_Knowledge_Distillation_for_Efficient_Denoising_WACVW_2022_paper.pdf)
- [Efficient Real-Time Raw-to-Raw Denoising for Extreme Low-Light Ultra HD Video on Mobile Devices](https://openaccess.thecvf.com/content/CVPR2026/papers/Pochimireddy_Efficient_Real-Time_Raw-to-Raw_Denoising_for_Extreme_Low-Light_Ultra_HD_Video_CVPR_2026_paper.pdf)
- [Real Image Denoising with Knowledge Distillation for High-Performance Mobile NPUs](https://openaccess.thecvf.com/content/CVPR2026W/MAI/papers/Kayani_Real_Image_Denoising_with_Knowledge_Distillation_for_High-Performance_Mobile_NPUs_CVPRW_2026_paper.pdf)

Feature-Align 论文自身存在参数口径不一致：Table 5 写 239,304 parameters，并明确说 INT8 约 239 KB；正文另一处写 89,696 bytes。本文采用表格与 INT8 说明一致的 239,304 参数，但保留此说明，避免把论文内部不一致当成确定事实。

PMRID 论文给出 3.6 GMAC/MP 和真机延迟，但没有直接列参数量；1,031,908 是按官方实现的卷积/反卷积权重和 bias 统计所得，约 4 MB FP32 checkpoint 也与之相符。因此表中将它明确标为“官方代码统计”，而不是论文原表数字。不同数据集、噪声范围和 ISP 显示流程下的 PSNR 不能横向排名。

### 4.2 端到端 learned ISP 的速度边界

Mobile AI 2025 的 Full-HD RAW-to-RGB 比赛在 Snapdragon 8 Elite 的 Adreno 830 和 Dimensity 9400 的 Immortalis-G925 GPU 上测试：

- 多个高效模型只有约 6–36 KB；
- 最快模型约 3.26–4.9 ms；
- MicroISP 约 155 KB，约 23–29 ms；
- PyNET 约 181 MB，约 3.4–4.7 秒。

这些结果证明“极小 ISP 网络”可以很快，也证明参数增长不是线性延迟关系。比赛任务、数据和画质目标与极暗光 RAW 去噪不同，所以它们应当作为效率边界，而不是直接照搬的质量结论。[Mobile AI 2025 Learned ISP](https://openaccess.thecvf.com/content/CVPR2025W/MAI/papers/Ignatov_Learned_Smartphone_ISP_on_Mobile_GPUs_Mobile_AI_2025_Challenge_CVPRW_2025_paper.pdf)

### 4.3 竞赛上限不是手机实时上限

AIM 2025 Real-World RAW Image Denoising Challenge 的规则是：

- 少于 15M 参数；
- `[1,4,512,512]` 输入下少于 150 GMAC；
- 禁止多模型 ensemble。

该 packed patch 对应约 `1024×1024` 个原始 Bayer 像素，因此竞赛上限约为 143 GMAC/传感器 MP。比赛没有手机真机延迟要求；这些限制是为了约束学术评测成本，不是手机全分辨率实时规格。[官方竞赛页](https://www.codabench.org/competitions/8529/)；[AIM 2025 挑战报告](https://openaccess.thecvf.com/content/ICCV2025W/AIM/papers/Li_AIM_2025_Challenge_on_Real-World_RAW_Image_Denoising_ICCVW_2025_paper.pdf)

例如当前 U-Net 在同一 packed patch 上约 54.8 GMAC，容易落入此类挑战上限；但扩展到 4K 后约 433.7 GMAC/frame，4K30 则要求理想 26 TOPS。二者并不矛盾，只是分辨率和应用场景不同。

---

## 5. 对当前 U-Net / NAFNet 的仓库实测

### 5.1 测量对象

- U-Net 配置：[options/base/network_g/unet.yaml](../options/base/network_g/unet.yaml)
- U-Net 代码：[led/archs/unet_arch.py](../led/archs/unet_arch.py)
- NAFNet 配置：[options/base/network_g/nafnet.yaml](../options/base/network_g/nafnet.yaml)
- NAFNet 代码：[led/archs/nafnet_arch.py](../led/archs/nafnet_arch.py)
- 仓库复杂度统计工具：[led/utils/model_complexity.py](../led/utils/model_complexity.py)

使用仓库的 `LED-ICCV23` 环境和 THOP，对 `[1,4,256,256]` 输入实测：

```text
U-Net32:
  parameters = 7,760,484
  MACs       = 13,706,985,472

NAFNet32:
  parameters = 7,600,228
  MACs       = 8,777,194,368
```

直接对 `[1,4,512,512]` packed patch 运行 THOP，U-Net32 与 NAFNet32 分别为 54.828、35.106 GMAC；不是用四舍五入后的 256 patch 数字反推。

当前 deploy checkpoint 中 U-Net 参数同样为 7,760,484，FP32 权重约 31.04 MB。

### 5.2 口径说明

`[1,4,256,256]` 是 2×2 Bayer packed RAW，因此对应的传感器像素区域是 `512×512`，不是 `256×256`。换算整帧时必须按传感器 RAW 像素数缩放。

THOP 返回 MAC，但当前工具日志沿用了“FLOPs”字样，并注明 THOP MAC convention。硬件规格文档建议以后统一直接写 `MACs`，同时另给 `2×MAC` 的 operations，避免口径误解。

对 U-Net 逐层复核后：

```text
THOP convention:
  13.706985472 GMAC @ [1,4,256,256]
  52,288 MAC/RAW-pixel

Non-zero Conv/ConvTranspose lower bound:
  12.096372736 GMAC @ [1,4,256,256]
  46,144 MAC/RAW-pixel
```

THOP 对 stride-2 `ConvTranspose2d` 使用输出网格统计；若硬件跳过插入的零点，实际乘加会更接近下界。为可复现性，模型报告应同时写清工具、版本、输入尺寸、反卷积口径以及是否统计 elementwise/normalization。

### 5.3 当前结构的硬件不友好点

当前 U-Net：

- 第一层即在高分辨率 packed 平面扩展到 32 通道；
- encoder 的大尺寸 skip 必须长期保存；
- decoder 使用 concat，使读写与峰值激活继续增加；
- 使用 ConvTranspose2d；
- 使用 reflect padding 和运行时 crop；
- LeakyReLU 由逐元素 max 实现。

这些在 PyTorch 中很自然，但对流式 FPGA/ASIC 和手机 NPU 不理想。主要问题不是 46 个权重 tensor，而是大尺寸激活的生命周期与反卷积/动态边界处理。

当前 NAFNet 虽然 MAC 较少，但包含：

- LayerNorm2d；
- AdaptiveAvgPool；
- 通道门控和逐元素乘法；
- PixelShuffle；
- 多级 encoder/decoder 激活。

在确定目标硬件完整支持这些算子之前，不能只根据 THOP 数字判定它更适合部署。

---

## 6. 推荐的混合 AI-ISP 数据流

建议的首版顺序是：

```text
MIPI CSI-2 RAW10/12/14
  → 解包、帧/行同步
  → DPC/BPC（坏点修复）
  → BLC（黑电平校正）与必要的行列噪声校正
  → 归一化 / 数字增益管理
  → 2×2 Bayer pack：H×W×1 → H/2×W/2×4
  → AI RAW residual denoise
  → Bayer unpack
  → LSC（或按训练定义放到 AI 前）
  → classical demosaic
  → AWB / CCM
  → gamma / tone mapping / color space conversion
  → RGB / YUV
```

关键点：

1. AI 输出应保持原 CFA 排列，便于后级经典 ISP 无缝工作；
2. 优先预测噪声残差，再由输入减去残差，较容易保持低频亮度和饱和区域；
3. ISO、模拟增益、数字增益、曝光时间、黑/白电平、温度或噪声参数应作为条件输入；
4. LSC 放在 AI 前还是后必须与训练数据严格一致：
   - 放在 AI 后，噪声较容易建模，但 LSC 会再次放大角落残余噪声；
   - 放在 AI 前，网络必须感知空间增益图和更强的空间异方差；
5. packed RAW 只是重排，不丢失 Bayer 样本，同时将空间面积降为 1/4；
6. 训练、软件参考和 RTL 必须使用完全一致的 black level、white level、gain、clamp、rounding 和 Bayer 顺序。

---

## 7. 建议的轻量 student 网络方向

保留当前 U-Net/NAFNet 作为 teacher，重新训练一个硬件友好 student，比直接剪枝当前 U-Net 更稳妥。

### 7.1 一个合理的首版搜索空间

```text
Input: packed RAW, 4 channels
  → 可选 noise/gain map
  → 3×3 Conv，基础宽度 8 或 16
  → 可选一次额外 S2D / stride-2
  → 4–8 个轻量 residual blocks
  → 1×1 / 3×3 fusion
  → residual RAW 输出
  → 与输入相减或相加
```

建议约束：

- 高分辨率阶段只保留 4–16 通道；
- 优先 3×3、1×1、add、ReLU/可移位实现的 LReLU、clip；
- depthwise/group convolution 可以显著降低理论 MAC，但必须在目标 FPGA/NPU 上验证实际利用率；
- 下采样尽量早，但必须保住细纹理与 CFA 相位；
- 用加法 skip 或压缩后的短 skip，避免大尺寸 concat；
- 训练时可使用多分支重参数化块，部署前融合为单个 3×3；
- 避免全局 attention、softmax、LayerNorm、deformable conv、动态卷积、任意 resize 和 transposed conv；
- 固定输入对齐、固定 padding 和静态 shape。

### 7.2 为什么要蒸馏和传感器噪声标定

PMRID 和 Feature-Align 都说明，小模型是否有效高度依赖：

- shot/read noise 的物理建模；
- 对目标 sensor/ISO 范围的校准；
- gain/noise conditioning；
- 大 teacher 的知识蒸馏；
- 在真实 RAW 上微调。

这通常比把 student 从 0.5M 扩到 5M 更划算。当前 LED 的 calibration-free/virtual camera 训练思路可继续作为预训练手段，但硬件最终仍应对目标手机传感器做真实 RAW 验证和少量适配。

---

## 8. FPGA / ASIC 数据流与存储设计

### 8.1 全流式层融合

- 每层有独立计算单元与行缓存；
- 中间特征不落 DDR；
- 可接近一个 packed pixel/cycle；
- 延迟确定、功耗较低；
- 面积和 DSP 随层数增加，适合很小且固定的网络。

### 8.2 时分复用 MAC 阵列

- 一套阵列依次执行多层；
- 面积较小，模型更易更新；
- 需要 tile SRAM、ping-pong 和调度器；
- 更容易受 SRAM/DDR 带宽限制。

手机 ASIC 的折中方式可以是：每 2–4 层构成 fused group，组内行流式，组间使用片上 ping-pong SRAM。

### 8.3 行缓存与整图特征的差距

4K Bayer pack 后宽度为 1920。对 C=32、INT8 的 3×3 卷积，单层两行缓存约：

```text
2 × 1920 × 32 × 1 byte
= 122,880 bytes
≈ 120 KiB
```

8 层全部流式时，粗略行缓存约 0.98 MB，再加权重、FIFO 和累加器，仍远小于保存整图特征。

### 8.4 如果每层都落 DDR

4K packed 后的 `2160/2 × 3840/2 × 32 × INT8` 单张特征约 66.4 MB。若 8 层每层都读一次、写一次：

```text
66.4 MB × 2 × 8 × 30 fps
≈ 31.9 GB/s
```

而 4K30 RAW12 单向输入仅约 0.37 GB/s。由此可见，真正需要消灭的是中间特征的反复外存访问。

### 8.5 Tile 与 halo

对 D 层连续 3×3、无下采样网络，感受野半径近似为 D。有效输出 tile 为 T×T 时，输入计算放大约为：

```text
halo_overhead = ((T + 2D) / T)^2
```

例如：

- D=8、T=256：约 12.9%；
- D=16、T=256：约 26.6%。

拼接时只保留有效中心区域，并确保训练、软件和 RTL 的 padding 完全一致，否则会出现 tile seam。

---

## 9. 量化建议

RAW10/12/14 的低照平滑区域对量化误差、banding 和色偏非常敏感。推荐：

1. 先完成 FP32 teacher/student；
2. 从 W8A8 QAT 开始，而不是只做 PTQ；
3. RAW 输入输出保持原始 10–16 bit；
4. 内部逐通道 weight scale，激活按硬件能力做 per-tensor 或受控分组；
5. INT32 累加，明确舍入、饱和和截断位置；
6. 若出现暗部 banding，将第一层、最后一层、残差输出层提高到 W10/W12 或 A10/A12；
7. bit-exact 仿真必须覆盖负数、饱和、高光 clipping、black-level 附近和最大 digital gain；
8. 比较必须经过完整经典 ISP 后再看 RGB，不能只看归一化 RAW PSNR。

Samsung 针对 RAW 恢复的相关研究显示，10/12-bit 权重有时比裸 INT8 更稳健；而其他 RGB 去噪工作也证明 W8A8 QAT 可以有效。因此最稳妥的路线是“INT8 起步 + 敏感层混合精度”，而不是一开始将所有层锁死为同一位宽。[Quantization for CNN-based RAW restoration](https://library.imaging.org/ei/articles/34/7/ISS-183)

---

## 10. 功耗、带宽和峰值 TOPS 的正确解读

### 10.1 峰值只是上界

公开规格例子：

- Snapdragon 865：AI Engine 15 TOPS，Spectra 480 ISP 2 Gpixel/s；
- Snapdragon 888：总 AI Engine 26 TOPS；
- Apple A16：16-core Neural Engine 接近 17 TOPS；
- AMD KV260 的单 B4096 DPU：1.23 TOPS INT8 峰值。

这些数字的数据类型、稀疏性、统计范围和可并发单元不同，不能横向直接等价，更不能假设去噪网络可达到 100%。

来源：

- [Qualcomm Snapdragon 865](https://www.qualcomm.com/smartphones/products/8-series/snapdragon-865-5g-mobile-platform)
- [Qualcomm Snapdragon 888](https://www.qualcomm.com/smartphones/products/8-series/snapdragon-888-5g-mobile-platform)
- [Apple A16 / iPhone 14 Pro](https://www.apple.com/newsroom/2022/09/apple-debuts-iphone-14-pro-and-iphone-14-pro-max/)
- [AMD KV260 DPU performance](https://docs.amd.com/r/3.0-English/ug1354-xilinx-ai-sdk/KV260-Vision-AI-Starter-Kit)

### 10.2 规划余量

作为早期容量规划，可把 NPU-native 规则卷积模型按峰值约 10–30% 的持续有效吞吐估算；跨芯片、算子复杂或可能回退时按 5–15% 更稳健。这是由公开真机 MAC/延迟粗略反推的工程经验，不是厂商保证。

### 10.3 功耗目标

建议把 AI 去噪块的项目目标设为：

- 连续预览/视频：优先 <0.5–1 W；整套连续 AI 工作负载不宜按超过 1–2 W 规划；
- 单张或短 burst：可暂时允许 2–3 W，但必须验证机身温升和热降频；
- FPGA 板卡功耗不能直接外推到手机 ASIC。

MicroISP 在 Dimensity 9000 上公布了 fps/W。由其数字粗略反推，APU FP16/INT8 运行 Full HD 的功耗约 2 W 量级；这是推算而非论文直接给出的瓦数，也未必是纯 AI 核功耗，故只适合做量级参考。

---

## 11. “FPGA 实现”与“部署到手机”是两条不同路径

### 11.1 FPGA 原型

FPGA 可以完成：

- 传感器/MIPI 接入；
- RAW 去噪 bit-exact 数据流验证；
- 行缓存、tile、DMA 和经典 ISP 集成；
- 资源、时钟、吞吐和板上功耗测量；
- 为 ASIC 提供功能与架构原型。

但 FPGA 板卡本身通常不会成为商用手机里的最终硬件。

### 11.2 手机 App 使用已有 GPU/NPU

Android Camera2 的 RAW capability 允许应用获取 RAW_SENSOR 并直接处理 RAW 图像；但设备是否支持、支持哪些尺寸和组合，必须运行时查询。[Android RAW capability](https://developer.android.com/reference/android/hardware/camera2/CameraMetadata.html#REQUEST_AVAILABLE_CAPABILITIES_RAW)

普通 App 能拿到 RAW，并不等于能够无缝替换 OEM 的实时预览/录像 ISP 链路。RAW 最大分辨率流可能是 stalling stream，持续帧率、并发 preview 和功耗受 Camera HAL 限制。

Android 15 已弃用 NNAPI；Google 当前建议迁移到 LiteRT/TensorFlow Lite GPU runtime 或相应厂商路径。[NNAPI Migration Guide](https://developer.android.com/ndk/guides/neuralnetworks/migration-guide)

### 11.3 真正集成到手机相机管线

若目标是在 sensor readout 后、demosaic 前插入自研 AI denoiser，需要：

- 手机 OEM / SoC 厂商的 Camera HAL 与 vendor camera stack 权限；
- ISP/NPU/自研 ASIC 的驱动、共享 buffer、同步 fence 和零拷贝通路；
- 对 3A、HDR、EIS、多摄同步、ZSL、录像和热管理做系统联调；
- 或将自研 ASIC IP 在 SoC 设计阶段集成进去。

Android Camera HAL 是框架 API 与底层驱动/硬件之间的接口；要改产品级相机管线，通常不是普通应用层 API 能完成的。[Android Camera HAL](https://source.android.com/docs/core/camera/camera3)

因此建议一开始就明确最终产品路线：

1. **FPGA → ASIC IP → OEM/SoC 集成**；或
2. **FPGA 做算法/定点验证 → 同一 student 导出 LiteRT/vendor SDK → 使用手机现有 NPU/GPU**。

两条路线可以共享模型和量化规范，但驱动、内存架构与算子集合不同。

---

## 12. 建议的开发里程碑

### 阶段 A：定义可测规格

- 指定分辨率与帧率：先 FHD30，再 4K30；
- 指定 RAW 格式：Bayer pattern、10/12/14 bit、black/white level；
- 指定 AI 段最大延迟、能量/帧、片上 SRAM 和 DDR 带宽；
- 统一 MAC/FLOP 口径；
- 将质量目标定义在完整 ISP 后，而不是只看 RAW loss。

### 阶段 B：建立 teacher 与轻量 student

- 当前 U-Net/NAFNet 作为 teacher；
- student 先限制在 ≤0.5M 参数、≤2,000 MAC/RAW-pixel；
- 传感器噪声建模 + gain/noise conditioning；
- residual RAW 输出；
- 用蒸馏追回纹理。

### 阶段 C：定点与 bit-exact

- W8A8 QAT；
- 敏感层混合 10/12 bit；
- INT32 accumulator；
- 在 Python/C++/RTL 三端固定 rounding、saturation、padding 和 Bayer order；
- 检查暗区 banding、色偏和高光 clipping。

### 阶段 D：FPGA FHD30

- 所有中间特征优先片上流式或 tile；
- 统计 DSP/LUT/BRAM/URAM、II、实际 DDR GB/s；
- 将 MIPI、pack/unpack、AI、经典 ISP、显示全部串起来；
- 记录端到端而非单 kernel 延迟。

### 阶段 E：4K30 与手机验证

- 提升像素吞吐并验证 tile seam；
- 导出 LiteRT / 厂商 SDK 模型，在至少 Qualcomm、MediaTek 各一台目标机测试；
- 确认没有 CPU fallback；
- 连续运行 10–20 分钟记录 fps、功耗和热降频；
- 若走 ASIC，进入 SRAM banking、MAC array、时钟门控、PPA 和门级功耗流程。

---

## 13. 验收清单

### 图像质量

- 不同 ISO、模拟/数字增益、曝光时间；
- 不同传感器温度与黑电平漂移；
- shot/read/row/fixed-pattern noise；
- hot/dead pixels；
- 低照平滑墙面是否 banding；
- 高频纹理是否被抹除；
- 饱和高光、暗部 clipping；
- RAW PSNR/SSIM；
- 完整 ISP 后 RGB PSNR/SSIM/LPIPS；
- ColorChecker ΔE、肤色和白平衡；
- 视频 temporal flicker。

### 硬件与系统

- 参数量与权重位宽；
- MAC/frame、MAC/RAW-pixel、MAC/s；
- 峰值与平均 activation SRAM；
- 实测 DDR bytes/frame 与 GB/s；
- MAC/DSP 利用率；
- p50/p95 单帧延迟；
- 首帧延迟与稳定 fps；
- 是否存在 unsupported-op fallback；
- tile halo 与边界接缝；
- FPGA 板上动态功耗或 ASIC activity-driven PPA；
- 10–20 分钟持续运行后的温度和降频；
- 包括 MIPI、DMA、pack、AI、unpack、经典 ISP 的端到端指标。

---

## 14. 最终建议

对当前项目，最合理的决策不是继续围绕 7.76M U-Net 做小幅剪枝，而是：

1. 保留现有模型作为质量 teacher；
2. 新建一个 50K–500K 参数、500–2,000 MAC/RAW-pixel 的规则卷积 student；
3. 通过噪声条件、传感器标定和蒸馏恢复质量；
4. 首版只承诺 FHD30，W8A8 QAT、INT32 accumulate、RAW I/O 12–16 bit；
5. 将大 skip concat、ConvTranspose、LayerNorm、动态 padding 从部署图中去掉；
6. 采用 Bayer pack、早期降采样、residual RAW、行缓存/层融合；
7. 达标后再扩到 1M 参数、2,000–5,000 MAC/pixel 和 4K30；
8. 手机端必须在真实目标机和真实 Camera HAL 数据流中验收，不能用桌面 GPU FLOPs 或 NPU 峰值 TOPS 代替。

一句话概括：

> **参数量控制在 0.5M 左右是一个好的起点，但真正的硬指标应写成“每 RAW 像素不超过约 2,000 MAC、中间特征尽量不出片、算子 100% 可映射”；当前 7.76M U-Net 的首要问题是约 46–52K MAC/RAW-pixel 和大激活，而不只是模型文件为 31 MB。**

---

## 15. 参考资料

### RAW 去噪与 learned ISP

1. [PMRID: Practical Deep Raw Image Denoising on Mobile Devices, ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510001.pdf)
2. [Feature-Align Network with Knowledge Distillation for Efficient Denoising, WACVW 2022](https://openaccess.thecvf.com/content/WACV2022W/WACI/papers/Young_Feature-Align_Network_With_Knowledge_Distillation_for_Efficient_Denoising_WACVW_2022_paper.pdf)
3. [MicroISP: Processing 32MP Photos on Mobile Devices with Deep Learning](https://arxiv.org/abs/2211.06770)
4. [Mobile AI 2021 Learned Smartphone ISP Challenge](https://openaccess.thecvf.com/content/CVPR2021W/MAI/papers/Ignatov_Learned_Smartphone_ISP_on_Mobile_NPUs_With_Deep_Learning_Mobile_CVPRW_2021_paper.pdf)
5. [Mobile AI 2025 Learned Smartphone ISP Challenge](https://openaccess.thecvf.com/content/CVPR2025W/MAI/papers/Ignatov_Learned_Smartphone_ISP_on_Mobile_GPUs_Mobile_AI_2025_Challenge_CVPRW_2025_paper.pdf)
6. [Efficient Real-Time Raw-to-Raw Denoising for Extreme Low-Light Ultra HD Video on Mobile Devices, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Pochimireddy_Efficient_Real-Time_Raw-to-Raw_Denoising_for_Extreme_Low-Light_Ultra_HD_Video_CVPR_2026_paper.pdf)
7. [LiteDenoiseNet: Real Image Denoising with Knowledge Distillation for Mobile NPUs, CVPRW 2026](https://openaccess.thecvf.com/content/CVPR2026W/MAI/papers/Kayani_Real_Image_Denoising_with_Knowledge_Distillation_for_High-Performance_Mobile_NPUs_CVPRW_2026_paper.pdf)
8. [MFDNet: memory-friendly mobile image denoising](https://arxiv.org/abs/2211.04687)
9. [RawIR: toward efficient RAW image restoration](https://arxiv.org/abs/2409.18204)
10. [AIM 2025 Real-World RAW Image Denoising Challenge](https://openaccess.thecvf.com/content/ICCV2025W/AIM/papers/Li_AIM_2025_Challenge_on_Real-World_RAW_Image_Denoising_ICCVW_2025_paper.pdf)
11. [AIM 2025 Codabench rules](https://www.codabench.org/competitions/8529/)

### FPGA / ASIC 与硬件

12. [AMD Kria KV260 Vision AI Starter Kit](https://www.amd.com/en/products/system-on-modules/kria/k26/kv260-vision-starter-kit.html)
13. [AMD ZCU104 DPU peak specification](https://docs.amd.com/r/3.0-English/ug1354-xilinx-ai-sdk/ZCU104-Evaluation-Board)
14. [AMD DPU I/O bandwidth requirements](https://docs.amd.com/r/en-US/pg338-dpu/I/O-Bandwidth-Requirements)
15. [AMD DPU depthwise convolution resource configuration](https://docs.amd.com/r/en-US/pg338-dpu/DepthwiseConv-ALU)
16. [FPGA Bilateral Grid for Real-Time Image Denoising](https://arxiv.org/pdf/2110.07186)
17. [ReTiDe: INT8 FPGA Denoising](https://rcsl-tcd.github.io/assets/pdf/Li2025CVMP.pdf)
18. [Quantization study for CNN-based RAW restoration](https://library.imaging.org/ei/articles/34/7/ISS-183)

### Android 与手机硬件

19. [Android Camera2 RAW capability](https://developer.android.com/reference/android/hardware/camera2/CameraMetadata.html#REQUEST_AVAILABLE_CAPABILITIES_RAW)
20. [Android Camera HAL](https://source.android.com/docs/core/camera/camera3)
21. [Android NNAPI Migration Guide](https://developer.android.com/ndk/guides/neuralnetworks/migration-guide)
22. [Qualcomm Snapdragon 865](https://www.qualcomm.com/smartphones/products/8-series/snapdragon-865-5g-mobile-platform)
23. [Qualcomm Snapdragon 888](https://www.qualcomm.com/smartphones/products/8-series/snapdragon-888-5g-mobile-platform)
24. [Apple A16 / iPhone 14 Pro](https://www.apple.com/newsroom/2022/09/apple-debuts-iphone-14-pro-and-iphone-14-pro-max/)
