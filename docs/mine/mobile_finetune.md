# 手机实采 RAW 少样本微调

从仓库根目录运行，环境为 `conda activate LED-ICCV23`，并设置 `export PYTHONPATH=.`。

## 数据与划分

原始目录：`../../Datasets/mine-mobile-image-dataset/`。

```bash
python scripts/data_preparation/prepare_mobile_pairs.py
python scripts/prepare_mobile_unet_checkpoint.py
```

准备脚本解压原始包，通过文件名的「组编号 + 时间戳」严格配对；每组 1 张 clean、3 张 noisy。固定 seed=2022，在裁剪前按完整 clean 组划分：23 组 / 69 对训练，2 组 / 6 对验证，不创建测试集。划分保存于数据目录 `prepared/train.json`、`prepared/val.json`，两套实验共用。若不同组实际拍摄的是同一场景的重复帧，应将它们进一步合并到同一划分中。

已确认 clean 仅进行了五帧平均，无其他处理。读取格式为小端 uint16，继承同组 noisy DNG 的尺寸、Bayer、黑白电平。当前 DNG 为 4096×3072、RGGB、黑电平 64、白电平 1023。packed 顺序遵循已有 UNet 的 RGBG，形状为 `[4,1536,2048]`。通道先减黑电平、除以白黑电平差；noisy 再乘逐对计算的曝光比：

```
ratio = clean_ISO * clean_EXP_ms / (noisy_ISO * noisy_EXP_ms)
```

avg5 是平均，不乘 5。训练输入和目标均限制在 [0,1]。缓存保存的是完整 packed uint16 数组（约 2.35 GiB），不是离线 patch；用 mmap 只读取需要的窗口。

## 在线增强

`MobilePairedRAWDataset` 在 packed 坐标上输出 `[4,256,256]`（即 256×256×4），对应原始 Bayer 的 512×512 区域。训练网格步长 128，即 patch 边长的一半。每次访问同一网格单元时重新采样起点，向右/下随机偏移 0…127 像素，边缘限制在合法范围内；因此重叠率随采样变化，不是固定 50%。每张图 165 个采样单元，总计 11385 个训练采样单元，训练中不缓存裁剪结果。

clean/noisy 共用同一裁剪坐标和翻转：水平、垂直翻转各以 0.5 概率独立执行，不交换 RGBG 通道、不额外旋转。验证不增强，每对取四角及中心 5 个固定 patch，共 30 个。验证指标是这些 patch 上的 RAW PSNR/SSIM 平均，不是整幅图像指标。

## 初始化和训练

指定归档权重实际为 RepNR UNet，不能直接严格加载到普通 UNet。转换脚本严格加载原权重，调用 `generalize()` 平均五个训练相机的对齐参数，再 `deploy()` 融合成普通 UNet；检验前向输出等价后，写入 `pretrained/mobile/led_generalized_unet.pth`。源权重保持原样。这里采用普通 UNet 全参数微调，未使用 LED 原论文只训练对齐分支的两阶段策略。

```bash
OMP_NUM_THREADS=4 python led/train.py -opt options/LED/finetune/Mobile_UNet_L1.yaml
OMP_NUM_THREADS=4 python led/train.py -opt options/LED/finetune/Mobile_UNet_L1_SSIM.yaml
```

两套配置仅实验名和 SSIM 损失不同：

| 参数 | 设置 |
| --- | --- |
| 优化器 | Adam，betas=(0.9,0.999)，weight_decay=0 |
| Batch | 4 / GPU |
| 总步数 | 3000（12000 个在线增强样本） |
| 学习率 | 1e-5 → 1e-6，余弦衰减 |
| Warmup | 前 100 步 |
| 验证 / 保存 | 每 250 步 |
| L1 | 权重 1.0 |
| 组合损失 | L1 + 0.1 × (1−SSIM) |
| SSIM | 现有 `SSIMLoss`，11×11 高斯窗口，sigma=1.5，data_range=1 |

参考实验归档实际 SSIM 权重为 0.3（其中注释写的是 0.1）；本次有意使用较保守的 0.1，防止小样本微调中过强改变预训练输出。若需要严格复现其权重，将 `train.ssim_opt.loss_weight` 改成 0.3。

不启用合成噪声，不做亮度拟合校正，不使用 EMA。验证使用 RAW 域，不经过 Sony 的色彩矩阵；本批 DNG 的 rawpy CCM 为空，因此目前不自动输出经项目 ISP 渲染的 RGB。后续可用实采图作独立可视化分析。

每套权重写入 `experiments/Mobile_UNet_L1/models/` 或 `experiments/Mobile_UNet_L1_SSIM/models/`，每 250 步有独立 checkpoint，最终为 `net_g_latest.pth`，参数键 `params`，架构为 `options/base/network_g/unet.yaml`。训练日志记录每次验证及最佳指标对应步数，选择最佳步数的 checkpoint 进行可视化更合适。恢复训练可在相同命令后添加 `--auto_resume`；全新重跑会由已有训练框架归档同名实验目录。

## 验证

新增测试覆盖文件名配对、组划分不相交、边缘坐标、随机起点、成对增强同步、确定性验证以及配置一致性。现有 `tests/test_ssim_loss.py::test_ablation_config_preserves_baseline` 硬编码预训练 SSIM 权重为 0.1，而当前预训练 YAML 已为 0.3，属于此次修改前的配置/测试不一致；未改动该预训练配置或其测试。
