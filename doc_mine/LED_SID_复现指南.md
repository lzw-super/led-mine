# LED 在 SID Sony A7S II 上的复现指南

本文面向已经配置好 Python/CUDA 环境、且 SID 数据集位于
`/home/shared_files/dataset/SID/` 的情况，记录如何复现本仓库中 LED 的
`CVPR20/ELD` 训练设置，并在 SID Sony A7S II 上完成微调与评测。

> 本文的“ELD/CVPR20”指 ELD 风格的噪声建模设置。LED 预训练使用 SID 的
> 干净 long RAW 作为图像内容，但还需要一个预先标定好的 ELD 虚拟相机噪声
> 生成器权重；不需要下载完整 ELD 图像数据集。

## 1. 选择复现路径

LED 的流程分为两个阶段：

```text
SID long 干净 RAW + ELD 虚拟噪声生成器
             │
             ▼
预训练（257,600 iterations）
             │
             ▼
SID 6 对真实 noisy/clean RAW 微调（1,500 iterations）
             │
             ▼
SID 测试集 ×100 / ×250 / ×300 评测
```

推荐优先使用路径 A；它使用官方预训练模型并复现 SID 上的 LED 微调和测试。

| 路径 | 适用目标 | 需要的权重 | 训练命令 |
| --- | --- | --- | --- |
| A. 快速复现（推荐） | 得到 SID 上的最终 LED 模型 | 虚拟噪声权重可不下载；下载 LED 预训练网络 | 仅运行微调 |
| B. 完整复现 | 从 LED 预训练开始训练 | 下载虚拟噪声权重；预训练网络不需要 | 预训练后再微调 |

相关配置：

- 预训练：[`options/LED/pretrain/CVPR20_ELD_Setting.yaml`](../options/LED/pretrain/CVPR20_ELD_Setting.yaml)
- 微调：[`options/LED/finetune/SID_SonyA7S2_CVPR20_ELD_Setting.yaml`](../options/LED/finetune/SID_SonyA7S2_CVPR20_ELD_Setting.yaml)
- SID 测试集：[`options/base/dataset/test/SID_SonyA7S2_val_split.yaml`](../options/base/dataset/test/SID_SonyA7S2_val_split.yaml)

## 2. 已下载 SID 数据的构成

本地目录应包含：

```text
/home/shared_files/dataset/SID/
├── Sony/
│   ├── long/             # 231 张 clean long-exposure .ARW
│   ├── short/            # 2,697 张 low-light short-exposure .ARW
│   ├── long_pack/        # 231 个已打包的 4 通道 Bayer .npy
│   ├── short_pack/       # 2,697 个已打包的 4 通道 Bayer .npy
│   └── long_post_int/    # 231 个后处理 RGB .npy
├── Sony_train_list.txt   # 161 个场景、1,865 对
├── Sony_val_list.txt     # 20 个场景、234 对
├── Sony_test_list.txt    # 50 个场景、598 对
├── Sony_npy/             # 由 ARW 转换的 2,928 个带元信息 NPZ（约 66 GiB）
└── Sony_train_long_patches/ # 预训练所需的 1,848 个 NPZ patch（约 3.6 GiB）
```

曝光时间为：

- `long`：10 s 或 30 s；
- `short`：0.033 s、0.04 s 或 0.1 s；
- 有效增益倍率为 ×100、×250、×300。对 `10 / 0.033 ≈ 303` 的样本，数据加载器会截断到 ×300。

`*_pack/*.npy` 的形状为 `(4, 1424, 2128)`，但不含 LED 数据加载器所需的
`black_level`、`white_level`、`wb`、`ccm` 元信息。因此不要把这些 `.npy`
直接填入本仓库的 YAML；下文会由 `.ARW` 生成带元信息的 `.npz`。

## 3. 前置检查

### 3.1 Python 包

本仓库需要能读取 Sony `.ARW` 的 `rawpy` 和 `opencv-python`：

```bash
python - <<'PY'
import torch
import cv2
import rawpy

print('torch:', torch.__version__)
print('cuda:', torch.cuda.is_available(), 'gpus:', torch.cuda.device_count())
print('cv2:', cv2.__version__)
print('rawpy:', rawpy.__version__)
PY
```

为严格复现仓库指标，请使用项目 [`docs/install.md`](../docs/install.md) 中说明的
ELD 定制版 `rawpy`。普通 PyPI `rawpy` 可能无法读取完全相同的相机色彩矩阵。

### 3.2 磁盘空间

`bayer_to_npy.py` 会把全部 2,928 张 ARW 转为未压缩 `.npz`，约需 66 GiB；
预训练 patch 约需 3.6 GiB，加上元信息和日志后建议至少预留 80 GiB 空间。

```bash
df -h . /home/shared_files
```

本仓库约定将上述生成数据直接保存到共享目录；仓库内
`datasets/ICCV23-LED` 只是一条指向 `/home/shared_files/dataset/SID` 的软链接。
不要删除原始 ARW。

## 4. 建立项目数据链接

在仓库根目录执行：

```bash
mkdir -p datasets
ln -s /home/shared_files/dataset/SID datasets/ICCV23-LED

# 预期输出：/home/shared_files/dataset/SID
readlink -f datasets/ICCV23-LED
```

若此前已经在仓库内生成过 `Sony_npy` 和 `Sony_train_long_patches`，先在共享目录
可写的机器上完成一次迁移，再建立上述链接：

```bash
mv datasets/ICCV23-LED/Sony_npy /home/shared_files/dataset/SID/
mv datasets/ICCV23-LED/Sony_train_long_patches /home/shared_files/dataset/SID/
rm datasets/ICCV23-LED/Sony  # 仅删除旧的内部 Sony 软链接，不删除 ARW
rmdir datasets/ICCV23-LED
ln -s /home/shared_files/dataset/SID datasets/ICCV23-LED
```

若目标软链接已存在，先确认它是否指向正确位置；不要直接执行会覆盖已有文件的命令。

## 5. 下载并放置权重

```bash
mkdir -p pretrained/noise_g pretrained/network_g
```

### 路径 A：快速复现所需的预训练网络

下载 [LED CVPR20/ELD Ratio100-300 预训练网络](https://drive.google.com/file/d/1kIN_eyNd4mlKhPV4PMmgzaoE3ddagjNU/view?usp=drive_link)，并命名为：

```text
pretrained/network_g/LED_Pretrain_None_None_CVPR20_Setting_Ratio100-300.pth
```

该名称与微调配置中的 `path.pretrain_network_g` 完全一致；放对位置后无需修改 YAML。

### 路径 B：完整预训练所需的虚拟噪声生成器

下载 [VirtualNoisyPairGenerator ELD ptrqc 5VirtualCameras](https://drive.google.com/file/d/1XgL8ofYcD0LWGI0VpaG6LjtddmSoUL2j/view?usp=drive_link)，并命名为：

```text
pretrained/noise_g/VirtualNoisyPairGenerator_ELD_ptrqc_5VirtualCameras.pth
```

预训练配置会从该路径加载 5 个虚拟相机的 `ptrqc` 噪声参数。

用下面命令确认文件已放好：

```bash
ls -lh \
  pretrained/noise_g/VirtualNoisyPairGenerator_ELD_ptrqc_5VirtualCameras.pth \
  pretrained/network_g/LED_Pretrain_None_None_CVPR20_Setting_Ratio100-300.pth
```

路径 A 只要求第二个文件；路径 B 只要求第一个文件，随后会自行产出预训练网络。

## 6. 一次性数据预处理

### 6.1 生成预训练 patch

预训练数据集 `RAWGTDataset` 从
`datasets/ICCV23-LED/Sony_train_long_patches/*.npz` 读取数据。运行：

```bash
python scripts/data_preparation/extract_bayer_subimages_with_metadata.py \
  --data-path datasets/ICCV23-LED/Sony/long \
  --save-path datasets/ICCV23-LED/Sony_train_long_patches \
  --suffix ARW \
  --n-thread 10
```

脚本将每个 4 通道 packed RAW 裁为默认的 `512×512` patch，并把像素与元信息写入
`*.ARW.npz`。当前 Sony 数据的尺寸固定，因此预期有 `231 × 8 = 1848` 个 patch：

```bash
find datasets/ICCV23-LED/Sony_train_long_patches -name '*.npz' | wc -l
# 预期：1848
```

### 6.2 生成验证用 `.npz`

微调训练直接读取 ARW，但训练结束后的验证与独立测试都读取 `Sony_npy` 下的
`.ARW.npz`。运行：

```bash
python scripts/data_preparation/bayer_to_npy.py \
  --data-path datasets/ICCV23-LED/Sony \
  --save-path datasets/ICCV23-LED/Sony_npy \
  --suffix ARW \
  --n-thread 8
```

预期转换 `231 + 2697 = 2928` 个 ARW：

```bash
find datasets/ICCV23-LED/Sony_npy -name '*.npz' | wc -l
# 预期：2928

test -f datasets/ICCV23-LED/Sony_npy/short/10191_00_0.1s.ARW.npz && echo OK
```

不要直接运行 `bash scripts/data_preparation/accelerate.sh`：它还会尝试处理当前未下载的 ELD 数据集。

## 7. 路径 A：用官方预训练模型进行 SID 微调

确认第 5 节中的 LED 预训练模型已放到 `pretrained/network_g/` 后，执行：

```bash
python led/train.py \
  -opt options/LED/finetune/SID_SonyA7S2_CVPR20_ELD_Setting.yaml
```

该配置的行为如下：

- 用 `datasets/txtfiles/SID/SonyA7S2/fewshot_2x.txt` 中 6 对真实图像训练；
- 前 1,000 iterations 优化对齐参数，后 500 iterations 优化 OOMN 分支；
- 训练集在内存中加载，原始 ARW 由 `rawpy` 读取；
- 最终模型保存为：

```text
experiments/LED_Finetune_SID_SonyA7S2_CVPR20_ELD_Setting/models/net_g_latest.pth
```

实时日志位于：

```text
experiments/LED_Finetune_SID_SonyA7S2_CVPR20_ELD_Setting/train_*.log
```

## 8. 路径 B：完整预训练后微调

### 8.1 预训练

确认虚拟噪声权重已按第 5 节路径放置，然后运行：

```bash
python led/train.py \
  -opt options/LED/pretrain/CVPR20_ELD_Setting.yaml
```

该配置使用 `RAWGTDataset`、5 个虚拟相机、倍率 `[100, 300]` 和 L1 损失，运行
257,600 iterations。预训练完成后模型路径为：

```text
experiments/LED_Pretrain_CVPR20_ELD_Setting/models/net_g_latest.pth
```

### 8.2 将本地预训练结果用于微调

不要编辑原始 YAML，可在命令行覆盖预训练模型路径：

```bash
python led/train.py \
  -opt options/LED/finetune/SID_SonyA7S2_CVPR20_ELD_Setting.yaml \
  --force_yml \
  path:pretrain_network_g=experiments/LED_Pretrain_CVPR20_ELD_Setting/models/net_g_latest.pth
```

微调输出路径与第 7 节相同。

## 9. SID 测试与可视化

`net_g_latest.pth` 的 LED 微调 checkpoint 同时保存了部署状态字典
`params_deploy`。评测时必须添加 `--led`，脚本才会使用这个键：

```bash
python scripts/benckmark.py \
  -t sid_led_cvpr20 \
  -p experiments/LED_Finetune_SID_SonyA7S2_CVPR20_ELD_Setting/models/net_g_latest.pth \
  --dataset SID SonyA7S2 \
  --led
```

若还要保存图像，附加 `--save_img`：

```bash
python scripts/benckmark.py \
  -t sid_led_cvpr20_img \
  -p experiments/LED_Finetune_SID_SonyA7S2_CVPR20_ELD_Setting/models/net_g_latest.pth \
  --dataset SID SonyA7S2 \
  --led \
  --save_img
```

结果与日志写入 `results/sid_led_cvpr20/` 或 `results/sid_led_cvpr20_img/`。评测会分别报告
SID ×100、×250、×300 三个验证集的 RAW PSNR/SSIM。

## 10. 训练划分与复现实验说明

仓库的默认预处理命令会处理 `Sony/long/` 下全部 231 个场景，因而也包含官方测试场景
的干净 long RAW。这是作者脚本的默认行为，适合按仓库设置复现。

若实验协议要求严格避免任何测试场景内容进入预训练，应仅选取
`Sony_train_list.txt` 中的 161 个训练场景的 long RAW 生成 patch，并使
`datasets/ICCV23-LED/Sony_train_long_patches` 只包含这些 patch。请在实验记录中明确说明
你选择的是“作者默认设置”还是“严格 train-only 设置”；两者结果不可直接视为完全相同的复现。

## 11. 常见问题

| 现象 | 原因与处理 |
| --- | --- |
| `FileNotFoundError: ...Sony...` | 检查第 4 节软链接，以及 `readlink -f datasets/ICCV23-LED/Sony` 输出。 |
| `ModuleNotFoundError: rawpy` 或 `cv2` | 当前环境缺少对应包；按第 3.1 节检查并安装。 |
| 验证阶段找不到 `*.ARW.npz` | 第 6.2 节尚未完成，或 `Sony_npy` 没有放在配置预期位置。 |
| 预训练无法加载 `VirtualNoisyPairGenerator...pth` | 完整路径 B 缺少噪声生成器权重，或文件名与第 5 节不一致。 |
| 微调无法加载 `LED_Pretrain...pth` | 路径 A 缺少官方预训练网络；路径 B 请使用第 8.2 节的 `--force_yml` 覆盖。 |
| 测试时 checkpoint key 不匹配 | 对微调后的 LED 模型必须使用 `scripts/benckmark.py ... --led`，它会加载 `params_deploy`。 |
| 预处理写盘失败 | 检查可用磁盘空间；完整 `.npz` 转换需要约 66 GiB，patch 还需约 3.6 GiB。 |

## 12. 最小验收清单

在启动训练前，下面命令应全部成功：

```bash
test -L datasets/ICCV23-LED/Sony
test -f datasets/ICCV23-LED/Sony_npy/long/00001_00_10s.ARW.npz
test -f datasets/ICCV23-LED/Sony_npy/short/10191_00_0.1s.ARW.npz

# 路径 A：检查官方预训练模型
test -f pretrained/network_g/LED_Pretrain_None_None_CVPR20_Setting_Ratio100-300.pth

# 路径 B：检查虚拟噪声生成器
test -f pretrained/noise_g/VirtualNoisyPairGenerator_ELD_ptrqc_5VirtualCameras.pth
```

只执行路径 A 时，不需要最后一项；只执行路径 B 的预训练时，不需要官方预训练模型。
