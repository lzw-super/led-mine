# LED 原始结构预训练对比

三个配置沿用 `options/LED/pretrain/CVPR20_ELD_Setting.yaml` 的数据、噪声、训练及验证/logger base 文件，替换网络配置和实验名，并明确设置 `repnr_opt: null`。不会进行 LED RepNR 转换；参考结构原有的 RepNCB / MRLFN 模块保持不变。原始参考配置中的损失、训练超参数及设备编号均不迁入。

| 脚本参数 | 配置（位于 options/LED/pretrain） | 结构 |
|---|---|---|
| dwt | CVPR20_ELD_LearningDWT_Haar_L3_RepNCB_W32.yaml | Haar L3，LL width32/depth5/RepNCB，无 LL normalization，concat 1x1，HF RepNCB PReLU residual，refiner width32/5 blocks |
| splitternet | CVPR20_ELD_SplitterNet.yaml | 4 通道，width32，4 层 splitting，无 LayerNorm |
| mrlfn | CVPR20_ELD_MRLFN_Paper_S2D_K4_N4_D32.yaml | Bayer mosaic S2D k4，N4，d32 |

共同训练参数：单 GPU、batch size 1、8 workers、seed 2022；Adam lr=1e-4、betas=[0.9,0.999]、weight_decay=0；257600 iterations，无 warmup；128800/231840 iterations 时学习率降到 5e-5/1e-5；L1Loss。验证每 1288 iterations，开启 illumination correction。数据和预定义噪声模型路径与基准一致。

在项目根目录、已安装项目依赖的环境下执行（最后的数字是物理 GPU 编号）：

```bash
bash scripts/train_led_arch_comparison.sh dwt 0
bash scripts/train_led_arch_comparison.sh splitternet 0
bash scripts/train_led_arch_comparison.sh mrlfn 0
```

按顺序运行全部三个：

```bash
bash scripts/train_led_arch_comparison.sh all 0
```

断点续训示例：

```bash
bash scripts/train_led_arch_comparison.sh dwt 0 --auto_resume
```

可用 `PYTHON=/path/to/python bash scripts/train_led_arch_comparison.sh ...` 指定环境。
实验分别写入 `experiments/LED_Pretrain_CVPR20_ELD_<结构名>/`，TensorBoard 使用各自实验名。

模型实现保存在 `led/archs/sid_reference/`，不依赖原项目路径；入口负责 LED 与参考项目 RAW 通道顺序转换。MRLFN k4 原实现要求 packed RAW 的高、宽均为偶数；本配置未改变原结构的尺寸约束。

验证：在 `/opt/conda/envs/LED-ICCV23/bin/python` 环境下执行以下命令，三个结构均通过配置一致性、通道适配、前向/反向和一次 Adam 更新检查：

```bash
python -m unittest discover -s tests -p test_sid_reference_archs.py -v
```

当前工作区缺少基准指定的 `pretrained/noise_g/VirtualNoisyPairGenerator_ELD_ptrqc_5VirtualCameras.pth`。正式训练前必须将相应噪声权重补齐到此路径；配置保留基准路径，未改成随机噪声参数。尚未执行真实数据的完整训练。
