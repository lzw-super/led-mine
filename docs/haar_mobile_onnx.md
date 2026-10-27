# Haar L3 RepNCB 手机端 ONNX

已从 `experiments/LED_Pretrain_CVPR20_ELD_LearningDWT_Haar_L3_RepNCB_W32/models/net_g_latest.pth` 的 `params` 导出：

`experiments/LED_Pretrain_CVPR20_ELD_LearningDWT_Haar_L3_RepNCB_W32/onnx/net_g_latest_mobile_1x4x360x640.onnx`

同目录 `.json` 保存权重 SHA256、ONNX SHA256、验证误差、算子统计和软件版本。原始 PTH 未修改。

## 输入输出

- 输入名 `raw`，输出名 `denoised_raw`。
- 固定 NCHW `[1,4,360,640]`，float32；即宽 640、高 360 的四通道 packed RAW，对应 1280×720 Bayer 原图。
- 通道顺序保持 LED 的 `[R,G1,B,G2]`，不是 RGB 图片，也不是 HWC 排列。
- 图内不包含黑电平/白电平归一化、曝光倍率调整、白平衡、去马赛克或 ISP。调用方需沿用训练/验证预处理。LED 验证路径先按通道计算 `(raw-black)/(white-black)`，乘曝光 ratio，再限制上界到 1；该配置不截断负值。
- 模型输出未做裁剪，仍为四通道 RAW。未包含仅用于验证指标的 illumination correction。

## 等价转换

- 三层 Haar DWT：固定 2×2、stride=2 的 Conv；IWT：对应 ConvTranspose。
- S2D：固定 one-hot 2×2、stride=2 的 Conv；D2S：对应 ConvTranspose。
- 11 个 RepNCB 的训练分支融合为单个 3×3 Conv，各自 PReLU 保留。这是原始结构的等价推理融合，不添加 LED RepNR。
- 输入输出通道排列合入固定卷积权重，去除 Gather。
- 因固定尺寸可被 8 整除，去除补边和裁剪；直接处理 LL/HF，不构建无用 atlas、阈值或辅助输出。
- 保留必要的 HF 残差 Add、LL 和 refiner 拼接融合；未删减有效模型计算。
- ONNX 共 49 节点：Conv 22、ConvTranspose 4、PRelu 11、Relu 1、Add 3、Concat 5、Split 3。无 Identity、Pad、Slice、Reshape、Transpose、Gather、SpaceToDepth、DepthToSpace。

## 验证与复现

ONNX opset 17，FP32，尚未量化。ONNX checker 和 ONNX Runtime CPU 验证通过。对完整目标尺寸的均匀随机、带负值的低照度模拟、边界脉冲三类输入，分别比较原始 PyTorch、融合 PyTorch 和 ONNX 输出。与原始模型最大绝对误差为 `7.152557373046875e-7`（校验容差 atol=1e-5、rtol=1e-4）。未进行手机 NPU 编译或真机性能测试。

```bash
/opt/conda/envs/LED-ICCV23/bin/python scripts/export_haar_mobile_onnx.py
```

脚本默认导出上述 720p Bayer 对应尺寸；可通过 `--height`、`--width`、`--output` 指定其他尺寸和文件名，高宽必须为 8 的倍数。环境已安装 `onnx==1.16.2`、`onnxruntime==1.19.2`。
