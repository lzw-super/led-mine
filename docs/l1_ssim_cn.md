# RAW 域 L1 + SSIM 对照实验

目标：`L = L1(pred, gt) + λ × (1 - SSIM(pred, gt))`。
起点 λ=0.1，建议对照 0、0.05、0.1、0.2；这只是实验起点，并不保证最佳。
L1 约束像素重建，SSIM 比较局部亮度、对比度与结构。SSIM 不是锐化项，
不能保证减少模糊或提高 PSNR。参考：https://ece.uwaterloo.ca/~z70wang/research/ssim/

实现使用 11×11、σ=1.5 的高斯窗口，在四通道 packed RAW 上独立计算后取平均，
data_range=1，采用有效卷积边界，不对预测值 clamp，也不进行显示 ISP。
训练日志 l_pix 和 l_ssim 均为已加权项。半精度输入使用 float32 计算局部统计。
RAW SSIM 与经过 ISP 的 RGB SSIM 不等价。

新配置：`options/LED/pretrain/CVPR20_ELD_Setting_L1_SSIM.yaml`。
原配置保持纯 L1；预训练和 LED 微调模型均支持可选的 `train.ssim_opt`。
新配置按当前实验要求改为原始卷积 U-Net，显式禁用 RepNR，从头训练。
数据仅使用 SID（训练为干净长曝光 RAW，验证为 SID 配对数据），噪声仍由 LED
VirtualNoisyPairGenerator 在线生成；复用归档实验的 `noise_g.pth` 相机噪声参数。
实验名为 `LED_Pretrain_SID_PlainUNet_L1_SSIM`。对照纯 L1 时，也应使用相同的
原始卷积 U-Net，以免将网络结构的变化与损失的变化混淆。

如从现有 checkpoint 继续做短实验，需要先确认权重结构与原始 U-Net 匹配。
归档的 `net_g_latest.pth` 是 RepNR 权重，不能直接加载到当前原始卷积网络；
应先通过 `scripts/prepare_led_deploy_checkpoint.py` 转换，再配置相应的
`path.pretrain_network_g` 和 `path.param_key_g: params_deploy`。
使用新的实验名称、较小学习率（例如 1e-5）和相匹配的短训练调度；
`resume_state` 保持空，避免恢复旧优化器/学习率进度。
纯 L1 对照也应从同一 checkpoint 开始，使用相同训练步数和数据随机种子。

评估同时检查配对验证集 PSNR/SSIM、固定 RAW ISP 下的边缘/纹理与噪声，
以及真实视频的逐帧闪烁。RAW-crate 没有干净 GT，不能直接用它监督 SSIM，
也不能计算可信的参考型 PSNR/SSIM。上一轮推理采用假设的传感器参数和
64 倍输入增益，预训练合成噪声与真实噪声的差异也应作为模糊的候选原因检查。

本变更仅实现与验证损失，没有启动长时间训练，也没有生成新的训练权重。
