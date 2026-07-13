# LED 虚拟相机与噪声图生成机制整理

本文整理本次会话中关于 LED 虚拟相机的分析，重点说明：

- LED 中虚拟相机是什么、有多少个。
- `options/base/noise_g/noise_g_virtual.yaml` 中各参数的含义。
- `predefined_noise_g` 指向的 `.pth` 文件到底保存了什么。
- 训练时如何用虚拟相机在线生成 noisy RAW。
- 如何改成不止 5 个虚拟相机，或添加新的噪声分布。

相关代码主要位于：

- `led/data/noise_utils/noise_generator.py`
- `led/data/noise_utils/common.py`
- `led/models/raw_image_denoising_model.py`
- `options/base/noise_g/noise_g_virtual.yaml`

## 1. 虚拟相机是什么

LED 中的“虚拟相机”不是一个真实相机，也不是一个 CNN 网络，而是一组传感器噪声参数。每个虚拟相机对应一套 read noise、row noise、color bias 等参数，用来把 clean RAW 在线合成为 noisy RAW。

默认配置中有 5 个虚拟相机：

```yaml
noise_g:
  type: VirtualNoisyPairGenerator
  virtual_camera_count: 5
```

运行时会打印类似：

```text
Current Using Cameras: ['IC0', 'IC1', 'IC2', 'IC3', 'IC4']
```

这 5 个虚拟相机通常也对应 RepNR 的 5 个 CSA 分支：

```yaml
repnr_opt:
  branch_num: 5
```

也就是说：

```text
IC0 -> CSA branch 0
IC1 -> CSA branch 1
IC2 -> CSA branch 2
IC3 -> CSA branch 3
IC4 -> CSA branch 4
```

训练时随机选一个 `camera_id`，同时用于：

1. `noise_g` 选择哪一台虚拟相机生成噪声。
2. RepNR 选择对应的 CSA 分支前向。

## 2. 默认 noise_g_virtual.yaml

默认虚拟相机配置如下：

```yaml
noise_g:
  type: VirtualNoisyPairGenerator
  virtual_camera_count: 5
  noise_type: ptrqc
  shuffle: false
  sample_strategy: coverage
  param_ranges:
    K: [0.1, 30]
    Gaussian:
      slope: [0.5407496082896145, 0.6771267783987617]
      bias: [0.799410107756515, 2.079863926921898]
      sigma: [0.21408231874600314, 0.26751211630129734]
    TukeyLambda:
      slope: [0.46210495986526035, 0.6558756156508007]
      bias: [-0.09044499670414502, 1.5370438794284738]
      sigma: [0.23920060971003035, 0.3888189465002228]
      lambda: [-0.24285714285714288, 0.2]
    Row:
      slope: [0.44829219631519307, 0.7292220505388829]
      bias: [-1.7977103277112132, -0.5917002648887661]
      sigma: [0.2321799439489326, 0.29428813016491145]
    ColorBias:
      bias: [-1.9080039, 4.080531]
      sigma: [0.006866596, 3.0379589]
      count: 16
```

### 2.1 type

```yaml
type: VirtualNoisyPairGenerator
```

表示使用虚拟相机噪声生成器。对应类在：

```text
led/data/noise_utils/noise_generator.py
```

类名：

```python
class VirtualNoisyPairGenerator(nn.Module):
```

### 2.2 virtual_camera_count

```yaml
virtual_camera_count: 5
```

表示构建 5 台虚拟相机。它控制下列参数向量的长度：

```text
tukey_lambdas: [5]
read_slopes:   [5]
read_biases:   [5]
read_sigmas:   [5]
row_slopes:    [5]
row_biases:    [5]
row_sigmas:    [5]
color_biases:  [5, 16, 4]
```

其中 `color_biases` 的第二维 16 来自 `ColorBias.count`，不是虚拟相机数量。

### 2.3 noise_type

```yaml
noise_type: ptrqc
```

每个字母代表一种噪声：

```text
p: photon shot noise / Poisson shot noise
t: Tukey-Lambda read noise
r: row noise
q: quantization noise
c: color bias
```

因此 `ptrqc` 表示同时添加：

```text
shot noise + Tukey read noise + row noise + quantization noise + color bias
```

代码中目前主要支持：

```text
p: shot noise
g: Gaussian read noise
t: Tukey-Lambda read noise
r: row noise
q: quantization noise
c: color bias
```

注意：`g` 和 `t` 不建议同时使用。代码逻辑是：

```python
if 'g' in noise_type:
    使用 Gaussian read noise
elif 't' in noise_type:
    使用 Tukey-Lambda read noise
```

如果同时写了 `g` 和 `t`，会优先使用 Gaussian。

### 2.4 sample_strategy

```yaml
sample_strategy: coverage
```

表示用覆盖式采样。代码中对应：

```python
sample = self.split_range if self.sample_strategy == 'coverage' else self.uniform_range
```

`coverage` 会把参数范围均匀切成 `virtual_camera_count` 个点。例如虚拟相机数量为 5 时，一个参数范围 `[min, max]` 会被采成：

```text
min
min + 1/4 * (max - min)
min + 2/4 * (max - min)
min + 3/4 * (max - min)
max
```

所以默认 5 台虚拟相机会覆盖从低噪声参数到高噪声参数的一组代表性配置。

如果使用非 `coverage` 的策略，代码会走 `uniform_range()`，即在范围内随机采样。

### 2.5 shuffle

```yaml
shuffle: false
```

表示采样后不打乱顺序。通常 `IC0` 更靠近参数范围低端，`IC4` 更靠近参数范围高端。

如果设成 `true`，采样出的参数点会被打乱，但仍然是这些覆盖点。

## 3. 各噪声参数的含义

### 3.1 K: overall system gain

```yaml
K: [0.1, 30]
```

`K` 是 overall system gain，用于控制 shot noise 强度。

生成每个 batch 的噪声时，`K` 不是固定取一个值，而是在 log 空间随机采样：

```python
log_K_max = torch.log(self.k_range[1])
log_K_min = torch.log(self.k_range[0])
log_K = uniform(log_K_min, log_K_max)
K = exp(log_K)
```

shot noise 公式在 `common.py` 中：

```python
def shot_noise(x, k):
    return torch.poisson(x / k) * k - x
```

可以理解为：

```text
noisy = Poisson(signal / K) * K
shot_noise = noisy - signal
```

`K` 越大，shot noise 越强。

注意：`K` 范围是所有虚拟相机共享的，不是每个虚拟相机固定一个 K。

### 3.2 Gaussian

```yaml
Gaussian:
  slope: [...]
  bias: [...]
  sigma: [...]
```

当 `noise_type` 中包含 `g` 时使用 Gaussian read noise。

read noise 的强度不是固定值，而是和当前采样到的 `K` 有关：

```python
mu = log(K) * slope + bias
read_sigma = exp(Normal(mu, sigma))
```

然后生成高斯读出噪声：

```python
read_noise = torch.randn_like(x) * read_sigma
```

默认 `noise_type: ptrqc` 不包含 `g`，所以默认不会使用 `Gaussian` 这一段。

### 3.3 TukeyLambda

```yaml
TukeyLambda:
  slope: [...]
  bias: [...]
  sigma: [...]
  lambda: [...]
```

默认 `noise_type: ptrqc` 中包含 `t`，所以默认使用 Tukey-Lambda read noise。

每个虚拟相机会采样一套：

```text
read_slope
read_bias
read_sigma
tukey_lambda
```

read noise 强度同样由 `K` 控制：

```python
mu = log(K) * slope + bias
read_sigma = exp(Normal(mu, sigma))
```

然后使用 Tukey-Lambda 分布生成 read noise：

```python
U = torch.rand_like(x)
Y = 1 / lambda * (U ** lambda - (1 - U) ** lambda)
read_noise = Y * read_sigma
```

Tukey-Lambda 的好处是能模拟比 Gaussian 更复杂的读出噪声尾部分布。

### 3.4 Row

```yaml
Row:
  slope: [...]
  bias: [...]
  sigma: [...]
```

Row 表示行噪声参数。它也根据 `K` 采样强度：

```python
mu = log(K) * row_slope + row_bias
row_sigma = exp(Normal(mu, row_sigma))
```

行噪声生成函数：

```python
def row_noise(x, scale, loc=0):
    B, _, H, W = x.shape
    noise = torch.randn((B, H * 2, 1)) * scale + loc
    noise = noise.repeat((1, 1, W * 2))
    return _pack_bayer(noise)
```

它先在原始 Bayer mosaic 尺度上生成按行共享的噪声，再 pack 成 4 通道 RAW。

### 3.5 ColorBias

```yaml
ColorBias:
  bias: [-1.9080039, 4.080531]
  sigma: [0.006866596, 3.0379589]
  count: 16
```

ColorBias 用于模拟不同 Bayer 通道的颜色偏置。packed RAW 通常有 4 个通道：

```text
R, G1, B, G2
```

代码会为每台虚拟相机构建：

```text
16 个 color bias 档位，每个档位 4 个通道偏置
```

所以整体形状是：

```text
color_biases: [virtual_camera_count, 16, 4]
默认即 [5, 16, 4]
```

生成噪声时，会根据当前采样到的 `K` 落在哪个区间来选择对应的 color bias：

```python
i_range = (Kmax - Kmin) / color_bias_count
index = ((K - Kmin) // i_range).long()
color_bias = color_biases[current_camera, index]
```

`count: 16` 不是 16 个虚拟相机，而是每台虚拟相机内部有 16 个随 K 变化的 color bias 档位。

## 4. 训练时如何生成 noisy RAW

训练入口在：

```text
led/models/raw_image_denoising_model.py
```

核心逻辑：

```python
self.camera_id = torch.randint(0, len(self.noise_g), (1,)).item()

scale = self.white_level - self.black_level
self.gt = (self.gt - self.black_level) / scale

self.gt, self.lq, self.curr_metadata = self.noise_g(
    self.gt, scale, self.ratio, self.camera_id
)
```

可以拆成 4 步。

### 4.1 随机选虚拟相机

```python
camera_id = randint(0, len(noise_g))
```

默认：

```text
camera_id in {0, 1, 2, 3, 4}
```

训练时传入的是一个标量 `camera_id`，所以当前 batch 通常使用同一台虚拟相机生成噪声。

### 4.2 GT RAW 归一化

```python
scale = white_level - black_level
gt_norm = (gt_raw - black_level) / scale
```

得到：

```text
gt_norm: [B, 4, H, W], 范围约 [0, 1]
```

### 4.3 在低曝光 raw 数值域中加噪

进入 `VirtualNoisyPairGenerator.forward()` 后：

```python
img_gt = clamp(img, 0, 1)
img = img_gt * scale / ratio
```

这里的含义是：

```text
img_gt 是干净目标图，归一化到 [0, 1]
img 是模拟低曝光下的 raw 信号数值
```

然后采样 `K`，并生成各类噪声：

```python
K = sample_overall_system_gain()

shot_noise
read_noise
row_noise
quant_noise
color_bias
```

### 4.4 合成 noisy RAW

代码中：

```python
for n in noise.values():
    img += n
img /= scale
img = img * ratio
img = clamp(img, max=1.0)
```

等价理解：

```text
noisy_low = low_signal
          + shot_noise
          + read_noise
          + row_noise
          + quant_noise
          + color_bias

lq = noisy_low / scale * ratio
lq = clamp(lq, max=1.0)
```

最终返回：

```python
return img_gt, img_lq, metadata
```

其中：

```text
img_gt: clean target, [B, 4, H, W]
img_lq: synthetic noisy input, [B, 4, H, W]
metadata:
  vcam_id
  noise
  noise_params
```

## 5. predefined_noise_g 的 .pth 是什么

配置中常见：

```yaml
path:
  predefined_noise_g: pretrained/noise_g/VirtualNoisyPairGenerator_ELD_ptrqc_5VirtualCameras.pth
```

或：

```yaml
path:
  predefined_noise_g: /home/shared_files/model_weights/led/VirtualNoisyPairGenerator_ELD_ptrqc_5VirtualCameras.pth
```

这个 `.pth` 不是去噪网络权重，也不是一个可学习的噪声网络。它保存的是：

```text
VirtualNoisyPairGenerator.state_dict()
```

也就是已经采样好的虚拟相机噪声参数：

```text
tukey_lambdas
read_slopes
read_biases
read_sigmas
row_slopes
row_biases
row_sigmas
color_biases
```

它通常很小，只有几 KB，因为里面只是参数表。

加载逻辑：

```python
self.noise_g = self.build_noise_g(opt['noise_g'])

noise_g_path = self.opt['path'].get('predefined_noise_g', None)
if noise_g_path is not None:
    self.load_network(self.noise_g, noise_g_path, strict_load_g, None)
```

这里 `param_key=None` 表示直接从 `.pth` 根目录读取 state_dict。

### 5.1 为什么需要 predefined_noise_g

如果不加载 `.pth`，每次实验启动时会根据 YAML 重新采样虚拟相机参数。尤其是 `color_biases` 中有随机项，所以不同运行可能得到不同的虚拟相机集合。

使用 `predefined_noise_g` 可以保证：

```text
每次实验使用完全相同的 IC0~IC4 噪声参数
预训练结果可复现
RepNR 的 5 个 CSA 分支和虚拟相机参数稳定对应
```

训练初始化后，代码还会保存当前使用的噪声生成器参数：

```python
torch.save(self.noise_g.state_dict(), experiments/.../noise_g.pth)
```

因此如果你自己重新生成了虚拟相机，可以把实验目录里的 `noise_g.pth` 留作以后复现。

## 6. 可以不用 pth，直接按自己的设定构建虚拟相机吗

可以。

只需要设置：

```yaml
path:
  predefined_noise_g: ~
```

这样代码会直接根据 YAML 中的 `param_ranges` 构建虚拟相机。

例如：

```yaml
noise_g:
  type: VirtualNoisyPairGenerator
  virtual_camera_count: 8
  noise_type: ptrqc
  shuffle: false
  sample_strategy: coverage
  param_ranges:
    K: [0.1, 30]
    TukeyLambda:
      slope: [0.46, 0.66]
      bias: [-0.09, 1.54]
      sigma: [0.24, 0.39]
      lambda: [-0.24, 0.2]
    Row:
      slope: [0.45, 0.73]
      bias: [-1.80, -0.59]
      sigma: [0.23, 0.29]
    ColorBias:
      bias: [-1.91, 4.08]
      sigma: [0.0069, 3.038]
      count: 16
```

如果不使用 Gaussian read noise，则可以省略 `Gaussian` 段；如果代码或配置校验没有依赖它，保留也没问题。

## 7. 可以把虚拟相机数量改成不止 5 个吗

可以。关键是同步修改 RepNR 分支数。

如果使用 RepNR：

```yaml
noise_g:
  virtual_camera_count: 8

repnr_opt:
  branch_num: 8

path:
  predefined_noise_g: ~
```

推荐保持：

```text
virtual_camera_count == repnr_opt.branch_num
```

否则可能出现：

```text
noise_g 生成 camera_id = 7
RepNR 只有 5 个 CSA 分支
```

这会导致分支索引不匹配。

如果不使用 RepNR，只训练普通去噪网络，那么 `virtual_camera_count` 可以自由设置，因为网络本身不需要 CSA 分支。

### 7.1 不能直接加载旧的 5 相机 pth

如果你把虚拟相机改成 8 个，就不能再加载：

```text
VirtualNoisyPairGenerator_ELD_ptrqc_5VirtualCameras.pth
```

因为旧 `.pth` 中参数形状是 5 个相机，例如：

```text
read_slopes: [5]
color_biases: [5, 16, 4]
```

而新配置需要：

```text
read_slopes: [8]
color_biases: [8, 16, 4]
```

这会导致 state_dict shape 不匹配。

正确做法：

```yaml
path:
  predefined_noise_g: ~
```

从头按新配置生成 8 个虚拟相机，并重新预训练网络。

## 8. 可以添加新的噪声分布吗

可以，但只改 YAML 不够，需要改代码。

当前 `VirtualNoisyPairGenerator` 只识别：

```text
p, g, t, r, q, c
```

如果想增加新噪声，例如：

```text
b: banding noise
h: hot pixel noise
f: fixed pattern noise
```

需要做几件事。

### 8.1 在 common.py 中添加噪声函数

例如：

```python
def banding_noise(x, scale):
    ...
    return noise
```

### 8.2 在 sample_virtual_cameras 中采样新参数

例如：

```python
if 'b' in self.noise_type:
    banding_slope_range = self.param_ranges['Banding']['slope']
    self.banding_slopes = sample(self.virtual_camera_count, banding_slope_range, ...)
    self.banding_slopes = nn.Parameter(self.banding_slopes, False)
```

如果希望新参数保存进 `noise_g.pth`，需要使用 `nn.Parameter(..., False)` 或 `register_buffer()`。

### 8.3 在 forward 中添加新噪声项

例如：

```python
if 'b' in self.noise_type:
    banding_param = self.sample_banding_sigma()
    noise['banding'] = banding_noise(img, banding_param)
```

这样它就会在 `add_noise()` 中被加到图像上：

```python
for n in noise.values():
    img += n
```

### 8.4 在 YAML 中添加参数段

例如：

```yaml
noise_type: ptrqcb
Banding:
  slope: [...]
  bias: [...]
  sigma: [...]
```

## 9. 如何单独生成虚拟 noisy 图片

训练时 noisy 图是在内存中在线生成的，不会默认保存到磁盘。若要单独生成，可以按下面伪代码使用 `VirtualNoisyPairGenerator`。

```python
import torch
from led.data.noise_utils.noise_generator import VirtualNoisyPairGenerator
from led.utils.options import yaml_load

opt = yaml_load('options/base/noise_g/noise_g_virtual.yaml')
noise_g = VirtualNoisyPairGenerator(opt['noise_g'], device='cuda').cuda()

# 如果需要复现实验中的 5 个虚拟相机参数，则加载 predefined_noise_g
state = torch.load(
    'pretrained/noise_g/VirtualNoisyPairGenerator_ELD_ptrqc_5VirtualCameras.pth',
    map_location='cuda'
)
noise_g.load_state_dict(state, strict=True)

# 假设 gt_raw 已经是 packed RAW: [B, 4, H, W]
# black_level / white_level / ratio 来自数据 metadata
scale = white_level - black_level
gt_norm = (gt_raw - black_level) / scale

camera_id = 0
gt_clean, lq_noisy, meta = noise_g(gt_norm, scale, ratio, camera_id)
```

其中：

```text
gt_clean: 干净目标图，归一化 RAW
lq_noisy: 合成 noisy RAW，可送入去噪网络
meta:     本次使用的虚拟相机编号、噪声张量和噪声参数
```

如果要保存可视化图片，还需要把 packed RAW 解包/走 ISP 或者保存为 `.npy/.npz`。LED 训练本身直接使用 packed RAW 张量，不要求先转成 RGB。

## 10. 一次完整训练前向流程

可以用下面流程概括 LED 预训练中的虚拟相机噪声生成：

```text
1. DataLoader 读取 clean GT RAW 和 metadata
   gt_raw: [B, 4, H, W]
   black_level, white_level, ratio

2. 随机选择虚拟相机
   camera_id = randint(0, virtual_camera_count)

3. 归一化 GT
   scale = white_level - black_level
   gt_norm = (gt_raw - black_level) / scale

4. 根据 camera_id 选择虚拟相机参数
   IC0 / IC1 / IC2 / ...

5. 采样本次 K
   K ~ Uniform(log(Kmin), log(Kmax)) in log space

6. 根据 K 和虚拟相机参数生成噪声
   shot noise
   Tukey/Gaussian read noise
   row noise
   quantization noise
   color bias

7. 合成 noisy RAW
   lq_noisy = add_noise(gt_norm, noise, scale, ratio)

8. 去噪网络前向
   如果是 RepNR:
       output = net_g(lq_noisy, camera_id)
   否则:
       output = net_g(lq_noisy)

9. 计算损失
   loss = L1(output, gt_clean)
```

## 11. 推荐修改模板

### 11.1 使用 8 个虚拟相机和 RepNR

```yaml
repnr_opt:
  branch_num: 8
  align_opts:
    init_weight: 1.0
    init_bias: 0.0

noise_g:
  type: VirtualNoisyPairGenerator
  virtual_camera_count: 8
  noise_type: ptrqc
  shuffle: false
  sample_strategy: coverage
  param_ranges:
    K: [0.1, 30]
    TukeyLambda:
      slope: [0.46210495986526035, 0.6558756156508007]
      bias: [-0.09044499670414502, 1.5370438794284738]
      sigma: [0.23920060971003035, 0.3888189465002228]
      lambda: [-0.24285714285714288, 0.2]
    Row:
      slope: [0.44829219631519307, 0.7292220505388829]
      bias: [-1.7977103277112132, -0.5917002648887661]
      sigma: [0.2321799439489326, 0.29428813016491145]
    ColorBias:
      bias: [-1.9080039, 4.080531]
      sigma: [0.006866596, 3.0379589]
      count: 16

path:
  predefined_noise_g: ~
```

### 11.2 使用 Gaussian read noise

```yaml
noise_g:
  noise_type: pgrqc
```

此时会使用：

```yaml
Gaussian:
  slope: [...]
  bias: [...]
  sigma: [...]
```

而不是 `TukeyLambda`。

### 11.3 固定一组自定义虚拟相机

第一次训练：

```yaml
path:
  predefined_noise_g: ~
```

训练启动后会保存：

```text
experiments/你的实验名/noise_g.pth
```

以后复现实验时：

```yaml
path:
  predefined_noise_g: experiments/你的实验名/noise_g.pth
```

## 12. 关键注意事项

1. 默认 LED 是 5 个虚拟相机，但数量可以改。
2. 使用 RepNR 时，`virtual_camera_count` 应等于 `repnr_opt.branch_num`。
3. 改虚拟相机数量后，不能再加载旧的 `5VirtualCameras.pth`。
4. `predefined_noise_g` 保存的是噪声参数，不是去噪模型权重。
5. `ColorBias.count=16` 是 color bias 档位数量，不是虚拟相机数量。
6. 默认 `noise_type=ptrqc` 使用 Tukey-Lambda read noise，不使用 Gaussian 段。
7. 新增噪声分布需要同时改 `common.py`、`noise_generator.py` 和 YAML。
8. noisy 图默认在线生成，不会自动保存成图片文件。

