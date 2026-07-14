# `ref-script`：BIN 文件处理流程

本文基于 `ref-script/bin_process.py` 与 `ref-script/raw10_rggb_to_png.py` 的实现分析。仓库中没有提交实际的 `.bin` 样本，因此分辨率、像素格式、Bayer 排列与黑电平均是脚本的**固定假设**，不是从文件中自动识别出的元数据。

## 结论概览

`ref-script` 把 `.bin` 当作 **MIPI RAW10 打包的单通道 Bayer 原始帧**处理，且固定为：

| 项目 | 脚本假设 |
| --- | --- |
| 分辨率 | `W=1280`、`H=720` |
| 像素格式 | MIPI RAW10 packed：4 个 10-bit 像素占 5 字节 |
| Bayer 排列 | RGGB，左上角为 R |
| 字节数 | `1280 × 720 × 5 / 4 = 1,152,000` 字节 |
| 数值范围 | 解包后 `uint16`，有效范围 0–1023 |
| 黑/白电平 | 默认黑电平 64、白电平 1023 |

两条独立输出链路如下：

```text
*.bin
  │
  ├─ bin_process.py
  │    RAW10 解包 → RGGB 四通道打包 → 写入带元数据的 .npz
  │                                      │
  │                                      └→ 供 LED 的 NPZ Dataset 读取/训练
  │
  └─ raw10_rggb_to_png.py
       RAW10 解包 → 简易 RGGB 去马赛克 → 8-bit RGB PNG（快速目视检查）
```

## 1. `.bin` 的读取与 RAW10 解包

两份脚本都使用 `np.fromfile(path, dtype=np.uint8)` 逐字节读取二进制文件。文件首先必须通过严格的字节数检查：

```python
expected = W * H * 5 // 4
if buf.size != expected:
    raise ValueError(...)
```

因此当前实现**不支持**行对齐填充（stride/padding）、帧头、帧尾或多帧拼接；这些额外数据会使大小检查失败。由于 RAW10 每组编码 4 像素，宽度也必须可被 4 整除；当前的 1280 满足该条件。

### RAW10 的 5 字节 → 4 像素规则

文件被重排为 `(-1, 5)`，记每组字节为 `b0, b1, b2, b3, b4`。代码假定前 4 个字节分别保存四个像素的高 8 位，`b4` 从低到高依次保存这四个像素各自的低 2 位：

```text
b0 = p0[9:2]       b4[1:0] = p0[1:0]
b1 = p1[9:2]       b4[3:2] = p1[1:0]
b2 = p2[9:2]       b4[5:4] = p2[1:0]
b3 = p3[9:2]       b4[7:6] = p3[1:0]
```

对应代码公式为：

```python
px0 = (b0 << 2) | (b4 & 0x03)
px1 = (b1 << 2) | ((b4 >> 2) & 0x03)
px2 = (b2 << 2) | ((b4 >> 4) & 0x03)
px3 = (b3 << 2) | ((b4 >> 6) & 0x03)
```

接着四个像素按原始光栅顺序交错写入一维数组，再 reshape 成 `(720, 1280)` 的 `uint16` 马赛克图：

```python
raw[0::4], raw[1::4], raw[2::4], raw[3::4] = px0, px1, px2, px3
raw = raw.reshape(H, W)
```

这一逻辑位于 [bin_process.py](../ref-script/bin_process.py) 的 `unpack_mipi_raw10` 和 [raw10_rggb_to_png.py](../ref-script/raw10_rggb_to_png.py) 的顶层解包代码中。

## 2. `bin_process.py`：转换为 LED 可读取的 NPZ

### 2.1 输入目录与遍历方式

`process_bin_dataset(input_dir, save_dir, bl_val=64.0)` 只查找下列两个一级目录中、扩展名恰为小写 `.bin` 的文件：

```text
<input_dir>/long/*.bin   → <save_dir>/long/*.npz
<input_dir>/short/*.bin  → <save_dir>/short/*.npz
```

当前脚本入口写死为：

```python
input_dir = "/home/aa509/Desktop/bin"
save_dir = "/home/aa509/Desktop/npz"
```

需要在运行前修改为实际路径。遍历使用 `glob.glob()`，未排序，也不递归进入子目录。单个文件的大小不匹配时，会打印错误并跳过该文件，其他文件继续处理。

### 2.2 RGGB 马赛克到四通道 packed RAW

解包后的二维马赛克图按 RGGB 规则取样：

```text
原始 2×2 单元                 输出通道与形状

(0,0) R    (0,1) G1          im[0] = R  = raw[0::2, 0::2]
(1,0) G2   (1,1) B           im[1] = G1 = raw[0::2, 1::2]
                               im[2] = B  = raw[1::2, 1::2]
                               im[3] = G2 = raw[1::2, 0::2]
```

因此输出 `im` 的 shape 是 **`(4, 360, 640)`**，通道顺序为 **`[R, G1, B, G2]`**。这与项目的 `led.data.raw_utils.pack_raw_bayer()` 对 RGGB 的输出顺序一致；虽然二维 RGGB 的第二行顺序是 `G2, B`，四通道张量中为了统一接口采用 `[R, G1, B, G2]`。

`pack_rggb_bayer_raw_scale()` 会先转为 `float32`，但并未做黑电平扣除、归一化、白平衡或增益，故 `im` 中仍是 0–1023 RAW DN 值。

### 2.3 写出的 NPZ 结构

每个 `.bin` 会生成同名 `.npz`，文件使用 `np.savez()` 保存以下字段：

| 字段 | shape / dtype | 当前来源和含义 |
| --- | --- | --- |
| `im` | `(4, 360, 640)` / `float32` | packed RGGB RAW，范围通常为 0–1023 |
| `black_level` | `(4, 1, 1)` / `float32` | 四通道均为 `bl_val`，默认 64 |
| `white_level` | `(4, 1, 1)` / `float32` | 四通道均为 1023 |
| `wb` | `(4,)` / `float32` | `[1, 1, 1, 1]` 占位白平衡 |
| `ccm` | `(3, 3)` / `float32` | 单位矩阵，占位颜色校正矩阵 |

项目中的 `RAWGTDataset` 及 MIPI 数据读取代码可直接读取这一组字段。成对去噪数据集会使用：

```python
normalized = (im - black_level) / (white_level - black_level)
```

按默认元数据，数值约变换为 `(im - 64) / 959`。若像素低于 64，归一化值为负；是否截断由后续数据集/模型流程决定。

## 3. `raw10_rggb_to_png.py`：仅用于快速可视化

该脚本没有生成 NPZ，而是固定读取一个文件并写出一个 RGB PNG：

```python
in_path = "./tdm_in_raw10_0000.bin"
out_path = "./tdm_in_raw10_0000_rggb_raw10.png"
```

RAW10 解包与前述过程完全相同，随后执行：

1. 将 10-bit 值线性映射到 8-bit：`round(raw * 255 / 1023)`；
2. 根据 RGGB 坐标建立 R、G、B 掩码；G 使用两个绿色位置；
3. 对每个颜色平面，把自身和 8 邻域中该颜色位置的值求平均，填补未采样位置；
4. 堆叠为 `RGB`，裁剪到 0–255 后以 `uint8` PNG 保存。

这里的去马赛克是调试级的邻域平均，不是相机 ISP：未扣黑电平、未做白平衡、CCM、镜头阴影校正、降噪、gamma 或色调映射。它适合确认 RAW10 解包和 Bayer 排列是否大体正确，不应视作颜色或画质结果。邻域实现用 `np.roll()`，图像边缘会与对侧像素环绕相邻，也可能在边界产生轻微伪影。

## 4. 与现有 LED 推理链路的关系

`bin_process.py` 的输出格式与项目训练/评测用的 NPZ 元数据格式兼容，但当前 [scripts/image_process.py](../scripts/image_process.py) 的可视化入口只接受能由 `rawpy.imread()` 读取的相机 RAW（例如 ARW、DNG）。它**不能直接读取本目录的裸 `.bin` 或本脚本产出的 `.npz`**。

因此当前仓库中的关系是：

| 目标 | 可用入口 |
| --- | --- |
| 把 `long/short` 的 RAW10 BIN 制作为训练/数据集样本 | `bin_process.py` → NPZ，再由相应 Dataset 读取 |
| 快速检查 BIN 内容和 RGGB 假设 | `raw10_rggb_to_png.py` → PNG |
| 使用 `quick_op/05_visualize_rgb.sh` 对 BIN 做 LED 推理并输出 RGB | 当前尚未接入，需要增加 BIN/NPZ 输入与 RGB 后处理路径 |

## 5. 使用前必须核实的项目

1. **分辨率与行 stride**：确认每帧的有效宽高真的是 1280×720，且每行没有对齐填充。若有 padding，不能用整帧 `reshape(-1, 5)`；必须先按每行有效字节数截取再解包。
2. **RAW10 打包次序**：确认设备是本脚本采用的 MIPI RAW10 低 2 位拼接次序。某些采集/ISP 导出的格式可能使用不同字节序、16-bit 对齐容器或已解包的 RAW10。
3. **Bayer pattern**：确认左上角确为 R（RGGB）。若实际是 BGGR、GRBG 或 GBRG，虽然能得到图像，但颜色通道会错位；需同时修改 packed 取样和 PNG 掩码。
4. **黑电平与饱和值**：`64/1023` 只是当前假设。应从传感器规格、采集端配置或暗场/平场标定获得真实值；错误黑电平会直接影响归一化和噪声统计。
5. **色彩元数据**：`wb=1` 与 `ccm=I` 仅使 NPZ 字段齐全，不代表真实相机色彩。若要得到可信 RGB，应标定或读取真实 WB/CCM，并使用正确的 ISP。
6. **long/short 配对与曝光比**：脚本只按目录把文件转成 NPZ，不验证同名配对、曝光时间或数字增益比例。训练成对去噪前应另行建立并检查配对清单。

## 6. 建议的最小验收

对任意一帧，建议按以下顺序确认：

```bash
# 1. 原始文件大小应为 1,152,000 字节
stat -c '%n %s bytes' path/to/frame.bin

# 2. 先将 raw10_rggb_to_png.py 中的输入/输出路径改为该文件，再运行
python ref-script/raw10_rggb_to_png.py

# 3. 检查 PNG 是否存在明显的条纹、四像素周期错位或异常颜色；
#    这通常分别提示 stride/字节序、RAW10 位拼接或 Bayer pattern 假设错误。
```

通过目视检查后，再使用 `bin_process.py` 批量生成 NPZ，并用 `np.load()` 检查 `im.shape == (4, 360, 640)`、`black_level.shape == white_level.shape == (4, 1, 1)` 及字段是否齐全。
