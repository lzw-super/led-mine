# 手动执行：QAT 模型 96/16、96/20、96/24 的 crop 测试

目标权重：`quantization/artifacts/qat_llstem_noact_phase_exp3_pres2d/relu_llstem_noact/qat_w8_mixed_qdq.pth`。

脚本已准备，**未执行完整或少量数据集评估**。仅执行了下文的小尺寸 CPU 一致性检查。运行后再分析实测结果。

## 1. 正式命令

在 LED-mine 仓库根目录执行：

```bash
set -o pipefail
/opt/conda/envs/LED-ICCV23/bin/python tile_software_test/benchmark_qat_crop.py \
  --artifact quantization/artifacts/qat_llstem_noact_phase_exp3_pres2d/relu_llstem_noact/qat_w8_mixed_qdq.pth \
  --output tile_software_test/results_qat_96 \
  --device cuda:0 \
  --batch-size 64 \
  --serial-repeats 3 \
  --resume 2>&1 | tee -a tile_software_test/qat_crop.log
```

`--resume` 可以首次运行时使用。中断后原样重跑；已完整完成的图像不会重复计算，一张图只在全部配置和必要的串行计时完成后原子写入。配置、权重、输入清单或受检查的代码变化会拒绝恢复，以免混合结果。要从头重新测量，请指定另一个 `--output` 目录。

默认运行全部 93 对图像（100× 29、250× 29、300× 35），每图均为 4×1424×2128，三组参数固定为 96/16、96/20、96/24，不裁掉原图边缘。程序遇到 CUDA OOM 等错误会停止；可降低 batch-size 并使用新的输出目录重跑，不与旧 batch 的时间混合。

可先由你执行每个 ratio 一张图的流程检查：

```bash
/opt/conda/envs/LED-ICCV23/bin/python tile_software_test/benchmark_qat_crop.py \
  --max-per-ratio 1 --serial-repeats 0 \
  --output tile_software_test/results_qat_96_smoke --resume
```

正式命令不要带 `--max-per-ratio`。小样本输出状态为 `smoke_complete`，不作为完整精度结论。

## 2. 实际测什么

这是 **W8、混合 A8/A16、固定尺度的 Q/DQ 仿真**。权重/激活按导出的整数码与尺度模拟，但计算使用 FP32 卷积与 round/clamp/multiply，不是原生 INT8 算子，也不是 FPGA 位精确模拟。**计时包含 FX Python 解释器及 Q/DQ 的软件开销，不能用于推断整数硬件加速比。** 为避免不必要的统计开销，测试关闭了原解释器中逐节点饱和计数；量化数值不变。

- 严格加载 artifact 中的权重、激活尺度、量化边界、bit map、graph；不重新校准、不调用 freeze、不按 tile 自适应尺度。
- 校验 QAT 元数据对应的 `net_g_latest.pth` 源权重哈希，并实时测量这份 FP32 整图基线。它的文件哈希不同于此前 tile 实验的 `net_g_255000.pth`，不能未经核对直接混用旧 FP32 指标。
- 同一图同时计算 FP32 整图、QDQ 整图、QDQ 三种 tile 输出。
- **量化损失**：配套 FP32 整图 PSNR − QDQ 整图 PSNR。
- **额外分块损失**：QDQ 整图 PSNR/SSIM − QDQ tile PSNR/SSIM。这是本次比较三组 halo 的主要指标。
- **总损失**：配套 FP32 整图 PSNR − QDQ tile PSNR。正值表示变差，负值表示该指标提升。
- 指标沿用已有 tile 软件实验：标准 PSNR（无 1e-8 epsilon）、RAW 四通道 SSIM、crop_border=2、全图 illumination correction（拼接后按 GT 拟合单个标量）。同时记录未校正 PSNR 与未经校正的输出差异。与 `quantization/evaluate_qat.py` 的 MSE+1e-8 指标有口径差异，不能直接逐位比较绝对 PSNR。

## 3. 边界处理与计时

导出的 `MobileHaarRepNCB` 是无 padding 的静态图。96/20 的部分边缘 tile 尺寸不是 8 倍数，直接送入该图会出现尺寸问题。适配器执行：

1. 仅在 Haar 输入分支右/下 replicate 到 8 倍数；
2. 在最终 Haar 重建后裁回原 tile 尺寸，再进入 pre-S2D A8 cast 和 refiner；
3. noisy skip 保持原尺寸，裁去 halo 后写回 core。

本任务所有完整图和 tile 宽高均为偶数，适配器明确拒绝奇数尺寸；没有把整幅 tile 一次补齐后穿过整个 refiner，避免改变原网络的边界含义。h=20 保留指定值，其内部起点为 4 mod 8，存在第三级 Haar 相位变化；h=16/24 对齐。

- 批量时间：最多 64 个同形状 tile 一批；预热后对全部图分别计时，汇总平均值。
- 串行时间：每个 ratio 的第一张完整图，batch=1、预热后重复 3 次；汇总共 9 次的中位数。另测 QDQ 整图的相同重复计时。
- CUDA 同步 wall time；计入窗口分组、模型、crop 写回，排除读取文件、H2D、指标及亮度校正。输入输出均在 GPU。AMP/TF32 关闭，cuDNN deterministic=True。
- QDQ 的浮点卷积在不同 batch/后端下可能发生微小舍入差异并跨过量化阈值，所以串行/批量不强制逐 bit 或简单浮点 allclose；真实差异会写入 `serial_timing.csv`。正式全集精度对应 batch=64，串行三图只用于时间与差异检查。

## 4. 输出与后续分析

输出在 `tile_software_test/results_qat_96/`：

- `REPORT.md`：自动生成的核心结果表；
- `summary.csv` / `summary.json`：全体与分 ratio 的结果；
- `full_image.csv`：配套 FP32 与 QDQ 整图基线、量化损失和时间；
- `per_image.csv`：完整运行后 279 条 tile 记录；
- `serial_timing.csv`：默认 36 条计时记录，包含串行/批量误差；
- `metadata.json`：权重/代码/数据清单哈希、位宽、设置及完成状态；
- `images/*.json`：93 个可恢复的逐图检查点。

看到 `DONE: .../REPORT.md` 且 metadata 的 `status` 为 `complete` 后，告知已运行完成即可继续分析。不会自动覆盖或合并原 FP32 实验结果。

## 5. 已执行的小尺寸验证

```bash
/opt/conda/envs/LED-ICCV23/bin/python tile_software_test/benchmark_qat_crop.py \
  --check-only --device cpu
```

已通过：32×40 输入与原 StaticPTQ 输出逐位一致；28×36、32×36、28×40 输入与独立显式插入 padding/crop 的原 QDQ 图逐位一致；三组 crop 在 identity 模型下完整重建输入。此命令不读取验证图像、不创建正式结果目录、不进行数据集计时。

## 6. 追加七组：80/8、80/10、80/16、88/10、88/16、88/20、96/10

用独立入口指定配对组合，不做 tile×halo 笛卡尔积，不会重复跑原来的 96/16、96/20、96/24。原脚本保持不变，已有三组结果及恢复校验不受影响。

在仓库根目录执行：

```bash
set -o pipefail
/opt/conda/envs/LED-ICCV23/bin/python tile_software_test/benchmark_qat_pairs.py \
  --pairs 80:8 80:10 80:16 88:10 88:16 88:20 96:10 \
  --output tile_software_test/results_qat_extra7 \
  --device cuda:0 --batch-size 64 --serial-repeats 3 \
  --resume 2>&1 | tee -a tile_software_test/qat_crop_extra7.log
```

默认 artifact 仍是本页顶部指定的 `qat_w8_mixed_qdq.pth`。时间和指标口径与前一轮相同，记录配套 FP32 整图、QDQ 整图以及七组 tile；三组 ratio 总共 93 幅图。halo=10/20 按指定值执行，存在 Haar 相位偏移，不自动改成 8 的倍数。

默认生成：

- `tile_software_test/results_qat_extra7/REPORT.md`、`summary.csv`、`summary.json`；
- `per_image.csv`：651 条（93×7）分块结果；
- `full_image.csv`：93 条整图基线；
- `serial_timing.csv`：72 条默认串行计时（3 幅×8 配置〔含 QDQ 整图〕×3 次）；
- `metadata.json` 和 93 个 `images/*.json` 断点文件。

可先由你执行小样本检查（不混入正式目录）：

```bash
/opt/conda/envs/LED-ICCV23/bin/python tile_software_test/benchmark_qat_pairs.py \
  --pairs 80:8 80:10 80:16 88:10 88:16 88:20 96:10 \
  --max-per-ratio 1 --serial-repeats 0 \
  --output tile_software_test/results_qat_extra7_smoke --resume
```

仅检查代码、量化图和小尺寸拼接，不读取验证集：

```bash
/opt/conda/envs/LED-ICCV23/bin/python tile_software_test/benchmark_qat_pairs.py \
  --check-only --device cpu
```

本轮仅准备命令并执行小尺寸 CPU 检查，未启动这七组的数据集评估。正式测试完成后，再联合 `results_qat_96` 和 `results_qat_extra7` 分析十组配置。

## 7. Git 跟踪范围

`.gitignore` 忽略本目录的 `results*/`、`pilot*/`、逐图断点目录、临时文件和生成的 `image_inventory.csv`；日志和 Python 缓存已有全局规则。脚本、说明文档、已整理的 Markdown 对比报告以及小型理论/数值检查 JSON 保留跟踪。

原先已入库的批量结果已取消 Git 跟踪，但本地文件仍保留；不会因 ignore 规则删除测试结果。克隆新工作区时这些原始结果不会随 Git 下载，需复跑脚本或单独备份复制；Markdown 中指向原始结果的链接也依赖本地生成文件。
