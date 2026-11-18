"""Aggregate benchmark CSVs without rounding away signed precision changes."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics

P=Path(__file__).resolve().parent
R=P/'results_fp32'
rows=list(csv.DictReader((R/'per_image.csv').open()))
full=list(csv.DictReader((R/'full_image.csv').open()))
serial=list(csv.DictReader((R/'serial_timing.csv').open()))
assert len(full)==93 and len(rows)==837 and len(serial)==90
assert len({(r['ratio'],r['index'],r['tile'],r['halo']) for r in rows})==837

def mean(group,key): return statistics.mean(float(r[key]) for r in group)

def aggregate(group):
    return {k:mean(group,k) for k in ['psnr_db','ssim','psnr_loss_db','ssim_loss','uncorrected_psnr_loss_db','milliseconds','full_milliseconds','mae_vs_full','psnr_vs_full_db']} | {'max_abs_vs_full':max(float(r['max_abs_vs_full']) for r in group),'p50_ms':statistics.median(float(r['milliseconds']) for r in group)}

summary={}
for ratio in ['all','100','250','300']:
    part=rows if ratio=='all' else [r for r in rows if r['ratio']==ratio]
    summary[ratio]={}
    for t in [80,88,96]:
        for h in [8,10,16]:
            group=[r for r in part if int(r['tile'])==t and int(r['halo'])==h]
            summary[ratio][f'{t}/{h}']=dict(n=len(group),**aggregate(group))
(R/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
flat = [dict(ratio=ratio, tile=int(key.split('/')[0]), halo=int(key.split('/')[1]), **value) for ratio, group in summary.items() for key, value in group.items()]
with (R/'summary.csv').open('w') as f:
    writer=csv.DictWriter(f, fieldnames=list(flat[0]))
    writer.writeheader(); writer.writerows(flat)
meta=json.loads((R/'metadata.json').read_text())
lines=['# FP32 小 tile / crop 实测结果','', '日期：2026-11-18。9 组组合，完整 93 对 SID Sony 验证图像（100×:29、250×:29、300×:35），每幅 packed 输入 4×1424×2128。没有裁掉原图边缘。','', '## 配置与计时口径','',f"- 权重：`{meta['checkpoint']}`；SHA256 `{meta['sha256']}`。",f"- GPU：{meta['gpu']}；PyTorch {meta['torch']}；CUDA {meta['cuda']}；cuDNN {meta['cudnn']}。",'- 同一份 RepNCB 部署融合模型用于整图与 tile；参数/激活 FP32，AMP、TF32 均关闭，eval/inference_mode；cuDNN deterministic=True、benchmark=False。','- T 是含 halo 的输入窗口最大尺寸，单边 h，core/步长 C=T−2h。crop 丢弃内部 halo，只复制 core；真实图像边缘裁短输入、由网络自行处理 padding。','- 当前三个 T 均小于 2×56=112，无法在这些输入窗口内同时容纳理论 halo=56 与正面积 core；本次测试是允许精度损失的近似分块方案，而不是严格等价分块。','- halo=10 保持用户指定值，不向 8 对齐：窗口起点相位变化，且部分边界块有内部补齐，其损失包含上下文不足与 Haar/S2D 相位变化，不能解释为单纯 halo 大小效应。h=8/16 的起点与 core 均为 8 倍数，但仍远小于理论 56。','- 全集精度与 GPU 批处理计时使用最多 64 个同形状 tile 一批；边缘按实际 shape 分组，不将边缘补成最大 T。所有输入输出在 GPU；同步 wall time 包含切片、组 batch、前向和 crop 写回，排除文件读取、CPU→GPU、指标及照度校正。每 ratio 首图预热全部配置，逐图随机配置顺序。','- 串行计时另选每个 ratio 的第一幅完整图（共 3 幅），每配置预热一次、计时三次，表中为 9 次测量的中位数。不是全集串行平均，也不是 FPGA 时延。并验证这三图每个配置串行/批量输出 allclose（rtol=1e-4,atol=2e-6）。','- 精度：RAW 4 通道；沿用实验全图 illumination correction（拼接后执行，使用 GT 拟合单个标量）与 crop_border=2；PSNR 使用 double MSE 的标准 10log10(1/MSE)，SSIM 用仓库 numpy 实现。PSNR 未加仓库 tensor 指标的 1e-8 epsilon，故绝对值不应与训练日志直接逐位比较。另提供未校正 PSNR 损失，避免全图亮度拟合掩盖变化。','- “损失”=整图−tile，正数表示精度降低，负数表示该项略有提升。对每幅图指标取算术平均，全体按 93 图加权，不先将三个 ratio 等权平均。相对整图误差在未经校正、未裁边的原始输出上计算。','', '## 全集精度与时间','',f"整图基线：PSNR **{mean(full,'psnr_db'):.4f} dB**，SSIM **{mean(full,'ssim'):.6f}**；93 图平均 **{mean(full,'milliseconds'):.2f} ms/图**。",'', '| T | h | core | tile 数/图 | PSNR dB | PSNR 损失 dB | SSIM | SSIM 损失 | 批量64 平均 ms | 相对整图耗时 | 串行1 中位 ms |','|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for t in [80,88,96]:
    for h in [8,10,16]:
        a=summary['all'][f'{t}/{h}']
        group=[r for r in serial if int(r['tile'])==t and int(r['halo'])==h]
        median=statistics.median(float(r['milliseconds']) for r in group)
        count=next(r['tile_count'] for r in rows if int(r['tile'])==t and int(r['halo'])==h)
        lines.append(f"| {t} | {h} | {t-2*h} | {count} | {a['psnr_db']:.4f} | {a['psnr_loss_db']:.4f} | {a['ssim']:.6f} | {a['ssim_loss']:.6f} | {a['milliseconds']:.2f} | {a['milliseconds']/mean(full,'milliseconds'):.2f}× | {median:.2f} |")
base=statistics.median(float(r['milliseconds']) for r in serial if r['tile']=='0')
lines += ['', f'串行测试组内整图基线（相同 3 图×3 次）中位数为 **{base:.2f} ms**。批量时间与串行时间的样本/统计量不同，应分别参照对应基线。','', '## 分 ratio 精度损失','', '| T/h | 100× PSNR损失 dB | 250× PSNR损失 dB | 300× PSNR损失 dB | 未校正全集PSNR损失 dB | 原始输出 MAE vs full | 原始输出最大误差 |','|---|---:|---:|---:|---:|---:|---:|']
for t in [80,88,96]:
    for h in [8,10,16]:
        key=f'{t}/{h}'; a=summary['all'][key]
        losses=' | '.join(f"{summary[r][key]['psnr_loss_db']:.4f}" for r in ['100','250','300'])
        lines.append(f"| {key} | {losses} | {a['uncorrected_psnr_loss_db']:.4f} | {a['mae_vs_full']:.6g} | {a['max_abs_vs_full']:.6g} |")
lines += ['', '## 复现与结果文件','', '```bash','/opt/conda/envs/LED-ICCV23/bin/python tile_software_test/benchmark_crop.py','python tile_software_test/summarize_benchmark.py','```','','已有结果目录会拒绝覆盖；复跑请通过 `--output` 指定新目录（汇总脚本默认读取 results_fp32）。', '', '- [全集逐图/逐配置数据](results_fp32/per_image.csv)：837 行。','- [整图逐图基线](results_fp32/full_image.csv)：93 行。','- [串行原始计时](results_fp32/serial_timing.csv)：90 行。','- [汇总 CSV](results_fp32/summary.csv)、[汇总 JSON](results_fp32/summary.json)、[环境与模型元数据](results_fp32/metadata.json)。','- pilot_fp32 为每 ratio 一图的冒烟测试，未混入正式汇总。','', '本实验的 GPU 批量输入与整帧输出常驻显存，用于软件精度/吞吐比较，不作为片上 RAM 峰值或 FPGA 性能结论。小 tile 会增加网络调用、重算和拼接开销，串行结果尤其受 Python/CUDA launch 开销影响。']
best=min(summary['all'],key=lambda k:summary['all'][k]['psnr_loss_db'])
fast=min(summary['all'],key=lambda k:summary['all'][k]['milliseconds'])
best_group=[r for r in rows if f"{r['tile']}/{r['halo']}"==best]
worst=max(best_group,key=lambda r:float(r['psnr_loss_db']))
conclusions=['## 结果解读','',f"- 平均精度优先：**{best}**，PSNR 平均损失 {summary['all'][best]['psnr_loss_db']:.4f} dB。该配置单图损失中位数 {statistics.median(float(r['psnr_loss_db']) for r in best_group):.4f} dB，最大损失 **{float(worst['psnr_loss_db']):.4f} dB**，发生于 {worst['ratio']}×、`{Path(worst['lq']).name}`；平均值不能替代最坏情况约束。",f"- GPU 批量时间最短：**{fast}**，平均 {summary['all'][fast]['milliseconds']:.2f} ms。但 halo=10 的相位不对齐使其不适合作为严格对齐硬件实现的基准，且其 GPU 性能优势不能外推到 FPGA。",'- 若要求相位对齐并优先串行软件时间，选择 96/8；若优先精度，选择 96/16。此结论仅限当前九个候选、当前模型与验证集。','- 所有九组在此次 GPU 实测中都慢于整图推理。没有任何一组满足理论严格等价条件；是否接受上述损失，应按应用的平均和最坏图像质量预算决定。','']
idx=lines.index('## 分 ratio 精度损失')
lines[idx:idx]=conclusions
(P/'FP32_CROP_RESULTS.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[lines.index('## 全集精度与时间'):lines.index('## 分 ratio 精度损失')]))
