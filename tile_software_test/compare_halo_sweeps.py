"""Compare the original nine configurations with the four h=20/24 additions."""
import csv
import json
import math
from pathlib import Path
import statistics as st

P=Path(__file__).resolve().parent
RUNS=[('original',P/'results_fp32',9),('h20_h24',P/'results_fp32_h20_h24',4)]
rows=[]; serial=[]; baselines={}; metas={}
for name,directory,nconfig in RUNS:
    def read(file):
        return list(csv.DictReader((directory/file).open()))
    part=read('per_image.csv'); timing=read('serial_timing.csv'); full=read('full_image.csv')
    meta=json.loads((directory/'metadata.json').read_text())
    assert len(part)==93*nconfig and len(full)==93 and len(timing)==9*(nconfig+1)
    assert len({(r['ratio'],r['index'],r['tile'],r['halo']) for r in part})==len(part)
    for r in part:
        for k in ['psnr_db','ssim','milliseconds','psnr_loss_db','ssim_loss']:
            assert math.isfinite(float(r[k])),(name,r,k)
    rows.extend(dict(run=name,**r) for r in part)
    serial.extend(dict(run=name,**r) for r in timing)
    baselines[name]=full; metas[name]=meta
for key in ['sha256','gpu','torch','cuda','cudnn','dtype','amp','tf32','deploy','batch_size','serial_repeats','crop_border','illumination_correct','seed','timing']:
    assert metas['original'][key]==metas['h20_h24'][key],key
old={(r['ratio'],r['index']):r for r in baselines['original']}
new={(r['ratio'],r['index']):r for r in baselines['h20_h24']}
assert old.keys()==new.keys()
for key in old:
    assert old[key]['lq']==new[key]['lq']
base_delta={metric:max(abs(float(old[k][metric])-float(new[k][metric])) for k in old) for metric in ['psnr_db','ssim','uncorrected_psnr_db']}
assert base_delta['psnr_db']<1e-5 and base_delta['ssim']<1e-7,base_delta

configs=sorted({(int(r['tile']),int(r['halo'])) for r in rows})
assert len(configs)==13 and len(rows)==1209

def mean(group,key): return st.mean(float(r[key]) for r in group)

summary=[]
for t,h in configs:
    g=[r for r in rows if int(r['tile'])==t and int(r['halo'])==h]
    tm=[r for r in serial if int(r['tile'])==t and int(r['halo'])==h]
    assert len(g)==93 and len(tm)==9
    run=g[0]['run']
    summary.append(dict(run=run,tile=t,halo=h,core=t-2*h,tile_count=int(g[0]['tile_count']),n=93,
        psnr_db=mean(g,'psnr_db'),psnr_loss_db=mean(g,'psnr_loss_db'),ssim=mean(g,'ssim'),ssim_loss=mean(g,'ssim_loss'),
        uncorrected_psnr_loss_db=mean(g,'uncorrected_psnr_loss_db'),batch_mean_ms=mean(g,'milliseconds'),
        serial_median_ms=st.median(float(r['milliseconds']) for r in tm),
        batch_time_vs_own_full=mean(g,'milliseconds')/mean(baselines[run],'milliseconds'),
        worst_psnr_loss_db=max(float(r['psnr_loss_db']) for r in g),
        **{f'ratio_{ratio}_psnr_loss_db':mean([r for r in g if r['ratio']==ratio],'psnr_loss_db') for ratio in ['100','250','300']}))
OUT=P/'results_fp32_combined'; OUT.mkdir(exist_ok=True)
for file,data in [('summary.csv',summary),('per_image.csv',rows)]:
    with (OUT/file).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
(OUT/'summary.json').write_text(json.dumps(dict(baseline_max_abs_difference=base_delta,results=summary),indent=2)+'\n')

lines=['# FP32 crop：13 组 tile/halo 联合比较','',
'沿用同一实验的 `net_g_255000.pth` 部署融合模型、RTX 4090、FP32（AMP/TF32 关闭），完整 93 对 SID Sony 验证图像。此前九组保留原始测量，本次新增 88/20、88/24、96/20、96/24 四组；未重跑旧九组。', '',
'## 一致性与计时口径','',
'- T 包含两侧 halo，core=T−2h；边缘裁短输入，输出完整 1424×2128，不裁原图、不自动对齐参数。',
'- 新旧运行的权重哈希、GPU、软件版本、精度和指标设置已逐项核对；93 个输入路径逐一匹配，整图基线逐图复测。',
f"- 整图 PSNR 最大跨轮差异 {base_delta['psnr_db']:.3g} dB；SSIM 最大差异 {base_delta['ssim']:.3g}。",
'- 精度沿用实验的全图 illumination correction（拼接后对 GT 拟合单个标量），RAW 四通道、crop_border=2。正的损失表示比同轮整图差；未校正损失及分 ratio 结果另列。',
'- 批量时间：每次最多 64 个同形状 tile，93 图平均；GPU 输入输出常驻，CUDA 同步 wall time，包括组 batch、模型前向、crop 写回，不含 IO/H2D、指标和照度校正。',
'- 串行时间：每个 ratio 首图各预热一次、测三次，合计九次的中位数；新旧使用相同三张图。已对新增四组核验串行/批量输出 allclose（rtol=1e-4、atol=2e-6）。',
'- 两轮计时并非同一次连续测量；时钟、系统调度等可能产生波动。表中保留各轮整图时间，用各自基线作耗时倍率参考；不能将小幅差异全部归因于参数。GPU 软件时间不是 FPGA 时延。',
'- h=24 保持三级 Haar 的 8 像素相位；h=20 时 core 虽是 8 倍数，内部输入起点却为 4 mod 8，造成第三级 Haar 相位变化。h=10 也不满足 8 对齐；这些不对齐组混合了上下文和相位因素。此处 h 均为偶数，Refiner S2D2 相位仍对齐。',
'- 这些输入 tile 均小于 112，无法容纳理论 h=56 与正面积 core；所有候选都是近似分块方案。', '',
'| 轮次 | 图数 | 整图 PSNR dB | 整图 SSIM | 整图平均 ms | 串行测时样本整图中位 ms |',
'|---|---:|---:|---:|---:|---:|']
for name,_,_ in RUNS:
    f=baselines[name]; tm=[r for r in serial if r['run']==name and r['tile']=='0']
    lines.append(f"| {name} | 93 | {mean(f,'psnr_db'):.6f} | {mean(f,'ssim'):.8f} | {mean(f,'milliseconds'):.2f} | {st.median(float(r['milliseconds']) for r in tm):.2f} |")
lines += ['', '## 全部 13 组','', '| T | h | core | tile数 | PSNR dB | PSNR损失 dB | SSIM损失 | 批量64平均 ms | 串行1中位 ms | 批量/本轮整图 |','|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in summary:
    lines.append(f"| {r['tile']} | {r['halo']} | {r['core']} | {r['tile_count']} | {r['psnr_db']:.4f} | {r['psnr_loss_db']:.4f} | {r['ssim_loss']:.6f} | {r['batch_mean_ms']:.2f} | {r['serial_median_ms']:.2f} | {r['batch_time_vs_own_full']:.2f}× |")
lines += ['', '## 新增组相对同 tile 的 h=16','', '| T/h | PSNR损失减少 dB（正值为改善） | SSIM损失减少 | 批量耗时比 | 串行耗时比 |', '|---|---:|---:|---:|---:|']
for r in summary:
    if r['halo'] not in [20,24]: continue
    oldrow=next(x for x in summary if x['tile']==r['tile'] and x['halo']==16)
    lines.append(f"| {r['tile']}/{r['halo']} | {oldrow['psnr_loss_db']-r['psnr_loss_db']:.4f} | {oldrow['ssim_loss']-r['ssim_loss']:.6f} | {r['batch_mean_ms']/oldrow['batch_mean_ms']:.2f}× | {r['serial_median_ms']/oldrow['serial_median_ms']:.2f}× |")
best=min(summary,key=lambda r:r['psnr_loss_db'])
lines += ['',f"13 组中平均 PSNR 损失最小的是 **{best['tile']}/{best['halo']}**：平均下降 **{best['psnr_loss_db']:.4f} dB**，SSIM 下降 **{best['ssim_loss']:.6f}**；但单图最大 PSNR 损失仍为 **{best['worst_psnr_loss_db']:.4f} dB**。最终选择须同时考虑平均/最差图像精度与重算成本。",'',
'## 分 ratio 和最坏图像','', '| T/h | 100× PSNR损失 dB | 250× PSNR损失 dB | 300× PSNR损失 dB | 未校正平均损失 dB | 最差单图损失 dB |','|---|---:|---:|---:|---:|---:|']
for r in summary:
    lines.append(f"| {r['tile']}/{r['halo']} | {r['ratio_100_psnr_loss_db']:.4f} | {r['ratio_250_psnr_loss_db']:.4f} | {r['ratio_300_psnr_loss_db']:.4f} | {r['uncorrected_psnr_loss_db']:.4f} | {r['worst_psnr_loss_db']:.4f} |")
lines += ['', '## 原始数据与复现', '',
'- [联合汇总 CSV](results_fp32_combined/summary.csv)、[JSON](results_fp32_combined/summary.json)、[1209 条逐图记录](results_fp32_combined/per_image.csv)。',
'- 新增组：[372 条逐图记录](results_fp32_h20_h24/per_image.csv)、[整图基线](results_fp32_h20_h24/full_image.csv)、[45 条串行计时](results_fp32_h20_h24/serial_timing.csv)、[元数据](results_fp32_h20_h24/metadata.json)。',
'- 原九组：[原报告](FP32_CROP_RESULTS.md)，原始文件保留在 `results_fp32/`。', '',
'```bash', '/opt/conda/envs/LED-ICCV23/bin/python tile_software_test/benchmark_crop.py --tiles 88 96 --halos 20 24 --output tile_software_test/results_fp32_h20_h24', 'python tile_software_test/compare_halo_sweeps.py','```','',
'已有结果不会覆盖；若重跑需指定新的 output 目录并同步调整汇总脚本 RUNS。']
(P/'FP32_CROP_COMPARISON_13.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[lines.index('## 全部 13 组'):lines.index('## 分 ratio 和最坏图像')]))
