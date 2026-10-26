"""Full-frame validation of a scratch-trained UNet best checkpoint with conventional ISP."""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
import torch

from led.archs import build_network
from scripts.visualize_mobile_validation import render, metrics, panel, save


@torch.inference_mode()
def main(args):
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    experiment = Path(args.experiment)
    checkpoint = experiment / 'models/net_g_best_psnr.pth'
    payload = torch.load(checkpoint, map_location='cpu')
    selection = payload['selection']
    key = selection.get('evaluated_params', 'params')
    net = build_network(dict(type='UNetArch', inchannels=4, outchannels=4, channels=32))
    net.load_state_dict(payload[key], strict=True)
    net = net.cuda().eval()
    manifest = Path(args.manifest)
    records = json.loads(manifest.read_text())['pairs']
    out = experiment / args.folder
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(manifest, out/'validation_manifest.json')
    shutil.copyfile(__file__, out/'run_script_snapshot.py')
    wb = {}
    for record in records:
        wb.setdefault(record['group'], record['wb'])
    label = 'Scratch L1+SSIM' if experiment.name.endswith('L1_SSIM') else 'Scratch L1'
    labels = ['Noisy', label, 'Clean reference']
    config = dict(checkpoint=str(checkpoint.resolve()), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  best_selection=selection, parameter_key=key, manifest=str(manifest.resolve()),
                  inference='full-frame float32, no tiles, no TTA',
                  normalization='clip((DN-black)/(white-black)*ratio,0,1); clean ratio=1',
                  isp='same per-scene DNG WB -> 16-bit bilinear demosaic -> identity CCM -> sRGB OETF -> 8-bit PNG',
                  wb_by_group=wb, exposure_gain=1, auto_brightness=False, extra_denoising=False, sharpening=False,
                  color_limitation='No valid DNG CCM; colors are not calibrated.',
                  metrics='full-frame packed RAW clipped [0,1], border=2, Gaussian SSIM sigma=1.5 population covariance',
                  columns=labels)
    (out/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    print(json.dumps(config), flush=True)
    rows, sections = [], []
    start = time.time()
    for record in records:
        stem = Path(record['lq_path']).stem
        folder = out/stem
        folder.mkdir()
        black = np.asarray(record['black_level'], dtype=np.float32)[:,None,None]
        scale = np.asarray(record['white_level'], dtype=np.float32)[:,None,None]-black
        inp = np.clip((np.load(manifest.parent/record['lq_packed']).astype(np.float32)-black)/scale*record['ratio'],0,1)
        gt = np.clip((np.load(manifest.parent/record['gt_packed']).astype(np.float32)-black)/scale,0,1)
        pred = net(torch.from_numpy(inp[None]).cuda()).cpu().numpy()[0]
        assert pred.shape == gt.shape and np.isfinite(pred).all()
        np.save(folder/'denoised_packed_raw.npy', pred)
        images = [render(array, wb[record['group']]) for array in [inp,pred,gt]]
        for name, image in zip(['noisy','denoised','clean'], images):
            save(folder/(name+'.png'), image)
        save(folder/'comparison_preview.png', panel(images,labels,width=768))
        h,w = images[0].shape[:2]
        rois = []
        for j,cx in enumerate([w//4,w//2,3*w//4],1):
            y,x = h//2-256,cx-256
            rois.append(dict(x=x,y=y,width=512,height=512))
            save(folder/f'detail_{j}.png', panel([im[y:y+512,x:x+512] for im in images],labels))
        (folder/'rois.json').write_text(json.dumps(rois,indent=2)+'\n')
        row = dict(file=stem, group=record['group'],ratio=record['ratio'])
        for name,array in [('noisy',inp),('denoised',pred)]:
            row.update({name+'_'+k:v for k,v in metrics(array,gt).items()})
        rows.append(row)
        print(json.dumps(row),flush=True)
        sections.append(f'<h2>{stem}</h2><p>ratio={record["ratio"]:g}</p>'
            f'<a href="{stem}/comparison_preview.png"><img loading="lazy" src="{stem}/comparison_preview.png"></a>'
            f'<h3>{label} 单独显示</h3><a href="{stem}/denoised.png"><img loading="lazy" src="{stem}/denoised.png"></a>'
            + ''.join(f'<p><a href="{stem}/{n}.png">{l} 完整分辨率 PNG</a></p>' for n,l in zip(['noisy','denoised','clean'],labels))
            + ''.join(f'<a href="{stem}/detail_{j}.png"><img loading="lazy" src="{stem}/detail_{j}.png"></a>' for j in range(1,4)))
    with (out/'metrics.csv').open('w') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    means = {k:float(np.mean([row[k] for row in rows])) for k in rows[0] if k.endswith(('_psnr','_ssim'))}
    result = dict(status='complete',images=len(rows),best_selection=selection,mean_metrics=means,rows=rows,
                  elapsed_seconds=time.time()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated())
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    title = html.escape(experiment.name)
    table='<table><tr><th>模型</th><th>RAW PSNR</th><th>RAW SSIM</th></tr>'
    for name in ['noisy','denoised']:
        table+=f'<tr><td>{name}</td><td>{means[name+"_psnr"]:.4f}</td><td>{means[name+"_ssim"]:.5f}</td></tr>'
    table+='</table>'
    intro=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>{title} 验证集</title>
<style>body{{background:#181818;color:#eee;font:16px/1.6 sans-serif;margin:24px auto;max-width:1800px}}img{{width:100%;height:auto}}a{{color:#92caff}}td,th{{padding:8px 24px;text-align:left}}</style>
<h1>{title}：best 模型验证集可视化</h1>
<p>best 选择步数：{selection['iteration']}，训练时验证 patch PSNR：{selection['psnr']:.4f} dB。</p>
<p>2 个场景 × 3 个噪声档位，完整 4096×3072 推理。列顺序：Noisy / {label} / Clean。</p>
<p>统一传统 ISP：同场景固定 DNG 白平衡、16-bit 双线性去马赛克、单位色彩矩阵、sRGB 曲线；无自动提亮、额外降噪或锐化。DNG 无有效 CCM，颜色未经标定。</p>
<p>下表为完整 packed RAW 指标，与选择 best 时的局部 patch 指标不同。局部图为原图 512×512，点击图片可按原尺寸查看。</p>'''
    (out/'report.html').write_text(intro+table+''.join(sections)+'</html>')
    (out/'README.md').write_text(f'''# {experiment.name} best 验证集可视化

打开 report.html 查看三列对比、去噪图独立显示及局部细节。
每个图像子目录包含 noisy.png、denoised.png、clean.png（4096×3072），comparison_preview.png、detail_1/2/3.png、rois.json、denoised_packed_raw.npy（float32 RGBG，未应用 ISP，未裁剪输出值）。

best 模型步数：{selection['iteration']}，选择时 patch PSNR：{selection['psnr']:.4f}。
完整 RAW 平均 PSNR：{means['denoised_psnr']:.4f}，SSIM：{means['denoised_ssim']:.5f}。
传统 ISP、验证图像、归一化和 ROI 与先前微调可视化一致。颜色矩阵为单位矩阵，颜色未经标定。
config.json 保存 checkpoint 哈希和参数，metrics.csv / result.json 保存逐图指标。

复现（仓库根目录，LED-ICCV23 环境；输出子目录必须不存在）：
```bash
PYTHONPATH=. OMP_NUM_THREADS=4 python scripts/visualize_mobile_scratch.py --experiment {experiment} --folder visualization_val_best_rerun
```
''')
    print('COMPLETE',str(out),json.dumps(means),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',required=True)
    parser.add_argument('--manifest',default='../../Datasets/mine-mobile-image-dataset/prepared/val.json')
    parser.add_argument('--folder',default='visualization_val_best_20261024')
    main(parser.parse_args())
