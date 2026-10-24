"""Compare pretrained/generalized LED and fine-tuned UNet with an identical conventional ISP."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity

from led.archs import build_network
from scripts.bin_image_process import depack_bayer
from scripts.prepare_mobile_unet_checkpoint import prepare


def render(packed, wb):
    # Demosaic in 16-bit linear space, then apply the standard sRGB OETF.
    mosaic = depack_bayer(np.clip(packed, 0, 1) * np.asarray(wb)[:, None, None], 'RGGB')
    mosaic = np.rint(np.clip(mosaic, 0, 1) * 65535).astype(np.uint16)
    bgr = cv2.cvtColor(mosaic, cv2.COLOR_BayerRGGB2BGR).astype(np.float32) / 65535
    bgr = np.where(bgr <= .0031308, 12.92 * bgr, 1.055 * bgr ** (1/2.4) - .055)
    return np.rint(np.clip(bgr, 0, 1) * 255).astype(np.uint8)


def save(path, image):
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise IOError(path)


def panel(images, labels, width=None):
    panels = []
    for im, label in zip(images, labels):
        if width is not None:
            im = cv2.resize(im, (width, round(im.shape[0]*width/im.shape[1])), interpolation=cv2.INTER_AREA)
        im = cv2.copyMakeBorder(im, 36, 0, 0, 0, cv2.BORDER_CONSTANT, value=(24,24,24))
        cv2.putText(im, label, (12,25), cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 1, cv2.LINE_AA)
        panels.append(im)
    return np.concatenate(panels, axis=1)


def metrics(pred, gt):
    # Normalized packed RAW, 2-pixel border excluded. Shared clipping for all outputs.
    pred, gt = np.clip(pred[:,2:-2,2:-2],0,1), gt[:,2:-2,2:-2]
    mse = np.mean((pred.astype(np.float64)-gt)**2)
    ssim = np.mean([structural_similarity(x,y,data_range=1,gaussian_weights=True,
                    sigma=1.5,use_sample_covariance=False) for x,y in zip(pred,gt)])
    return dict(psnr=float(-10*np.log10(mse)), ssim=float(ssim))


@torch.inference_mode()
def main(args):
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    manifest = Path(args.manifest)
    records = json.loads(manifest.read_text())['pairs']
    (out/'validation_manifest.json').write_text(manifest.read_text())
    prepare(args.pretrained, out/'pretrained_generalized_unet.pth')
    models = {}
    for name, path in [('pretrained',out/'pretrained_generalized_unet.pth'),('finetuned',args.finetuned)]:
        net = build_network(dict(type='UNetArch',inchannels=4,outchannels=4,channels=32))
        net.load_state_dict(torch.load(path,map_location='cpu')['params'],strict=True)
        models[name] = net.cuda().eval()
    # Same WB within each scene, derived once from its first noisy DNG metadata.
    wb_by_group = {}
    for record in records:
        wb_by_group.setdefault(record['group'],record['wb'])
    config = dict(pretrained=args.pretrained, finetuned=args.finetuned,
        checkpoint_sha256={name:hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for name,path in [('pretrained',args.pretrained),('finetuned',args.finetuned)]},
        pretrained_conversion='strict load; generalize mean of 5 camera alignments; deploy to plain UNet',
        inference='full frame float32, no tiles, no test-time augmentation',
        normalization='clip((DN-black)/(white-black)*ratio,0,1); clean ratio=1',
        isp='shared scene DNG WB -> 16-bit bilinear demosaic -> identity CCM -> sRGB OETF -> 8-bit PNG',
        wb_by_group=wb_by_group, exposure_gain=1,
        no_auto_brightness=True, no_extra_denoising=True, no_sharpening=True,
        metrics='full-frame packed RAW clipped [0,1], border=2; Gaussian SSIM sigma=1.5, population covariance',
        color_limitation='DNG rawpy CCM is zero; identity CCM used for all outputs, not calibrated color',
        columns=['Noisy','LED pretrained','Mobile L1','Clean reference'])
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    rows, sections = [], []
    started = time.time()
    for i,record in enumerate(records):
        stem = Path(record['lq_path']).stem
        folder = out/stem
        folder.mkdir()
        bl = np.asarray(record['black_level'],dtype=np.float32)[:,None,None]
        scale = np.asarray(record['white_level'],dtype=np.float32)[:,None,None]-bl
        gt = np.clip((np.load(manifest.parent/record['gt_packed']).astype(np.float32)-bl)/scale,0,1)
        inp = np.clip((np.load(manifest.parent/record['lq_packed']).astype(np.float32)-bl)/scale*record['ratio'],0,1)
        arrays = dict(noisy=inp)
        for name,net in models.items():
            prediction = net(torch.from_numpy(inp[None]).cuda()).cpu().numpy()[0]
            if prediction.shape != inp.shape or not np.isfinite(prediction).all():
                raise ValueError(f'Invalid prediction: {stem} {name}')
            arrays[name] = prediction
            np.save(folder/(name+'_packed_raw.npy'),prediction)
        arrays['clean'] = gt
        images = [render(arrays[name],wb_by_group[record['group']]) for name in arrays]
        for name,image in zip(arrays,images):
            save(folder/(name+'.png'),image)
        save(folder/'comparison_preview.png',panel(images,config['columns'],width=768))
        # Three fixed 512px native-resolution ROIs: left, centre, right mid-frame.
        h,w = images[0].shape[:2]
        rois=[]
        for j,cx in enumerate([w//4,w//2,3*w//4]):
            y,x = h//2-256,cx-256
            rois.append(dict(x=x,y=y,width=512,height=512))
            crops=[image[y:y+512,x:x+512] for image in images]
            save(folder/f'detail_{j+1}.png',panel(crops,config['columns']))
        (folder/'rois.json').write_text(json.dumps(rois,indent=2)+'\n')
        entry=dict(file=stem,group=record['group'],ratio=record['ratio'])
        for name in ['noisy','pretrained','finetuned']:
            values=metrics(arrays[name],gt)
            entry.update({name+'_'+key:value for key,value in values.items()})
        rows.append(entry)
        print(json.dumps(entry),flush=True)
        sections.append(f'<h2>{stem}</h2><p>ratio={record["ratio"]:g}</p>'
          f'<a href="{stem}/comparison_preview.png"><img src="{stem}/comparison_preview.png"></a>'
          + ''.join(f'<p><a href="{stem}/{name}.png">{label} 完整分辨率 PNG</a></p>'
                    for name,label in zip(arrays,config['columns']))
          + ''.join(f'<a href="{stem}/detail_{j}.png"><img loading="lazy" src="{stem}/detail_{j}.png"></a>' for j in range(1,4)))
    with (out/'metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    means={key:float(np.mean([row[key] for row in rows])) for key in rows[0] if key.endswith(('_psnr','_ssim'))}
    result=dict(status='complete',images=len(rows),mean_metrics=means,rows=rows,
                elapsed_seconds=time.time()-started,peak_gpu_bytes=torch.cuda.max_memory_allocated())
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    table='<table><tr><th>模型</th><th>RAW PSNR</th><th>RAW SSIM</th></tr>'
    for name in ['noisy','pretrained','finetuned']:
        table+=f'<tr><td>{name}</td><td>{means[name+"_psnr"]:.4f}</td><td>{means[name+"_ssim"]:.5f}</td></tr>'
    table+='</table>'
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Mobile 验证集去噪对比</title>
<style>body{background:#181818;color:#eee;font:16px/1.6 sans-serif;margin:24px auto;max-width:1800px}img{width:100%;height:auto}a{color:#92caff}td,th{padding:8px 24px;text-align:left}</style>
<h1>手机实采验证集：预训练 vs L1 微调</h1><p>2 组场景 × 3 个噪声档位，完整 4096×3072 输出。每行顺序：Noisy / LED pretrained / Mobile L1 / Clean。</p>
<p>统一传统 ISP：同场景固定 DNG 白平衡、16-bit 双线性去马赛克、单位色彩矩阵、sRGB 曲线。无自动提亮、额外降噪或锐化。DNG 缺少可用 CCM，显示颜色未经标定。</p>
<p>下表为完整 packed RAW 的指标，与训练时 5 个固定 patch 的验证指标不同。局部图为原图 512×512 区域，打开图片可按 100% 查看。</p>'''
    (out/'report.html').write_text(html+table+''.join(sections)+'</html>')
    print('COMPLETE',out,json.dumps(means),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',default='../../Datasets/mine-mobile-image-dataset/prepared/val.json')
    parser.add_argument('--pretrained',default='experiments/LED_Pretrain_CVPR20_ELD_Setting_archived_20260726_230625/models/net_g_latest.pth')
    parser.add_argument('--finetuned',default='experiments/Mobile_UNet_L1/models/net_g_latest.pth')
    parser.add_argument('--output',default='experiments/Mobile_UNet_L1/visualization_val_comparison_20261024')
    main(parser.parse_args())
