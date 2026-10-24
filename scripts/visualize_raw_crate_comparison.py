"""Independent single-frame RAW-crate denoising, shared traditional ISP."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
import torch

from led.archs import build_network
from scripts.bin_image_process import pack_bayer
from scripts.prepare_mobile_unet_checkpoint import prepare
from scripts.visualize_mobile_validation import render, panel, save


@torch.inference_mode()
def main(args):
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    source = Path(args.source)
    files = sorted(source.glob('*.raw'))
    if not files or any(f.stat().st_size != 4096*3072*2 for f in files):
        raise ValueError('Expected tightly packed 4096x3072 uint16 RAW frames.')
    previous = json.loads(Path(args.reference_config).read_text())
    wb = previous['white_balance']
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    prepare(args.pretrained,out/'pretrained_generalized_unet.pth')
    models={}
    for name,path in [('pretrained',out/'pretrained_generalized_unet.pth'),('finetuned',args.finetuned)]:
        net=build_network(dict(type='UNetArch',inchannels=4,outchannels=4,channels=32))
        net.load_state_dict(torch.load(path,map_location='cpu')['params'],strict=True)
        models[name]=net.cuda().eval()
    labels=['Noisy (gain 64)','LED pretrained','Mobile L1']
    config=dict(source=str(source.resolve()),frames=len(files),
        checkpoints={name:dict(path=str(Path(path).resolve()),sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
            for name,path in [('pretrained',args.pretrained),('finetuned',args.finetuned)]},
        sensor_assumptions=previous['sensor_assumptions'], input_gain=64,
        parameter_reference=args.reference_config, white_balance=wb,
        normalization='clip((DN-64)/(1023-64)*64,0,1)',
        pretrained_conversion='mean of 5 RepNR camera alignments via generalize(), then deploy()',
        inference='each frame independently, full resolution FP32; no temporal inputs, tiling, or TTA',
        isp='fixed first-frame gray-world WB from previous RAW-crate run; 16-bit bilinear demosaic; identity CCM; sRGB OETF; 8-bit PNG',
        auto_brightness=False,extra_denoising=False,sharpening=False,ground_truth=False,
        columns=labels)
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    sections=[];stats=[];start=time.time()
    for i,path in enumerate(files):
        t=time.time()
        folder=out/path.stem;folder.mkdir()
        mosaic=np.fromfile(path,dtype='<u2').reshape(3072,4096).astype(np.float32)
        inp=np.clip((pack_bayer(mosaic,'RGGB')-64)/959*64,0,1)
        arrays={'noisy':inp}
        for name,net in models.items():
            arrays[name]=net(torch.from_numpy(inp[None]).cuda()).cpu().numpy()[0]
            if arrays[name].shape!=inp.shape or not np.isfinite(arrays[name]).all():
                raise ValueError(f'Invalid prediction: {path.name}/{name}')
        images=[render(array,wb) for array in arrays.values()]
        for name,im in zip(arrays,images):save(folder/(name+'.png'),im)
        save(folder/'comparison_preview.png',panel(images,labels,width=768))
        rois=[]
        for j,cx in enumerate([1024,2048,3072]):
            x,y=cx-256,1280
            rois.append(dict(x=x,y=y,width=512,height=512))
            save(folder/f'detail_{j+1}.png',panel([im[y:y+512,x:x+512] for im in images],labels))
        (folder/'rois.json').write_text(json.dumps(rois,indent=2)+'\n')
        stat=dict(file=path.name,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            finite=True,input_saturated_fraction=float((inp>=1).mean()),
            output_ranges={name:[float(array.min()),float(array.max())] for name,array in arrays.items()},
            seconds=time.time()-t)
        stats.append(stat)
        sections.append(f'<h2>{path.stem}</h2><a href="{path.stem}/comparison_preview.png"><img loading="lazy" src="{path.stem}/comparison_preview.png"></a>'
          + '<p>'+' | '.join(f'<a href="{path.stem}/{name}.png">{label} 全分辨率</a>' for name,label in zip(arrays,labels))+'</p>'
          + ''.join(f'<a href="{path.stem}/detail_{j}.png"><img loading="lazy" src="{path.stem}/detail_{j}.png"></a>' for j in range(1,4)))
        print(f'{i+1}/{len(files)} {path.name}: saved, {stat["seconds"]:.1f}s',flush=True)
    result=dict(status='complete',frames=len(files),elapsed_seconds=time.time()-start,
                peak_gpu_bytes=torch.cuda.max_memory_allocated(),statistics=stats)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>RAW-crate 单帧去噪比较</title>
<style>body{font:16px/1.6 sans-serif;background:#181818;color:#eee;max-width:1800px;margin:24px auto}img{width:100%;height:auto}a{color:#92caff}</style>
<h1>RAW-crate：预训练模型 vs 手机 L1 微调模型</h1>
<p>18 帧分别进行独立单帧推理，不使用相邻帧。每行从左至右：含噪输入（增益64） / LED 预训练 / Mobile L1 微调。</p>
<p>统一传统 ISP：固定白平衡 → 16-bit 双线性去马赛克 → 单位色彩矩阵 → sRGB 曲线，无自动亮度、额外降噪或锐化。沿用此前该数据的 RGGB、黑电平64、白电平1023 参数；无头 RAW 自身不含元数据。</p>
<p>完整图像为4096×3072；局部图为原图512×512裁剪。点击图片后以100%查看细节。没有clean参考，不计算PSNR/SSIM，也不能仅凭更平滑判定细节恢复更好。</p>
<p><a href="config.json">参数与权重记录</a> | <a href="result.json">逐帧处理记录</a></p>'''
    (out/'report.html').write_text(html+''.join(sections)+'</html>')
    shutil.copyfile(__file__,out/'run_script_snapshot.py')
    (out/'README.md').write_text('''# RAW-crate 单帧去噪可视化

打开 report.html 浏览18帧的独立单帧比较。
每个帧目录：noisy.png / pretrained.png / finetuned.png 为全分辨率输出；comparison_preview.png 为三列预览；detail_1/2/3.png 为三个固定位置的原尺寸局部对比。参数和checkpoint SHA256见config.json。

没有clean，不计算有参考指标。输出统一使用传统ISP、固定输入增益64和固定白平衡。预训练模型使用五个RepNR相机分支平均后融合的普通UNet，与本次微调初始化方式一致。

复现（仓库根目录、LED-ICCV23环境，输出目录须不存在）：

```bash
PYTHONPATH=. OMP_NUM_THREADS=4 python scripts/visualize_raw_crate_comparison.py --output experiments/Mobile_UNet_L1/test_RAW_crate_rerun
```
''')
    print('COMPLETE',out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',default='../RViDeformer/mydata_test/RAW-crate/RAW')
    p.add_argument('--reference-config',default='experiments/LED_Pretrain_CVPR20_ELD_Setting_archived_20260726_230625/visualization_RAW-crate_20261021/config.json')
    p.add_argument('--pretrained',default='experiments/LED_Pretrain_CVPR20_ELD_Setting_archived_20260726_230625/models/net_g_latest.pth')
    p.add_argument('--finetuned',default='experiments/Mobile_UNet_L1/models/net_g_latest.pth')
    p.add_argument('--output',default='experiments/Mobile_UNet_L1/test_RAW_crate_comparison_20261024')
    main(p.parse_args())
