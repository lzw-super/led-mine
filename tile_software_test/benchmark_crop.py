"""Full-resolution FP32 crop benchmark. No phase rounding, no image cropping."""
import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import yaml
from led.archs.sid_reference_arch import SIDReferenceDenoiser
from led.data.paired_raw_dataset import PairedRAWDataset
from led.metrics.psnr_ssim import calculate_ssim
from scripts.evaluate_crop_merge_sid import illuminance_correct, calculate_psnr

EXP = ROOT/'experiments/LED_Pretrain_CVPR20_ELD_LearningDWT_Haar_L3_RepNCB_W32_ReLU_LLStemNoAct'
CONFIGS = [(t,h) for t in (80,88,96) for h in (8,10,16)]

def boxes(height,width,t,h):
    core=t-2*h
    for y in range(0,height,core):
        y1=min(y+core,height)
        a,b=max(0,y-h),min(height,y1+h)
        for x in range(0,width,core):
            x1=min(x+core,width)
            c,d=max(0,x-h),min(width,x1+h)
            yield (y,y1,x,x1,a,b,c,d)

def tiled(net,image,t,h,batch):
    groups=defaultdict(list)
    for box in boxes(*image.shape[-2:],t,h):
        y,y1,x,x1,a,b,c,d=box
        groups[(b-a,d-c)].append(box)
    out=torch.empty_like(image)
    for group in groups.values():
        for i in range(0,len(group),batch):
            chunk=group[i:i+batch]
            patches=torch.cat([image[...,a:b,c:d] for _,_,_,_,a,b,c,d in chunk])
            values=net(patches)
            for value,(y,y1,x,x1,a,b,c,d) in zip(values,chunk):
                out[0,:,y:y1,x:x1]=value[:,y-a:y1-a,x-c:x1-c]
    return out

def timed(fn):
    torch.cuda.synchronize()
    start=time.perf_counter()
    out=fn()
    torch.cuda.synchronize()
    return out,(time.perf_counter()-start)*1000

def metrics(pred,gt):
    raw=calculate_psnr(pred,gt,2)
    corrected=illuminance_correct(pred,gt)
    psnr=calculate_psnr(corrected,gt,2)
    # Exact repository numpy SSIM, 4 packed channels; metrics excluded from timing.
    a=corrected[0].permute(1,2,0).cpu().numpy()*255.
    b=gt[0].permute(1,2,0).cpu().numpy()*255.
    return dict(psnr_db=psnr,ssim=float(calculate_ssim(a,b,2)),uncorrected_psnr_db=raw)

def append(path,row):
    exists=path.exists()
    with path.open('a') as f:
        writer=csv.DictWriter(f,fieldnames=list(row))
        if not exists: writer.writeheader()
        writer.writerow(row)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--tiles', type=int, nargs='+', default=[80,88,96])
    ap.add_argument('--halos', type=int, nargs='+', default=[8,10,16])
    ap.add_argument('--max-per-ratio',type=int,default=0)
    ap.add_argument('--batch-size',type=int,default=64)
    ap.add_argument('--serial-repeats',type=int,default=3)
    ap.add_argument('--output',type=Path,default=ROOT/'tile_software_test/results_fp32')
    args=ap.parse_args()
    run_configs=[(t,h) for t in args.tiles for h in args.halos]
    if args.batch_size < 1 or any(t <= 2*h or h < 0 for t,h in run_configs):
        ap.error('Require batch_size > 0 and tile > 2*halo >= 0')
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'per_image.csv').exists():
        raise RuntimeError('Use a new output directory to prevent duplicate measurements.')
    torch.set_num_threads(4)
    torch.manual_seed(2026)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    blob=(EXP/'models/net_g_255000.pth').read_bytes()
    opts=yaml.safe_load(next(EXP.glob('*.yaml')).read_text())['network_g'].copy()
    opts.pop('type')
    model=SIDReferenceDenoiser(**opts).eval()
    model.load_state_dict(torch.load(io.BytesIO(blob),map_location='cpu')['params'],strict=True)
    net=model.deploy().eval().cuda().float()
    del model
    meta=dict(checkpoint=str(EXP.relative_to(ROOT)/'models/net_g_255000.pth'),sha256=hashlib.sha256(blob).hexdigest(),gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),dtype='float32',amp=False,tf32=False,deploy=True,batch_size=args.batch_size,serial_repeats=args.serial_repeats,crop_border=2,illumination_correct=True,seed=2026,timing='CUDA-synchronized wall time, input/output on GPU; includes slicing, batching, forward and crop-copy; excludes IO, H2D, metrics and illumination correction',configs=run_configs)
    (args.output/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    options=yaml.safe_load((ROOT/'options/base/dataset/test/SID_SonyA7S2_val_split.yaml').read_text())['datasets']
    rng=random.Random(2026)
    for ratio in (100,250,300):
        opt=options[f'val_{ratio}'].copy()
        opt['phase']='val'
        opt['dataroot']=str(ROOT/opt['dataroot'])
        opt['data_pair_list']=str(ROOT/opt['data_pair_list'])
        dataset=PairedRAWDataset(opt)
        count=min(len(dataset),args.max_per_ratio) if args.max_per_ratio else len(dataset)
        for index in range(count):
            item=dataset[index]
            image=item['lq'][None].cuda()
            gt=item['gt'][None].cuda()
            with torch.inference_mode():
                if index==0:
                    for _ in range(2): net(image)
                    for t,h in run_configs: tiled(net,image,t,h,args.batch_size)
                full,full_ms=timed(lambda:net(image))
                fm=metrics(full,gt)
                append(args.output/'full_image.csv',dict(ratio=ratio,index=index,lq=item['lq_path'],milliseconds=full_ms,**fm))
                configs=run_configs.copy(); rng.shuffle(configs)
                for t,h in configs:
                    pred,ms=timed(lambda:tiled(net,image,t,h,args.batch_size))
                    m=metrics(pred,gt)
                    diff=pred-full
                    mse=diff.double().square().mean().item()
                    row=dict(ratio=ratio,index=index,lq=item['lq_path'],tile=t,halo=h,core=t-2*h,tile_count=len(list(boxes(*image.shape[-2:],t,h))),batch_size=args.batch_size,milliseconds=ms,full_milliseconds=full_ms,**m,full_psnr_db=fm['psnr_db'],full_ssim=fm['ssim'],psnr_loss_db=fm['psnr_db']-m['psnr_db'],ssim_loss=fm['ssim']-m['ssim'],uncorrected_psnr_loss_db=fm['uncorrected_psnr_db']-m['uncorrected_psnr_db'],mae_vs_full=diff.abs().mean().item(),max_abs_vs_full=diff.abs().max().item(),psnr_vs_full_db=-10*np.log10(mse))
                    append(args.output/'per_image.csv',row)
                # Serial timing: first image of each ratio, warmed once, three repeats.
                if index==0 and args.serial_repeats:
                    timing_configs=[(0,0)]+run_configs.copy()
                    rng.shuffle(timing_configs)
                    for t,h in timing_configs:
                        fn=(lambda:net(image)) if t==0 else (lambda t=t,h=h:tiled(net,image,t,h,1))
                        serial=fn(); torch.cuda.synchronize()
                        if t:
                            batch_out=tiled(net,image,t,h,args.batch_size)
                            err=(serial-batch_out).abs().max().item()
                            torch.testing.assert_close(serial,batch_out,rtol=1e-4,atol=2e-6)
                        else: err=0.
                        for rep in range(args.serial_repeats):
                            _,elapsed=timed(fn)
                            append(args.output/'serial_timing.csv',dict(ratio=ratio,index=index,tile=t,halo=h,repeat=rep,milliseconds=elapsed,max_abs_serial_vs_batch=err))
                print(f'ratio={ratio} image={index+1}/{count} full={full_ms:.2f}ms PSNR={fm["psnr_db"]:.4f}; {len(run_configs)} configurations complete',flush=True)
    print('DONE',flush=True)

if __name__=='__main__':
    main()
