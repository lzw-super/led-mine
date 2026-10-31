"""Evaluate an exported QAT model with the exact previous SID baseline metrics."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import torch
from quantization.run_baseline import metrics, write_reports, sha
from quantization.train_qat import build_mobile
from quantization.static_ptq import StaticPTQ
from scripts.evaluate_crop_merge_sid import build_sid_datasets
from led.models.raw_denoising_utils import IlluminanceCorrect


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--max-images-per-ratio',type=int,default=0)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if args.max_images_per_ratio<0:p.error('image limit must be nonnegative')
    torch.set_num_threads(4);cv2.setNumThreads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    artifact=torch.load(args.artifact,map_location='cpu');meta=artifact['metadata']
    if not meta.get('training_complete'):raise ValueError('Expected completed QAT export')
    mobile,options,source,exp=build_mobile(meta['model'],torch.device(args.device))
    if sha(source)!=meta['checkpoint_sha256']:raise ValueError('Original FP32 source changed')
    runner=StaticPTQ(mobile);runner.load_artifact(args.artifact)
    run_name=meta['settings']['run_name']+('_smoke' if args.max_images_per_ratio else '')
    out=exp/'evaluation'/run_name/'qat'
    artifact_hash=sha(args.artifact)
    datasets=build_sid_datasets(ROOT/'options/base/dataset/test/SID_SonyA7S2_val_split.yaml',[100,250,300],None,'npz')
    manifests={str(r):{'list':str(d.opt['data_pair_list']),'sha256':sha(d.opt['data_pair_list']),'count':len(d)} for r,d in datasets}
    metadata=dict(mode='qat',status='running',artifact=str(args.artifact),artifact_sha256=artifact_hash,
                  checkpoint_sha256=meta['checkpoint_sha256'],dataset=manifests,training=meta,
                  max_images_per_ratio=args.max_images_per_ratio,
                  protocol='Same LED SID baseline: full frame, RAW, border=2, LED IlluminanceCorrect, PSNR MSE+1e-8, official float64 SSIM; no AMP/TF32')
    rows=[];report=out/'metrics.json'
    if report.exists():
        previous=json.loads(report.read_text())
        if not args.resume:raise FileExistsError('Evaluation exists; pass --resume or use another training run')
        if previous['metadata']['artifact_sha256']!=artifact_hash or previous['metadata']['dataset']!=manifests:
            raise ValueError('Resume artifact/data mismatch')
        if previous['metadata']['max_images_per_ratio']!=args.max_images_per_ratio:
            raise ValueError('Resume image limit differs')
        rows=previous['per_image']
    completed={(int(r['ratio']),int(r['index'])) for r in rows}
    with torch.inference_mode():
        for ratio,dataset in datasets:
            limit=min(len(dataset),args.max_images_per_ratio) if args.max_images_per_ratio else len(dataset)
            for index in range(limit):
                if (ratio,index) in completed:continue
                start=time.time();item=dataset[index]
                x=item['lq'][None].to(args.device);gt=item['gt'][None].to(args.device)
                pred=runner.run(x)
                if not torch.isfinite(pred).all():raise FloatingPointError('Nonfinite QAT output')
                raw_psnr,raw_ssim=metrics(pred,gt)
                psnr,ssim=metrics(IlluminanceCorrect()(pred,gt),gt)
                rows.append(dict(ratio=ratio,index=index,lq_path=item['lq_path'],gt_path=item['gt_path'],psnr=psnr,ssim=ssim,
                                 psnr_uncorrected=raw_psnr,ssim_uncorrected=raw_ssim,seconds=time.time()-start))
                write_reports(out,rows,metadata)
                print(meta['model'],'QAT',ratio,index+1,'/',limit,psnr,ssim,flush=True)
    metadata['status']='smoke_complete' if args.max_images_per_ratio else 'complete'
    summary=write_reports(out,rows,metadata)
    if not args.max_images_per_ratio:
        expected={(r,i) for r,d in datasets for i in range(len(d))}
        if {(r['ratio'],r['index']) for r in rows}!=expected:raise ValueError('Incomplete evaluation')
        comparisons=[]
        baseline_root=exp/'evaluation'/meta['settings']['ptq_run']
        for name in ('fp32','ptq'):
            original=json.loads((baseline_root/name/'metrics.json').read_text())
            if original['metadata']['status']!='complete' or original['metadata']['checkpoint_sha256']!=meta['checkpoint_sha256'] or original['metadata']['dataset']!=manifests:
                raise ValueError('Baseline provenance differs from QAT evaluation')
            lookup={r['ratio']:r for r in original['summary']}
            for row in summary:
                comparisons.append(dict(baseline=name,ratio=row['ratio'],**{
                    k+'_gain':row[k]-lookup[row['ratio']][k] for k in ('psnr','ssim','psnr_uncorrected','ssim_uncorrected')}))
        (out/'comparison.json').write_text(json.dumps(dict(gain_definition='QAT minus baseline; positive is better',summary=comparisons),indent=2))
        lines=['# QAT 对比：增益 = QAT − baseline','','|基线|倍率|PSNR 增益 dB|SSIM 增益|','|---|---|---:|---:|']
        for r in comparisons:lines.append(f"|{r['baseline']}|{r['ratio']}|{r['psnr_gain']:.6f}|{r['ssim_gain']:.8f}|")
        (out/'comparison.md').write_text('\n'.join(lines)+'\n')
    print('Results:',out,flush=True)

if __name__=='__main__':main()
