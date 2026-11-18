"""Compare the new LLStemNoAct checkpoint only: FP32 and three matched QAT runs."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from quantization.run_baseline import EXPERIMENTS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-run',default='sid_llstem_noact_20261117')
    p.add_argument('--a8-run',default='qat_llstem_noact_w8a8')
    p.add_argument('--a16-run',default='qat_llstem_noact_w8a16')
    p.add_argument('--mixed-run',default='qat_llstem_noact_phase_exp3_pres2d')
    args=p.parse_args()
    root=ROOT/'experiments'/EXPERIMENTS['relu_llstem_noact']/'evaluation'
    specs=[('FP32',args.baseline_run,'fp32',None,None),
           ('W8A8 QAT',args.a8_run,'qat',8,'none'),
           ('W8A16 QAT',args.a16_run,'qat',16,'none'),
           ('实验3：S2D前A8',args.mixed_run,'qat',16,'phase_exp3_pres2d')]
    records={}; provenance=None; train_ref=None
    for label,run,mode,bits,profile in specs:
        path=root/run/mode/'metrics.json'
        if not path.exists():continue
        data=json.loads(path.read_text());meta=data['metadata']
        if meta['status']!='complete':continue
        current=(meta['checkpoint_sha256'],meta['dataset'])
        if provenance is not None and current!=provenance:raise ValueError('Source/data mismatch: '+str(path))
        provenance=current
        pairs={(int(r['ratio']),int(r['index'])) for r in data['per_image']}
        expected={(int(r),i) for r,v in meta['dataset'].items() for i in range(v['count'])}
        if pairs!=expected or len(pairs)!=len(data['per_image']):raise ValueError('Incomplete/duplicate evaluation')
        if mode=='qat':
            t=meta['training'];s=t['settings']
            if s['model']!='relu_llstem_noact' or s['boundary_profile']!='opt1' or s.get('mixed_profile','none')!=profile or s['activation_bits']!=bits:
                raise ValueError('Unexpected QAT configuration: '+str(path))
            settings={k:v for k,v in s.items() if k not in ('run_name','activation_bits','mixed_profile')}
            current_train=(settings,t['training_data'],t['ptq_sha256'])
            if train_ref is not None and current_train!=train_ref:raise ValueError('Training provenance/settings differ')
            train_ref=current_train
            if profile!='none':
                policy=t['boundary_policy'];mapping=t['activation_bit_map']
                assert mapping['phase2_input_a8']==8 and mapping['conv_transpose2d_2']==16
                assert policy['aliases']['conv2d_3']=='phase2_input_a8'
        records[label]={str(r['ratio']):r for r in data['summary']}
    lines=['# ReLU LLStemNoAct：FP32 / W8A8 / W8A16 / 修改后实验3','','只比较该新 checkpoint。主指标为 LED 校正 RAW；差值 = 当前配置 − 原始 FP32。',
           'QAT 使用额外 20k 步训练/蒸馏，不能将与原始 FP32 的全部差异归因于量化本身。','',
           '|配置|倍率|PSNR dB|SSIM|PSNR−FP32|SSIM−FP32|','|---|---|---:|---:|---:|---:|']
    for label,run,mode,_,_ in specs:
        for ratio in ('100','250','300','all'):
            row=records.get(label,{}).get(ratio);fp=records.get('FP32',{}).get(ratio)
            if row is None:
                lines.append(f'|{label}|{ratio}|待完成|—|—|—|');continue
            dp=f"{row['psnr']-fp['psnr']:+.6f}" if fp else '—'
            ds=f"{row['ssim']-fp['ssim']:+.8f}" if fp else '—'
            lines.append(f"|{label}|{ratio}|{row['psnr']:.6f}|{row['ssim']:.8f}|{dp}|{ds}|")
    lines+=['','## 原始记录','']
    for label,run,mode,_,_ in specs:lines.append(f'- {label}：`{root/run/mode}`')
    out=ROOT/'quantization/runs/llstem_noact_summary.md';out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text('\n'.join(lines)+'\n');print(out)


if __name__=='__main__':main()
