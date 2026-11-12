"""Collect phase mixed precision evaluations with provenance checks."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quantization.run_baseline import EXPERIMENTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for i in range(1, 7):
        parser.add_argument(f'--exp{i}-run', default=f'qat_phase_exp{i}')
    parser.add_argument('--model', choices=['prelu','relu','all'], default='all')
    args = parser.parse_args()
    lines = ['# Phase 混合激活精度对照', '', 'LED 校正 RAW；差值 = 当前配置 − 全 A16 opt1。缺失结果显示待完成。', '']
    for model in (('prelu', 'relu') if args.model == 'all' else (args.model,)):
        root = ROOT/'experiments'/EXPERIMENTS[model]/'evaluation'
        specs = [('FP32','sid_w8a8_20261031','fp32'),('A8 opt1','qat_ablation_opt1','qat'),
                 ('A16 opt1','qat_ablation_opt1_w8a16','qat')]
        specs += [(f'phase_exp{i}', getattr(args, f'exp{i}_run'), 'qat') for i in range(1,7)]
        lines += [f'{model} 实验目录：' + '、'.join(run for _,run,_ in specs[3:]), '']
        records = {}; reference = None; training_reference = None
        for label, run, mode in specs:
            path = root/run/mode/'metrics.json'
            if not path.exists():continue
            data = json.loads(path.read_text()); meta = data['metadata']
            if meta['status'] != 'complete':continue
            provenance = (meta['checkpoint_sha256'],meta['dataset'])
            if reference is not None and reference != provenance:raise ValueError('Source/dataset mismatch: '+str(path))
            reference = provenance
            if mode == 'qat':
                train = meta['training']; settings = train['settings']
                ignored = ('run_name','activation_bits','mixed_profile')
                comparable = ({k:v for k,v in settings.items() if k not in ignored},train['training_data'],train['ptq_sha256'])
                if training_reference is not None and comparable != training_reference:raise ValueError('Training settings differ: '+str(path))
                training_reference = comparable
                expected = label if label.startswith('phase_') else 'none'
                if settings.get('mixed_profile','none') != expected:raise ValueError('Unexpected mixed profile')
            records[label] = {str(r['ratio']):r for r in data['summary']}
        lines += [f'## {model}', '', '|配置|倍率|PSNR dB|SSIM|相对全 A16 PSNR|相对全 A16 SSIM|', '|---|---|---:|---:|---:|---:|']
        for label,_,_ in specs:
            for ratio in ('100','250','300','all'):
                row = records.get(label,{}).get(ratio); base = records.get('A16 opt1',{}).get(ratio)
                if row is None:
                    lines.append(f'|{label}|{ratio}|待完成|—|—|—|');continue
                dp = f"{row['psnr']-base['psnr']:+.6f}" if base else '—'
                ds = f"{row['ssim']-base['ssim']:+.8f}" if base else '—'
                lines.append(f"|{label}|{ratio}|{row['psnr']:.6f}|{row['ssim']:.8f}|{dp}|{ds}|")
        lines.append('')
    out = ROOT/'quantization/runs/phase_mixed_summary.md'
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text('\n'.join(lines)+'\n');print(out)


if __name__ == '__main__':main()
