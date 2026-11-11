"""Collect completed SID evaluations without running inference or training."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quantization.run_baseline import EXPERIMENTS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-run', default='sid_w8a8_20261031')
    p.add_argument('--qat-baseline-run', default='qat_w8a8_v1')
    p.add_argument('--prefix', default='qat_ablation')
    p.add_argument('--output', type=Path, default=ROOT/'quantization/runs/boundary_ablation_summary.md')
    args = p.parse_args()
    lines = ['# 量化边界对照实验', '', 'LED 校正后的 RAW 指标；差值 = 当前组 − 对照组。缺失结果不视为零。', '',
             '|模型|组别|PSNR dB|SSIM|相对 FP32 dB|相对原始 QAT dB|', '|---|---|---:|---:|---:|---:|']
    for model in ('prelu', 'relu'):
        root = ROOT/'experiments'/EXPERIMENTS[model]/'evaluation'
        specs = [('FP32', args.baseline_run, 'fp32'), ('PTQ', args.baseline_run, 'ptq'),
                 ('QAT baseline', args.qat_baseline_run, 'qat')]
        specs += [(profile, args.prefix+'_'+profile, 'qat') for profile in ('opt1', 'opt2', 'opt12')]
        records = {}
        reference = None
        for label, run, mode in specs:
            path = root/run/mode/'metrics.json'
            if not path.exists():
                continue
            report = json.loads(path.read_text())
            meta = report['metadata']
            if meta['status'] != 'complete':
                continue
            provenance = (meta['checkpoint_sha256'], meta['dataset'])
            if reference is None:
                reference = provenance
            elif provenance != reference:
                raise ValueError('Checkpoint or dataset mismatch: '+str(path))
            records[label] = next(r for r in report['summary'] if str(r['ratio']) == 'all')
        for label, _, _ in specs:
            if label not in records:
                lines.append(f'|{model}|{label}|待完成|—|—|—|')
                continue
            row = records[label]
            deltas = [f"{row['psnr']-records[k]['psnr']:+.6f}" if k in records else '—' for k in ('FP32', 'QAT baseline')]
            lines.append(f"|{model}|{label}|{row['psnr']:.6f}|{row['ssim']:.8f}|{deltas[0]}|{deltas[1]}|")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(lines)+'\n')
    print(args.output)


if __name__ == '__main__':
    main()
