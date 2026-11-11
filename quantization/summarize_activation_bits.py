"""Compare opt1 W8A8 and W8A16 with matched provenance and training settings."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quantization.run_baseline import EXPERIMENTS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fp32-run', default='sid_w8a8_20261031')
    p.add_argument('--a8-run', default='qat_ablation_opt1')
    p.add_argument('--a16-run', default='qat_ablation_opt1_w8a16')
    p.add_argument('--output', type=Path, default=ROOT/'quantization/runs/activation_bits_summary.md')
    args = p.parse_args()
    lines = ['# opt1 激活位宽对照', '', '指标采用 LED 校正后的 RAW；总体按 93 张图像平均。FP32 是 QAT 前的原始模型。', '',
             'PSNR/SSIM 差值列均为 W8A16 − W8A8。']
    for model in ('prelu', 'relu'):
        lines += ['', f'## {model}：A8 与 A16', '',
                  '|模型|倍率|W8A8 PSNR|W8A16 PSNR|PSNR 差值 dB|W8A8 SSIM|W8A16 SSIM|SSIM 差值|',
                  '|---|---|---:|---:|---:|---:|---:|---:|']
        root = ROOT/'experiments'/EXPERIMENTS[model]/'evaluation'
        records = {}
        for bits, run in ((8, args.a8_run), (16, args.a16_run)):
            path = root/run/'qat/metrics.json'
            if not path.exists():
                continue
            report = json.loads(path.read_text())
            if report['metadata']['status'] != 'complete':
                continue
            meta = report['metadata']; training = meta['training']; settings = training['settings']
            if settings.get('activation_bits', 8) != bits or settings['boundary_profile'] != 'opt1':
                raise ValueError('Unexpected precision/profile: '+str(path))
            records[bits] = report
        if 8 in records and 16 in records:
            a, b = [records[k]['metadata'] for k in (8, 16)]
            for key in ('checkpoint_sha256', 'dataset', 'protocol'):
                if a[key] != b[key]:
                    raise ValueError('Evaluation provenance differs: '+key)
            for key in ('ptq_sha256', 'training_data', 'boundary_policy'):
                if a['training'][key] != b['training'][key]:
                    raise ValueError('Training provenance differs: '+key)
            def comparable(meta):
                return {k:v for k,v in meta['training']['settings'].items() if k not in ('run_name', 'activation_bits')}
            if comparable(a) != comparable(b):
                raise ValueError('Training settings differ beyond activation bits')
        fp_path = root/args.fp32_run/'fp32/metrics.json'
        if fp_path.exists():
            fp = json.loads(fp_path.read_text())
            if fp['metadata']['status'] == 'complete':
                for report in records.values():
                    for key in ('checkpoint_sha256', 'dataset'):
                        if report['metadata'][key] != fp['metadata'][key]:
                            raise ValueError('FP32 provenance differs: '+key)
                records[32] = fp
        lookup = {bits: {str(r['ratio']):r for r in data['summary']} for bits,data in records.items()}
        for ratio in ('100', '250', '300', 'all'):
            a, b = [lookup.get(bits, {}).get(ratio) for bits in (8,16)]
            fmt = lambda row,key: f'{row[key]:.6f}' if row else '待完成'
            dp = f"{b['psnr']-a['psnr']:+.6f}" if a and b else '—'
            ds = f"{b['ssim']-a['ssim']:+.8f}" if a and b else '—'
            lines.append(f"|{model}|{ratio}|{fmt(a,'psnr')}|{fmt(b,'psnr')}|{dp}|{fmt(a,'ssim')}|{fmt(b,'ssim')}|{ds}|")
        lines += ['', f'## {model}：与 FP32 对照', '',
                  '|倍率|FP32 PSNR|W8A8−FP32 dB|W8A16−FP32 dB|FP32 SSIM|W8A8−FP32 SSIM|W8A16−FP32 SSIM|',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for ratio in ('100', '250', '300', 'all'):
            f, a, b = [lookup.get(bits, {}).get(ratio) for bits in (32,8,16)]
            def delta(row, key):
                return f'{row[key]-f[key]:+.8f}' if row and f else '—'
            lines.append(f"|{ratio}|{fmt(f,'psnr')}|{delta(a,'psnr')}|{delta(b,'psnr')}|{fmt(f,'ssim')}|{delta(a,'ssim')}|{delta(b,'ssim')}|")
    lines += ['', '说明：W8A16 经过额外 20k 步 QAT 和蒸馏。PSNR 高于原始 FP32 不代表量化本身提高精度；需与同预算 FP32 微调组比较。SSIM 与各倍率需分别判断。', '', '混合精度分析见 [MIXED_PRECISION_ANALYSIS.md](../MIXED_PRECISION_ANALYSIS.md)。']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(lines)+'\n')
    print(args.output)


if __name__ == '__main__':
    main()
