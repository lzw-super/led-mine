"""Create a verified cross-model report only after every full evaluation finishes."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EXPERIMENTS={
 'prelu':'LED_Pretrain_CVPR20_ELD_LearningDWT_Haar_L3_RepNCB_W32',
 'relu':'LED_Pretrain_CVPR20_ELD_LearningDWT_Haar_L3_RepNCB_W32_ReLU',
 'led':'LED_Pretrain_CVPR20_ELD_Setting_archived_20260726_230625',
 'mrlfn':'LED_Pretrain_CVPR20_ELD_MRLFN_Paper_S2D_K4_N4_D32',
 'splitter':'LED_Pretrain_CVPR20_ELD_SplitterNet'}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-name',default='sid_w8a8_20261031');a=parser.parse_args()
    reports={};manifest=None;table=[]
    for model,exp in EXPERIMENTS.items():
        for mode in (['fp32','ptq'] if model in ('prelu','relu') else ['fp32']):
            p=ROOT/'experiments'/exp/'evaluation'/a.run_name/mode/'metrics.json'
            r=json.loads(p.read_text());m=r['metadata']
            assert m['status']=='complete' and m['max_images_per_ratio']==0,(model,mode,m['status'])
            assert len(r['per_image'])==sum(d['count'] for d in m['dataset'].values())
            assert len({(v['ratio'],v['index']) for v in r['per_image']})==len(r['per_image'])
            expected={(int(ratio),i) for ratio,d in m['dataset'].items() for i in range(d['count'])}
            assert {(v['ratio'],v['index']) for v in r['per_image']}==expected
            assert all(math.isfinite(v[k]) for v in r['per_image'] for k in ('psnr','ssim','psnr_uncorrected','ssim_uncorrected'))
            if manifest is None:manifest=m['dataset']
            assert manifest==m['dataset']
            reports[model,mode]=r
            for s in r['summary']:table.append(dict(model=model,mode=mode,**s))
    lines=['# 五模型 FP32 与 LearningDWT W8A8 PTQ baseline', '',
           '日期：2026-10-31。五个模型均使用 net_g_latest.pth；默认 LED SID Sony 测试清单 100/250/300，共 93 对整图。', '',
           'RAW 域，crop_border=2，主指标采用 LED 原有 GT 辅助亮度校正。每行是逐图指标平均。详情与未校正指标保存在各实验目录。', '',
           '|模型|模式|倍率|张数|PSNR (dB)|SSIM|', '|---|---|---|---:|---:|---:|']
    for r in table:
        lines.append(f"|{r['model']}|{r['mode']}|{r['ratio']}|{r['images']}|{r['psnr']:.6f}|{r['ssim']:.8f}|")
    lines+=['','## 量化损失','', '损失 = FP32 − PTQ；正数表示下降。','',
            '|模型|倍率|PSNR 损失 (dB)|SSIM 损失|PSNR 损失（未校正）|SSIM 损失（未校正）|',
            '|---|---|---:|---:|---:|---:|']
    losses=[]
    for model in ('prelu','relu'):
        f=reports[model,'fp32'];q=reports[model,'ptq']
        assert f['metadata']['checkpoint_sha256']==q['metadata']['checkpoint_sha256']
        lookup={x['ratio']:x for x in f['summary']};local=[]
        for r in q['summary']:
            orig=lookup[r['ratio']]
            loss=dict(model=model,ratio=r['ratio'],**{k+'_loss':orig[k]-r[k] for k in ('psnr','ssim','psnr_uncorrected','ssim_uncorrected')})
            losses.append(loss);local.append(loss)
            lines.append(f"|{model}|{r['ratio']}|{loss['psnr_loss']:.6f}|{loss['ssim_loss']:.8f}|{loss['psnr_uncorrected_loss']:.6f}|{loss['ssim_uncorrected_loss']:.8f}|")
        dst=ROOT/'experiments'/EXPERIMENTS[model]/'evaluation'/a.run_name/'ptq_vs_fp32.json'
        dst.write_text(json.dumps({'checkpoint_sha256':f['metadata']['checkpoint_sha256'],'loss_definition':'FP32 minus PTQ','summary':local},indent=2))
    lines += ['', '## 静态范围诊断', '', '|模型|RAW 输入步长|RAW 对称上界|最终输出步长|', '|---|---:|---:|---:|']
    for model in ('prelu','relu'):
        path=ROOT/'experiments'/EXPERIMENTS[model]/'evaluation'/a.run_name/'ptq/quantization_scales.json'
        scales=json.loads(path.read_text())
        lines.append(f"|{model}|{scales['raw']['scale']:.8f}|{scales['raw']['absmax']:.8f}|{scales['conv_transpose2d_3']['scale']:.8f}|")
    lines += ['', 'RAW 输入使用 zero_clip=false 且上限为 1；absmax 校准保留负向异常值会拉大对称范围，减少正常亮度区间的有效码字。以上是观察到的范围，尚未通过逐节点消融证明其对总损失的贡献。此轮未引入分位数裁剪、混合精度或重训练。']
    lines+=['','## 解释范围','',
        'PTQ 使用 32 张训练图 × 8 个 256×256 crop 校准；训练场景与这 93 对测试图场景不重叠。权重逐输出通道，激活逐张量，均为静态 absmax 对称均匀量化，码域 [-127,127]，zero-point=0。', '',
        '此结果为 FP32 核执行 Q/DQ 的量化精度仿真；卷积 bias 量化到 INT32。PReLU 斜率与固定 Haar/重排常量保留 FP32，数据接口激活量化至 A8。尚非全整数 FPGA 实现，不提供 INT8 加速比。', '',
        '没有使用混合精度、QAT、AdaRound、测试集调参或基于测试结果反复选择校准范围。统一量化 RAW 输入及 Haar/残差/拼接接口可能导致明显误差；该结果用于后续改进的严格 baseline。', '',
        '## 文件位置','']
    for model,exp in EXPERIMENTS.items():
        lines.append(f'- {model}：[实验评估目录](../experiments/{exp}/evaluation/{a.run_name}/)')
    out=ROOT/'quantization';(out/'BASELINE_RESULTS.md').write_text('\n'.join(lines)+'\n')
    artifacts=out/'artifacts'/a.run_name
    for name,records in [('comparison',table),('quantization_loss',losses)]:
        with (artifacts/(name+'.csv')).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    source={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [out/'run_baseline.py',out/'static_ptq.py',out/'summarize_baseline.py',ROOT/'led/metrics/psnr_ssim.py',ROOT/'led/models/raw_denoising_utils.py']}
    (artifacts/'source_manifest.json').write_text(json.dumps(source,indent=2))
    print('\n'.join(lines))

if __name__=='__main__':main()
