"""Evaluate five final checkpoints and two LearningDWT static PTQ baselines."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
import cv2
import numpy as np
import torch
from led.archs import build_network
from led.archs.sid_reference.mobile_haar import MobileHaarRepNCB
from led.data import build_dataset
from led.metrics.psnr_ssim import calculate_ssim, calculate_ssim_pt
from led.models.raw_denoising_utils import IlluminanceCorrect
from led.utils.options import yaml_load
from copy import deepcopy
from scripts.evaluate_crop_merge_sid import load_led_network, build_sid_datasets
from scripts.prepare_led_deploy_checkpoint import strip_thop_profile_buffers
from quantization.static_ptq import StaticPTQ

EXPERIMENTS = {
 'prelu': 'LED_Pretrain_CVPR20_ELD_LearningDWT_Haar_L3_RepNCB_W32',
 'relu': 'LED_Pretrain_CVPR20_ELD_LearningDWT_Haar_L3_RepNCB_W32_ReLU',
 'led': 'LED_Pretrain_CVPR20_ELD_Setting_archived_20260726_230625',
 'mrlfn': 'LED_Pretrain_CVPR20_ELD_MRLFN_Paper_S2D_K4_N4_D32',
 'splitter': 'LED_Pretrain_CVPR20_ELD_SplitterNet'}


def load_options(path):
    def merge(a,b):
        result=deepcopy(b)
        for k,v in a.items():
            result[k]=merge(v,result[k]) if isinstance(v,dict) and isinstance(result.get(k),dict) else deepcopy(v)
        return result
    options=yaml_load(str(path));bases=options.pop('base',[]);result={}
    for base in bases:
        result=merge(result,load_options(ROOT/base))
    return merge(options,result)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(pred, gt):
    # Same LED batch PSNR epsilon; official numpy SSIM shares the float64
    # Gaussian-window definition with calculate_ssim_pt. No 8-bit rounding.
    p = pred[0].detach().cpu().numpy().transpose(1, 2, 0).astype(np.float64)
    g = gt[0].detach().cpu().numpy().transpose(1, 2, 0).astype(np.float64)
    mse = np.mean((p[2:-2, 2:-2] - g[2:-2, 2:-2]) ** 2)
    return float(10 * np.log10(1 / (mse + 1e-8))), float(calculate_ssim(p * 255, g * 255, 2))


def write_reports(directory, rows, metadata):
    directory.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for row in rows:
        groups[str(row['ratio'])].append(row)
        groups['all'].append(row)
    summaries = []
    for key, group in groups.items():
        summaries.append(dict(ratio=key, images=len(group), **{
            k: float(np.mean([r[k] for r in group]))
            for k in ('psnr', 'ssim', 'psnr_uncorrected', 'ssim_uncorrected')}))
    for name, records in [('per_image', rows), ('summary', summaries)]:
        if records:
            with (directory / (name + '.csv')).open('w') as f:
                writer = csv.DictWriter(f, fieldnames=list(records[0]))
                writer.writeheader(); writer.writerows(records)
    (directory / 'metrics.json').write_text(json.dumps(dict(metadata=metadata, summary=summaries, per_image=rows), indent=2))
    lines = ['# SID Sony RAW 评估', '', f"状态：{metadata['status']}；模式：{metadata['mode']}", '',
             '主指标：LED 亮度校正后；uncorrected：模型原输出，不进行亮度校正或输出裁剪。', '',
             '|倍率|张数|PSNR (dB)|SSIM|PSNR uncorrected|SSIM uncorrected|', '|---|---:|---:|---:|---:|---:|']
    for r in summaries:
        lines.append(f"|{r['ratio']}|{r['images']}|{r['psnr']:.6f}|{r['ssim']:.8f}|{r['psnr_uncorrected']:.6f}|{r['ssim_uncorrected']:.8f}|")
    lines += ['', '详细评估协议、权重与数据清单哈希见 metrics.json。all 为逐图指标平均，并非三个倍率等权平均。']
    (directory / 'README.md').write_text('\n'.join(lines) + '\n')
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=list(EXPERIMENTS))
    parser.add_argument('--modes', nargs='+', default=['fp32', 'ptq'])
    parser.add_argument('--run-name', default='sid_w8a8_20261031')
    parser.add_argument('--calibration-images', type=int, default=32)
    parser.add_argument('--crops-per-image', type=int, default=8)
    parser.add_argument('--max-images', type=int, default=0, help='Smoke only; 0 means all test images')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4); cv2.setNumThreads(4)
    torch.manual_seed(2026); np.random.seed(2026); random.seed(2026)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda:0')
    datasets = build_sid_datasets(ROOT/'options/base/dataset/test/SID_SonyA7S2_val_split.yaml', [100, 250, 300], None, 'npz')
    manifests = {str(r): {'list': str(d.opt['data_pair_list']), 'sha256': sha(d.opt['data_pair_list']), 'count': len(d)} for r,d in datasets}
    test_scenes = {Path(p).name.split('_')[0] for _, d in datasets for p in d.lq_paths}
    artifact_root = ROOT/'quantization/artifacts'/args.run_name
    artifact_root.mkdir(parents=True, exist_ok=True)
    # Verify the SSIM implementation used for speed matches LED's tensor metric.
    a, b = torch.rand(1, 4, 32, 48), torch.rand(1, 4, 32, 48)
    s1 = metrics(a,b)[1]; s2 = float(calculate_ssim_pt(a,b,2))
    assert abs(s1-s2) < 1e-10, (s1,s2)
    for short in args.models:
        exp = ROOT/'experiments'/EXPERIMENTS[short]
        ckpt = exp/'models/net_g_latest.pth'
        cfg = next(exp.glob('*.yaml'))
        options = load_options(cfg)
        resolved = artifact_root/(short+'_resolved.yaml')
        # yaml_load does not resolve base inheritance; save fully resolved config.
        import yaml
        resolved.write_text(yaml.safe_dump(json.loads(json.dumps(options)), sort_keys=False))
        network, load_info = load_led_network(ckpt, resolved, 'params', device)
        metadata_base = dict(checkpoint=str(ckpt.relative_to(ROOT)), checkpoint_sha256=sha(ckpt),
            config=str(cfg.relative_to(ROOT)), config_sha256=sha(cfg), resolved_config=str(resolved.relative_to(ROOT)),
            load=load_info, torch_version=torch.__version__, dataset=manifests,
            protocol='LED SID valid_100/250/300; full-frame packed RAW; crop_border=2; no AMP/TF32; mean of per-image metrics',
            psnr='10*log10(1/(MSE+1e-8)), float64, LED calculate_psnr_pt convention',
            ssim='official LED calculate_ssim float64, no uint8 rounding; verified against calculate_ssim_pt',
            illumination='official IlluminanceCorrect, clamps prediction to [0,1] then GT-based scalar; no post-correction clamp',
            max_images_per_ratio=args.max_images)
        for mode in args.modes:
            if mode == 'ptq' and short not in ('prelu','relu'):
                continue
            output_dir = exp/'evaluation'/args.run_name/mode
            report_file = output_dir/'metrics.json'
            if args.resume and report_file.exists():
                old = json.loads(report_file.read_text())
                if old['metadata']['checkpoint_sha256'] != metadata_base['checkpoint_sha256']:
                    raise ValueError('Checkpoint changed; choose a new run name')
                if old['metadata']['status'] == 'complete':
                    print('SKIP COMPLETE',short,mode,flush=True);continue
            metadata = dict(metadata_base, mode=mode, status='running')
            ptq = None
            if mode == 'ptq':
                reference = build_network(options['network_g']).eval()
                state,_ = strip_thop_profile_buffers(torch.load(ckpt,map_location='cpu')['params'])
                reference.load_state_dict(state,strict=True)
                mobile = MobileHaarRepNCB(reference).to(device).eval()
                with torch.inference_mode():
                    probe = torch.randn(1,4,64,80,device=device)*.1
                    torch.testing.assert_close(mobile(probe),network(probe),atol=1e-5,rtol=1e-4)
                ptq = StaticPTQ(mobile)
                artifact = artifact_root/(short+'_w8a8.pth')
                if args.resume and artifact.exists():
                    metadata['quantization'] = ptq.load_artifact(artifact)
                    if metadata['quantization']['checkpoint_sha256'] != metadata['checkpoint_sha256']:
                        raise ValueError('PTQ source changed')
                else:
                    cal_options = dict(datasets[0][1].opt)
                    cal_options.update(name='SID_train_calibration',data_pair_list='datasets/txtfiles/SID/SonyA7S2/train.txt')
                    train = build_dataset(cal_options)
                    assert not test_scenes.intersection(Path(p).name.split('_')[0] for p in train.lq_paths)
                    rng = np.random.RandomState(2026)
                    # Round-robin across exposure ratios, then unique frames.
                    buckets = defaultdict(list)
                    for i, ratio in enumerate(train.ratios): buckets[ratio].append(i)
                    for bucket in buckets.values(): rng.shuffle(bucket)
                    selected=[]
                    while len(selected)<args.calibration_images:
                        for ratio in sorted(buckets):
                            if buckets[ratio] and len(selected)<args.calibration_images: selected.append(buckets[ratio].pop())
                        if not any(buckets.values()) and len(selected)<args.calibration_images: raise ValueError('Not enough calibration frames')
                    cal_manifest=[]
                    with torch.inference_mode():
                        for i in selected:
                            item=train[i];x=item['lq'];h,w=x.shape[-2:]
                            positions=[]
                            for j in range(args.crops_per_image):
                                y=int(rng.randint(0,(h-256)//8+1))*8;z=int(rng.randint(0,(w-256)//8+1))*8
                                ptq.run(x[:,y:y+256,z:z+256].unsqueeze(0).to(device))
                                positions.append([y,z,256,256])
                            cal_manifest.append(dict(index=i,lq_path=item['lq_path'],ratio=float(item['ratio']),crops=positions))
                            print('CALIBRATE',short,len(cal_manifest),'/',len(selected),flush=True)
                    ptq.freeze()
                    metadata['quantization']=dict(checkpoint_sha256=metadata['checkpoint_sha256'],weights='INT8 symmetric per-output-channel absmax',
                        activations='INT8 symmetric per-tensor static absmax', qmin=-127,qmax=127,zero_point=0,
                        rounding='nearest ties to even',bias='INT32, scale=input_scale*weight_scale',
                        execution='FP32 Q/DQ accuracy simulation, not integer-only hardware or speed benchmark',
                        float_exceptions=['PReLU slopes','fixed Haar/space-depth constants; their data activations are quantized'],
                        calibration_train_list_sha256=sha(cal_options['data_pair_list']),seed=2026,calibration=cal_manifest)
                    ptq.export(artifact,metadata['quantization'])
                    (artifact_root/(short+'_graph.txt')).write_text(str(ptq.module.graph))
                infer=ptq.run
            else: infer=network
            rows=[]
            if args.resume and report_file.exists():rows=json.loads(report_file.read_text())['per_image']
            completed={(int(r['ratio']),int(r['index'])) for r in rows}
            with torch.inference_mode():
                for ratio,dataset in datasets:
                    limit=min(len(dataset),args.max_images) if args.max_images else len(dataset)
                    for i in range(limit):
                        if (ratio,i) in completed:continue
                        start=time.time();item=dataset[i]
                        x=item['lq'].unsqueeze(0).to(device);gt=item['gt'].unsqueeze(0).to(device)
                        out=infer(x)
                        if not torch.isfinite(out).all():raise FloatingPointError('Nonfinite prediction')
                        raw_psnr,raw_ssim=metrics(out,gt)
                        corrected=IlluminanceCorrect()(out,gt)
                        psnr,ssim=metrics(corrected,gt)
                        row=dict(ratio=ratio,index=i,lq_path=item['lq_path'],gt_path=item['gt_path'],
                            psnr=psnr,ssim=ssim,psnr_uncorrected=raw_psnr,ssim_uncorrected=raw_ssim,seconds=time.time()-start)
                        rows.append(row)
                        write_reports(output_dir,rows,metadata)
                        print('EVAL',short,mode,ratio,i+1,'/',limit,f'PSNR={psnr:.5f} SSIM={ssim:.6f} seconds={row["seconds"]:.1f}',flush=True)
                        del x,gt,out,corrected
            metadata['status']='complete' if not args.max_images else 'smoke_complete'
            if ptq:
                (output_dir/'quantization_scales.json').write_text(json.dumps(ptq.statistics(),indent=2))
            write_reports(output_dir,rows,metadata)
            del ptq
        del network
        torch.cuda.empty_cache()

if __name__=='__main__':main()
