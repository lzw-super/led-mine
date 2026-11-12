"""Train LearningDWT W8A8/W8A16 QAT. Test set is used only by explicit post-training evaluation."""
import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data._utils.collate import default_collate
from led.archs import build_network
from led.archs.sid_reference.mobile_haar import MobileHaarRepNCB
from led.data import build_dataset
from led.data.noise_utils import VirtualNoisyPairGenerator
from quantization.run_baseline import EXPERIMENTS, load_options, sha
from quantization.qat_model import QATModel
from quantization.boundaries import PROFILES
from quantization.mixed_precision import MIXED_PROFILES
from scripts.prepare_led_deploy_checkpoint import strip_thop_profile_buffers


def atomic_save(data, path):
    temporary = path.with_suffix('.tmp')
    torch.save(data, temporary)
    temporary.replace(path)


def build_mobile(model, device):
    exp = ROOT/'experiments'/EXPERIMENTS[model]
    cfg = next(exp.glob('*.yaml'))
    options = load_options(cfg)
    source = exp/'models/net_g_latest.pth'
    net = build_network(options['network_g']).eval()
    state, _ = strip_thop_profile_buffers(torch.load(source, map_location='cpu')['params'])
    net.load_state_dict(state, strict=True)
    return MobileHaarRepNCB(net).to(device).eval(), options, source, exp


def make_training_data(options, exp, kind, crop, device):
    if kind == 'synthetic':
        opt = copy.deepcopy(options['datasets']['train'])
        opt.update(phase='train', crop_size=crop, load_in_mem=False)
        dataset = build_dataset(opt)
        train_list = ROOT/'datasets/txtfiles/SID/SonyA7S2/train.txt'
        allowed = {Path(line.split()[1]).name.split('_')[0] for line in train_list.read_text().splitlines()}
        candidates = len(dataset.data_paths)
        dataset.data_paths = sorted(p for p in dataset.data_paths if Path(p).name.split('_')[0] in allowed)
        removed = candidates - len(dataset.data_paths)
        config = copy.deepcopy(options['noise_g']); config.pop('type')
        noise = VirtualNoisyPairGenerator(config, device=device).to(device).eval()
        noise_path = exp/'noise_g.pth'
        noise.load_state_dict(torch.load(noise_path, map_location=device), strict=True)
        paths = dataset.data_paths
        extra = {'noise_checkpoint': str(noise_path.relative_to(ROOT)), 'noise_sha256': sha(noise_path),
                 'train_pair_list_sha256': sha(train_list), 'excluded_nontraining_patches': removed}
    else:
        opt = copy.deepcopy(options['datasets']['val_100'])
        opt.update(name='QAT_SID_train', phase='train', data_pair_list='datasets/txtfiles/SID/SonyA7S2/train.txt',
                   crop_size=crop, use_hflip=True, use_rot=True, load_in_mem=False)
        dataset = build_dataset(opt); noise = None; paths = dataset.lq_paths
        extra = {'train_pair_list_sha256': sha(opt['data_pair_list'])}
    test_scenes = set()
    for key, value in options['datasets'].items():
        if key.startswith('val'):
            test_scenes.update(Path(line.split()[0]).name.split('_')[0] for line in Path(value['data_pair_list']).read_text().splitlines())
    scenes = {Path(p).name.split('_')[0] for p in paths}
    if scenes & test_scenes:
        raise ValueError('Training and test scenes overlap: '+str(scenes & test_scenes))
    if not len(dataset):
        raise ValueError('Training dataset is empty')
    extra.update(kind=kind, samples=len(dataset), scene_count=len(scenes), test_scene_overlap=0,
                 path_list_sha256=hashlib.sha256('\n'.join(paths).encode()).hexdigest())
    return dataset, noise, extra


def train_batch(dataset, noise, batch_size, device):
    indices = torch.randint(len(dataset), (batch_size,)).tolist()
    batch = default_collate([dataset[i] for i in indices])
    gt = batch['gt'].to(device)
    with torch.no_grad():
        if noise is not None:
            black = batch['black_level'].to(device)
            scale = batch['white_level'].to(device) - black
            ratio = batch['ratio'].to(device)
            camera = int(torch.randint(len(noise), (1,)))
            gt, x, _ = noise((gt-black)/scale, scale, ratio, camera)
        else:
            x = batch['lq'].to(device)
    return x, gt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['prelu','relu'], required=True)
    p.add_argument('--run-name', default=None)
    p.add_argument('--boundary-profile', choices=PROFILES, default='baseline')
    p.add_argument('--mixed-profile', choices=MIXED_PROFILES, default='none')
    p.add_argument('--activation-bits', type=int, choices=[8,16], default=8)
    p.add_argument('--ptq-run', default='sid_w8a8_20261031')
    p.add_argument('--steps', type=int, default=20000)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--crop-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--scale-lr', type=float, default=1e-3)
    p.add_argument('--freeze-scales-at', type=float, default=0.8)
    p.add_argument('--distill-weight', type=float, default=0.1)
    p.add_argument('--data', choices=['synthetic','paired'], default='synthetic')
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--save-every', type=int, default=1000)
    p.add_argument('--log-every', type=int, default=50)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--resume', type=Path)
    p.add_argument('--eval-after-train', action='store_true')
    args = p.parse_args()
    if args.mixed_profile != "none" and (args.boundary_profile != "opt1" or args.activation_bits != 16):
        p.error("Mixed phase profiles require --boundary-profile opt1 --activation-bits 16")
    if args.run_name is None:
        args.run_name = "qat_w8a8_v1" if args.boundary_profile == "baseline" else "qat_ablation_"+args.boundary_profile
        if args.activation_bits != 8:
            args.run_name += "_w8a"+str(args.activation_bits)
        if args.mixed_profile != "none":
            args.run_name = "qat_"+args.mixed_profile
    named_profile = re.search(r'(?:^|_)phase_exp([1-6])(?:_|$)', args.run_name)
    if named_profile and args.mixed_profile != 'phase_exp'+named_profile.group(1):
        p.error('Run name implies phase_exp'+named_profile.group(1)+
                ' but --mixed-profile is '+args.mixed_profile+'; correct both before training')
    if min(args.steps,args.batch_size,args.crop_size,args.save_every,args.log_every)<=0 or args.crop_size%8:
        p.error('positive sizes/steps required; crop size must be divisible by 8')
    if not 0<=args.freeze_scales_at<=1 or min(args.lr,args.scale_lr)<=0 or args.distill_weight<0:
        p.error('invalid training hyperparameters')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    device=torch.device(args.device)
    mobile,options,source,exp=build_mobile(args.model,device)
    artifact_path=ROOT/'quantization/artifacts'/args.ptq_run/(args.model+'_w8a8.pth')
    artifact=torch.load(artifact_path,map_location='cpu')
    if artifact['metadata']['checkpoint_sha256']!=sha(source):
        raise ValueError('PTQ encodings were calibrated for a different source checkpoint')
    out=ROOT/'quantization/artifacts'/args.run_name/args.model
    out.mkdir(parents=True,exist_ok=True)
    if (out/'run.json').exists() and not args.resume:
        raise FileExistsError('Run exists. Use --resume or a new --run-name.')
    dataset,noise,data_meta=make_training_data(options,exp,args.data,args.crop_size,device)
    qat=QATModel(mobile,artifact,args.boundary_profile,args.activation_bits,args.mixed_profile).to(device).train()
    teacher=mobile.requires_grad_(False).eval()
    optimizer=torch.optim.Adam([
        {'params':qat.graph.parameters(),'lr':args.lr},
        {'params':qat.scale_parameters(),'lr':args.scale_lr}])
    settings={k:v for k,v in vars(args).items() if k not in ('resume','eval_after_train','device','log_every','save_every')}
    metadata=dict(model=args.model,settings=settings,source_checkpoint=str(source.relative_to(ROOT)),
                  checkpoint_sha256=sha(source),ptq_artifact=str(artifact_path.relative_to(ROOT)),
                  ptq_sha256=sha(artifact_path),training_data=data_meta,torch_version=str(torch.__version__),
                  quantization=f'W8, activation profile {args.mixed_profile}, default A{args.activation_bits}; per-node activation_bit_map is authoritative; symmetric, W [-127,127], A [-{qat.qmax},{qat.qmax}]; weights per output channel, activations per tensor; learned step sizes, then static',
                  activation_bits=args.activation_bits, weight_bits=8,
                  scale_initialization='Preserve PTQ clipping ranges: s_new = s_ptq * qmax_ptq / qmax_new',
                  boundary_policy=qat.policy, mixed_profile=args.mixed_profile, activation_bit_map=qat.activation_bit_map,
                  exceptions=['FP32 PReLU slopes','exact fixed Haar/rearrangement constants'],
                  evaluation='FP32 Q/DQ simulation, not bit-exact integer FPGA',
                  source_sha256={str(f.relative_to(ROOT)):sha(f) for f in [Path(__file__).resolve(),ROOT/'quantization/qat_model.py',ROOT/'quantization/static_ptq.py',ROOT/'quantization/boundaries.py',ROOT/'quantization/mixed_precision.py']})
    start=0
    if args.resume:
        saved=torch.load(args.resume,map_location=device)
        saved['metadata']['settings'].setdefault('boundary_profile', 'baseline')
        saved['metadata']['settings'].setdefault('activation_bits', 8)
        saved['metadata']['settings'].setdefault('mixed_profile', 'none')
        if saved['metadata'].get('activation_bit_map', qat.activation_bit_map) != qat.activation_bit_map:
            raise ValueError('Resume activation bit map differs')
        if saved['metadata'].get('boundary_policy', qat.policy) != qat.policy:
            raise ValueError('Resume quantization boundary policy differs')
        if saved['metadata']['settings']!=settings or saved['metadata']['checkpoint_sha256']!=metadata['checkpoint_sha256'] or saved['metadata']['ptq_sha256']!=metadata['ptq_sha256']:
            raise ValueError('Resume settings/source/encodings differ; reuse original hyperparameters')
        if saved['metadata']['training_data']!=data_meta:
            raise ValueError('Training data differs from resume checkpoint')
        qat.load_state_dict(saved['qat_state'],strict=True);optimizer.load_state_dict(saved['optimizer']);start=saved['step']
        if not 0<=start<=args.steps:raise ValueError('Resume step is outside configured training length')
    (out/'run.json').write_text(json.dumps(metadata,indent=2))
    if start==0 and args.boundary_profile == "baseline" and args.activation_bits == 8:
        with torch.no_grad():
            from quantization.static_ptq import StaticPTQ
            baseline=StaticPTQ(mobile);baseline.load_artifact(artifact_path)
            probe=torch.randn(1,4,32,48,device=device)*.02+.03
            torch.testing.assert_close(qat(probe),baseline.run(probe),rtol=1e-5,atol=1e-6)
        del baseline
    log=out/'training.csv'
    fields=['step','loss','l1','distillation','weight_lr','scale_lr','input_scale','scales_frozen','seconds']
    if start==0:
        with log.open('w') as f:csv.DictWriter(f,fieldnames=fields).writeheader()
    elif log.exists():
        with log.open() as f:retained=[r for r in csv.DictReader(f) if int(r['step'])<=start]
        with log.open('w') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(retained)
    freeze_step=max(1,int(args.steps*args.freeze_scales_at))
    for step in range(start+1,args.steps+1):
        t=time.time()
        # Per-step RNG makes direct-dataset sampling reproducible across resume.
        torch.manual_seed(args.seed+step);np.random.seed(args.seed+step);random.seed(args.seed+step)
        frozen=step>=freeze_step
        if frozen:qat.freeze_scales()
        cosine=.1+.9*.5*(1+math.cos(math.pi*(step-1)/args.steps))
        optimizer.param_groups[0]['lr']=args.lr*cosine
        optimizer.param_groups[1]['lr']=0. if frozen else args.scale_lr*cosine
        x,gt=train_batch(dataset,noise,args.batch_size,device)
        optimizer.zero_grad(set_to_none=True)
        prediction=qat(x)
        l1=F.l1_loss(prediction,gt)
        if args.distill_weight:
            with torch.no_grad():target=teacher(x)
            distill=F.l1_loss(prediction,target)
        else:distill=l1.new_zeros(())
        loss=l1+args.distill_weight*distill
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite QAT loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(qat.parameters(),1.,error_if_nonfinite=True)
        optimizer.step();qat.clamp_scales()
        if step==1 or step%args.log_every==0 or step==args.steps:
            row=dict(step=step,loss=float(loss),l1=float(l1),distillation=float(distill),weight_lr=optimizer.param_groups[0]['lr'],
                     scale_lr=optimizer.param_groups[1]['lr'],input_scale=float(qat.activation_scale('raw')),scales_frozen=frozen,seconds=time.time()-t)
            with log.open('a') as f:csv.DictWriter(f,fieldnames=fields).writerow(row)
            print(json.dumps(row),flush=True)
        if step%args.save_every==0 or step==args.steps:
            saved=dict(step=step,qat_state=qat.state_dict(),optimizer=optimizer.state_dict(),metadata=metadata)
            atomic_save(saved,out/f'qat_step_{step}.pth');atomic_save(saved,out/'qat_latest.pth')
    qat.eval()
    export_path=out/(f'qat_w8a{args.activation_bits}_qdq.pth' if args.mixed_profile == 'none' else 'qat_w8_mixed_qdq.pth')
    runner=qat.export_qdq(export_path,dict(metadata,step=args.steps,training_complete=True))
    with torch.no_grad():
        probe=torch.randn(1,4,64,80,device=device)*.02+.03
        a,b=qat(probe),runner.run(probe)
        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
        error=float((a-b).abs().max())
    (out/'export_check.json').write_text(json.dumps(dict(max_abs_error=error,steps=args.steps,artifact_sha256=sha(export_path)),indent=2))
    print('Training/export finished:',export_path,flush=True)
    if args.eval_after_train:
        del qat,teacher,mobile,runner,optimizer
        torch.cuda.empty_cache()
        subprocess.run([sys.executable,str(ROOT/'quantization/evaluate_qat.py'),'--artifact',str(export_path),'--device',args.device,'--resume'],check=True)

if __name__=='__main__':main()
