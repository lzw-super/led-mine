"""Manual QAT Q/DQ crop benchmark; static encodings, no recalibration, no INT8 runtime."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch import fx
from torch.nn import functional as F
from quantization.train_qat import build_mobile
from quantization.static_ptq import StaticPTQ, qdq
from scripts.evaluate_crop_merge_sid import build_sid_datasets
from tile_software_test.benchmark_crop import tiled, metrics, boxes

ARTIFACT = ROOT/'quantization/artifacts/qat_llstem_noact_phase_exp3_pres2d/relu_llstem_noact/qat_w8_mixed_qdq.pth'
CONFIGS = [(96, h) for h in (16, 20, 24)]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


class CropQDQ(StaticPTQ):
    """Original static QDQ arithmetic, without saturation counters in timed runs.

    Patch sizes must be even. Pad only the Haar branch; crop its reconstruction
    before the refiner/pre-S2D A8 cast. The noisy skip keeps its original size.
    Boundary names are checked against this particular exported graph.
    """
    def __call__(self, x):
        if x.ndim != 4 or x.shape[1] != 4 or any(n % 2 for n in x.shape[-2:]):
            raise ValueError('Expected N,4,H,W with even H,W')
        self.input_hw = x.shape[-2:]
        return self.run(x)

    def run_node(self, node):
        if self.mode != 'quantize':
            raise RuntimeError('Only frozen quantization is allowed')
        if node.name == 'conv2d':
            args, kwargs = self.fetch_args_kwargs_from_env(node)
            h, w = self.input_hw
            x = args[0]
            if h % 8 or w % 8:
                x = F.pad(x, (0, -w % 8, 0, -h % 8), mode='replicate')
            result = self.call_function(node.target, (x, *args[1:]), kwargs)
        else:
            result = fx.Interpreter.run_node(self, node)
        if node.name == 'conv_transpose2d_2':
            h, w = self.input_hw
            result = result[..., :h, :w]
        if node.name in self.boundaries and isinstance(result, torch.Tensor):
            return qdq(result, self.scales[node.name], self.node_qmax(node.name))
        return result


def load_models(artifact_path, device):
    data = torch.load(artifact_path, map_location='cpu')
    meta = data['metadata']
    if not meta.get('training_complete') or meta['model'] != 'relu_llstem_noact':
        raise ValueError('Expected completed relu_llstem_noact QAT artifact')
    mobile, _, source, _ = build_mobile(meta['model'], device)
    if sha(source) != meta['checkpoint_sha256']:
        raise ValueError('FP32 source checkpoint hash differs from QAT training metadata')
    runner = CropQDQ(mobile)
    runner.load_artifact(artifact_path)  # Checks graph, state, policy, bits and scales.
    nodes = runner.nodes
    if (nodes['conv2d'].args[0].name != 'raw'
            or nodes['phase2_input_a8'].args[0].name != 'conv_transpose2d_2'
            or nodes['conv2d_4'].args[0].name != 'raw'):
        raise ValueError('Unexpected Haar/refiner/skip graph boundaries')
    if data['mixed_profile'] != 'phase_exp3_pres2d':
        raise ValueError('This adapter is verified for phase_exp3_pres2d only')
    return mobile.eval(), runner, data, source


def timed(fn, device):
    if device.type == 'cuda': torch.cuda.synchronize(device)
    start = time.perf_counter()
    output = fn()
    if device.type == 'cuda': torch.cuda.synchronize(device)
    return output, (time.perf_counter()-start)*1000


def difference(a, b):
    d = (a-b).double()
    return dict(mae=d.abs().mean().item(), max_abs=d.abs().max().item(), mse=d.square().mean().item())


def check_small(mobile, runner, artifact_path, device):
    """Check against the unmodified interpreter and an independent padded FX graph."""
    official = StaticPTQ(mobile)
    official.load_artifact(artifact_path)
    results = []
    with torch.inference_mode():
        x = torch.rand(1, 4, 32, 40, device=device)
        expected = official.run(x)
        actual = runner(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        results.append(dict(case='aligned_vs_official', **difference(actual, expected)))
        # Insert explicit pad/crop in a copy of the official graph. Its original
        # run_node still performs all boundary QDQ and saturation counting.
        for h, w in [(28, 36), (32, 36), (28, 40)]:
            reference = StaticPTQ(mobile)
            reference.load_artifact(artifact_path)
            graph = reference.module.graph
            nodes = {n.name:n for n in graph.nodes}
            first = nodes['conv2d']
            with graph.inserting_before(first):
                pad = graph.call_function(F.pad, (nodes['raw'], (0,-w%8,0,-h%8)), {'mode':'replicate'})
            first.replace_input_with(nodes['raw'], pad)
            reconstruction = nodes['conv_transpose2d_2']
            users = list(reconstruction.users)
            import operator
            with graph.inserting_after(reconstruction):
                crop = graph.call_function(operator.getitem, (reconstruction,(Ellipsis,slice(None,h),slice(None,w))))
            for user in users: user.replace_input_with(reconstruction, crop)
            graph.lint(); reference.module.recompile()
            x = torch.rand(1,4,h,w,device=device)
            expected, actual = reference.run(x), runner(x)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            results.append(dict(case=f'boundary_{h}x{w}_vs_explicit_graph', **difference(actual,expected)))
        for t,h in CONFIGS:
            x = torch.rand(1,4,104,120,device=device)
            actual = tiled(torch.nn.Identity(),x,t,h,4)
            assert torch.equal(actual,x)
            results.append(dict(case=f'crop_identity_{t}_{h}', passed=True))
    print(json.dumps(dict(status='small_checks_passed',device=str(device),checks=results),indent=2))


def summarize(out, records, metadata):
    full = [r['full'] for r in records]
    rows = [row for r in records for row in r['tiles']]
    serial = [row for r in records for row in r['serial']]
    for name, values in [('full_image.csv',full),('per_image.csv',rows),('serial_timing.csv',serial)]:
        if not values: continue
        with (out/name).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
    mean=lambda group,key: statistics.mean(float(r[key]) for r in group)
    summary=[]
    for ratio in ['all',100,250,300]:
        for t,h in CONFIGS:
            group=[r for r in rows if r['tile']==t and r['halo']==h and (ratio=='all' or r['ratio']==ratio)]
            if not group: continue
            times=[r['milliseconds'] for r in serial if r['tile']==t and r['halo']==h and (ratio=='all' or r['ratio']==ratio)]
            summary.append(dict(ratio=ratio,tile=t,halo=h,n=len(group),
                **{k:mean(group,k) for k in ['psnr_db','ssim','tiling_psnr_loss_db','tiling_ssim_loss','total_psnr_loss_vs_fp32_db','milliseconds','full_qdq_ms','full_fp32_ms']},
                worst_tiling_psnr_loss_db=max(r['tiling_psnr_loss_db'] for r in group),
                serial_median_ms=statistics.median(times) if times else None))
    atomic_json(out/'summary.json',dict(metadata=metadata,summary=summary))
    with (out/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    lines=['# QAT Q/DQ crop 测试结果','',f"状态：{metadata['status']}；已完成 {len(records)}/{metadata['expected_images']} 幅图。",'',
        '**时间是 FP32 Q/DQ 仿真 + FX 解释器时间，不是原生 INT8/FPGA 性能。**', '',
        f"FP32 整图 PSNR {mean(full,'fp32_psnr_db'):.6f} dB；QDQ 整图 PSNR {mean(full,'qdq_psnr_db'):.6f} dB。",
        f"FP32 整图平均 {mean(full,'fp32_ms'):.2f} ms；QDQ 整图平均 {mean(full,'qdq_ms'):.2f} ms。",'',
        '| T/h | PSNR dB | 分块损失 dB（QDQ整图−tile） | SSIM分块损失 | 总损失 dB（FP32整图−QDQ tile） | 批量平均 ms | 串行中位 ms |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        if r['ratio']!='all':continue
        sm='未测' if r['serial_median_ms'] is None else f"{r['serial_median_ms']:.2f}"
        lines.append(f"| {r['tile']}/{r['halo']} | {r['psnr_db']:.6f} | {r['tiling_psnr_loss_db']:.6f} | {r['tiling_ssim_loss']:.8f} | {r['total_psnr_loss_vs_fp32_db']:.6f} | {r['milliseconds']:.2f} | {sm} |")
    lines += ['', '串行/批量 QDQ 可能因浮点卷积舍入跨越量化阈值而不逐 bit 相同；差异记录在 serial_timing.csv，未把浮点 allclose 当成整数等价保证。',
        '指标沿用 tile_software_test 的标准 PSNR（无 1e-8 epsilon）、全图 illumination correction、RAW SSIM、crop_border=2；原始输出误差不做校正。',
        '本轮 FP32 基线为 QAT 元数据对应的源权重，不能直接将旧 net_g_255000.pth 基线与其混用。halo=20 按原值执行，存在 Haar L3 相位偏移。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact',type=Path,default=ARTIFACT)
    p.add_argument('--output',type=Path,default=ROOT/'tile_software_test/results_qat_96')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--serial-repeats',type=int,default=3)
    p.add_argument('--max-per-ratio',type=int,default=0)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--check-only',action='store_true')
    args=p.parse_args()
    if args.batch_size<1 or args.serial_repeats<0 or args.max_per_ratio<0:p.error('Invalid batch/repeat/image count')
    torch.set_num_threads(4);torch.manual_seed(2026)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    device=torch.device(args.device)
    if device.type=='cuda':torch.cuda.set_device(device)
    mobile,runner,artifact,source=load_models(args.artifact,device)
    if args.check_only:
        check_small(mobile,runner,args.artifact,device);return
    datasets=build_sid_datasets(ROOT/'options/base/dataset/test/SID_SonyA7S2_val_split.yaml',[100,250,300],None,'npz')
    limits={ratio:min(len(d),args.max_per_ratio) if args.max_per_ratio else len(d) for ratio,d in datasets}
    signature=dict(artifact_sha256=sha(args.artifact),source_sha256=sha(source),configs=CONFIGS,batch_size=args.batch_size,
        serial_repeats=args.serial_repeats,max_per_ratio=args.max_per_ratio,device=str(device),
        gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu',torch=torch.__version__,
        cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
        dataset={str(r):dict(count=len(d),list_sha256=sha(d.opt['data_pair_list'])) for r,d in datasets},
        script_sha256=sha(__file__),tiling_script_sha256=sha(ROOT/'tile_software_test/benchmark_crop.py'),
        runtime_source_sha256={name:sha(ROOT/name) for name in ['quantization/static_ptq.py','quantization/mixed_precision.py','quantization/boundaries.py','quantization/train_qat.py','led/archs/sid_reference/mobile_haar.py']})
    signature=json.loads(json.dumps(signature))
    out=args.output
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError('Output exists; use --resume or a new --output')
    out.mkdir(parents=True,exist_ok=True);(out/'images').mkdir(exist_ok=True)
    meta_path=out/'metadata.json'
    if args.resume and meta_path.exists():
        if json.loads(meta_path.read_text())['signature']!=signature:raise ValueError('Resume configuration/source mismatch')
    elif args.resume and any((out/'images').iterdir()):
        raise ValueError('Cannot resume image results without metadata')
    metadata=dict(status='running',signature=signature,expected_images=sum(limits.values()),artifact=str(args.artifact),
        source=str(source),quantization_profile=artifact['mixed_profile'],activation_bit_map=artifact['activation_bit_map'],
        source_training=artifact['metadata'],runtime='FP32 QDQ FX interpreter; saturation counters disabled',
        boundary='Pad Haar input right/bottom replicate to 8; crop preliminary before refiner A8 cast; even dimensions only',
        timing='CUDA synchronized wall; GPU input/output, includes grouping/forward/crop; excludes IO/H2D/metrics/correction',
        amp=False,tf32=False,cudnn_deterministic=True)
    atomic_json(meta_path,metadata)
    records=[]
    with torch.inference_mode():
        for ratio,dataset in datasets:
            warmed=False
            for index in range(limits[ratio]):
                path=out/'images'/f'{ratio}_{index:03d}.json'
                if path.exists():
                    records.append(json.loads(path.read_text()));continue
                item=dataset[index]
                x=item['lq'][None].to(device);gt=item['gt'][None].to(device)
                if not warmed:
                    for _ in range(2):mobile(x);runner(x)
                    for t,h in CONFIGS:tiled(runner,x,t,h,args.batch_size)
                    warmed=True
                fp,fp_ms=timed(lambda:mobile(x),device)
                full,full_ms=timed(lambda:runner(x),device)
                if not torch.isfinite(full).all() or not torch.isfinite(fp).all():raise FloatingPointError('Nonfinite full output')
                fm,qm=metrics(fp,gt),metrics(full,gt)
                record=dict(full=dict(ratio=ratio,index=index,lq=item['lq_path'],fp32_ms=fp_ms,qdq_ms=full_ms,
                    **{'fp32_'+k:v for k,v in fm.items()},**{'qdq_'+k:v for k,v in qm.items()},
                    quantization_psnr_loss_db=fm['psnr_db']-qm['psnr_db']),tiles=[],serial=[])
                configs=CONFIGS.copy();random.Random(2026+ratio*1000+index).shuffle(configs)
                for t,h in configs:
                    pred,ms=timed(lambda:tiled(runner,x,t,h,args.batch_size),device)
                    if not torch.isfinite(pred).all():raise FloatingPointError('Nonfinite tile output')
                    m=metrics(pred,gt);diff=difference(pred,full)
                    record['tiles'].append(dict(ratio=ratio,index=index,lq=item['lq_path'],tile=t,halo=h,core=t-2*h,
                        tile_count=len(list(boxes(*x.shape[-2:],t,h))),batch_size=args.batch_size,milliseconds=ms,
                        full_qdq_ms=full_ms,full_fp32_ms=fp_ms,**m,
                        tiling_psnr_loss_db=qm['psnr_db']-m['psnr_db'],tiling_ssim_loss=qm['ssim']-m['ssim'],
                        total_psnr_loss_vs_fp32_db=fm['psnr_db']-m['psnr_db'],
                        uncorrected_tiling_psnr_loss_db=qm['uncorrected_psnr_db']-m['uncorrected_psnr_db'],
                        **{k+'_vs_qdq_full':v for k,v in diff.items()}))
                if index==0 and args.serial_repeats:
                    for t,h in [(0,0)]+CONFIGS:
                        fn=(lambda:runner(x)) if t==0 else (lambda t=t,h=h:tiled(runner,x,t,h,1))
                        serial_pred=fn()
                        batch_pred=full if t==0 else tiled(runner,x,t,h,args.batch_size)
                        diff=difference(serial_pred,batch_pred)
                        if not all(math.isfinite(v) for v in diff.values()):raise FloatingPointError('Invalid serial output')
                        for rep in range(args.serial_repeats):
                            _,ms=timed(fn,device)
                            record['serial'].append(dict(ratio=ratio,index=index,tile=t,halo=h,repeat=rep,milliseconds=ms,
                                **{k+'_serial_vs_batch':v for k,v in diff.items()}))
                atomic_json(path,record);records.append(record)
                summarize(out,records,metadata)
                print(f'{len(records)}/{sum(limits.values())}: ratio={ratio}, index={index}, QDQ full PSNR={qm["psnr_db"]:.4f}',flush=True)
    expected={(ratio,i) for ratio in limits for i in range(limits[ratio])}
    assert {(r['full']['ratio'],r['full']['index']) for r in records}==expected
    metadata['status']='smoke_complete' if args.max_per_ratio else 'complete'
    atomic_json(meta_path,metadata);summarize(out,records,metadata)
    print(f'DONE: {out}/REPORT.md',flush=True)

if __name__=='__main__':main()
