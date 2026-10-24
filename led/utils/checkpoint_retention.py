"""Opt-in retention for training: latest + two numbered snapshots + best PSNR."""
import math
import os
from pathlib import Path
import shutil

import torch
from led.utils import get_root_logger


class CheckpointRetention:
    def __init__(self, model, opt):
        self.model = model
        self.root = Path(opt['path']['models'])
        self.states = Path(opt['path']['training_states'])
        self.dataset = opt['logger']['checkpoint_retention']['dataset']
        self.enabled = opt.get('rank', 0) == 0
        self.best = -math.inf
        self.best_path = self.root / 'net_g_best_psnr.pth'
        if self.enabled and self.best_path.exists():
            metadata = torch.load(self.best_path, map_location='cpu')['selection']
            if metadata['dataset'] != self.dataset:
                raise ValueError('Best checkpoint belongs to a different validation dataset.')
            self.best = metadata['psnr']
            if hasattr(model, '_initialize_best_metric_results'):
                model._initialize_best_metric_results(self.dataset)
                model.best_metric_results[self.dataset]['psnr'].update(
                    val=self.best, iter=metadata['iteration'])

    def save(self, epoch, iteration):
        if not self.enabled:
            return
        self.model.save(epoch, iteration)
        source = self.root / f'net_g_{iteration}.pth'
        # Check existing framework writes succeeded before publishing or pruning.
        torch.load(source, map_location='cpu')
        torch.load(self.states / f'{iteration}.state', map_location='cpu')
        latest = self.root / 'net_g_latest.pth'
        temporary = latest.with_suffix('.pth.tmp')
        shutil.copyfile(source, temporary)
        os.replace(temporary, latest)
        numbered = sorted((int(p.stem[len('net_g_'):]), p)
                          for p in self.root.glob('net_g_*.pth')
                          if p.stem[len('net_g_'):].isdigit())
        keep = {step for step, _ in numbered[-2:]}
        for step, path in numbered[:-2]:
            path.unlink()
        for path in self.states.glob('*.state'):
            if path.stem.isdigit() and int(path.stem) not in keep:
                path.unlink()

    def consider_best(self, dataset, iteration):
        if not self.enabled or dataset != self.dataset:
            return
        score = float(self.model.metric_results['psnr'])
        if not math.isfinite(score) or score <= self.best:
            return
        payload = {}
        networks = [('params', self.model.net_g)]
        if hasattr(self.model, 'net_g_ema'):
            networks.append(('params_ema', self.model.net_g_ema))
        for key, net in networks:
            bare = self.model.get_bare_model(net)
            payload[key] = {k: v.detach().cpu() for k, v in bare.state_dict().items()}
        payload['selection'] = dict(dataset=dataset, psnr=score, iteration=iteration,
                                    evaluated_params='params_ema' if hasattr(self.model, 'net_g_ema') else 'params')
        temporary = self.best_path.with_suffix('.pth.tmp')
        torch.save(payload, temporary)
        os.replace(temporary, self.best_path)
        self.best = score
        get_root_logger().info(
            f'Saved best PSNR checkpoint: {score:.6f} at iter {iteration} ({dataset}).')
