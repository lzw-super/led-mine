"""Scene-split mobile RAW pairs with online stratified random crops."""
import bisect
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from led.utils.registry import DATASET_REGISTRY


def crop_starts(length, size, stride):
    if size <= 0 or not 0 < stride <= size or length < size:
        raise ValueError('Require length >= crop_size >= stride > 0.')
    starts = list(range(0, length - size + 1, stride))
    if starts[-1] != length - size:
        starts.append(length - size)
    return starts


@DATASET_REGISTRY.register()
class MobilePairedRAWDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        manifest = Path(opt['manifest'])
        self.root = manifest.parent
        self.records = json.loads(manifest.read_text())['pairs']
        if not self.records:
            raise ValueError('Empty pair manifest.')
        self.training = opt['phase'] == 'train'
        self.size = int(opt.get('crop_size', 256))
        self.stride = int(opt.get('stride', self.size // 2))
        self.grids, self.ends = [], []
        for record in self.records:
            _, h, w = record['shape']
            ys = crop_starts(h, self.size, self.stride)
            xs = crop_starts(w, self.size, self.stride)
            if self.training:
                grid = [(y, x) for y in ys for x in xs]
            else:
                # Fixed corners + centre, no stochastic validation augmentation.
                grid = list(dict.fromkeys([(0, 0), (0, w-self.size),
                    (h-self.size, 0), (h-self.size, w-self.size),
                    ((h-self.size)//2, (w-self.size)//2)]))
            self.grids.append(grid)
            self.ends.append((self.ends[-1] if self.ends else 0) + len(grid))

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        pair = bisect.bisect_right(self.ends, index)
        local = index - (self.ends[pair-1] if pair else 0)
        record = self.records[pair]
        y, x = self.grids[pair][local]
        _, h, w = record['shape']
        if self.training:
            # One sample per stride cell, redrawn on every access (inclusive edges).
            y += torch.randint(min(self.stride-1, h-self.size-y)+1, ()).item()
            x += torch.randint(min(self.stride-1, w-self.size-x)+1, ()).item()
        patches = []
        for kind in ('lq', 'gt'):
            packed = np.load(self.root / record[kind + '_packed'], mmap_mode='r')
            patch = np.array(packed[:, y:y+self.size, x:x+self.size], dtype=np.float32)
            black = np.asarray(record['black_level'], dtype=np.float32)[:, None, None]
            white = np.asarray(record['white_level'], dtype=np.float32)[:, None, None]
            patch = (patch - black) / (white - black)
            if kind == 'lq':
                patch *= record['ratio']
            patches.append(torch.from_numpy(patch.clip(0, 1)))
        if self.training:
            for option, axis in [('use_hflip', 2), ('use_vflip', 1)]:
                if self.opt.get(option, True) and torch.rand(()).item() < 0.5:
                    patches = [patch.flip(axis) for patch in patches]
        return dict(lq=patches[0], gt=patches[1],
                    ratio=torch.tensor(record['ratio'], dtype=torch.float32),
                    wb=torch.tensor(record['wb'], dtype=torch.float32),
                    ccm=torch.eye(3),  # RAW-only metrics; DNG has no usable CCM.
                    lq_path=record['lq_path'], gt_path=record['gt_path'],
                    crop_origin=torch.tensor([y, x]))
