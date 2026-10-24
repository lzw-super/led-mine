"""Prepare full packed RAW arrays and group-level splits; never save cropped patches.

Run from the repository root with PYTHONPATH=.
"""
import argparse
import json
from pathlib import Path
import random
import re
import zipfile

import numpy as np
import rawpy

from led.data.raw_utils import get_white_level_per_channel, pack_raw_bayer

NAME = re.compile(r'^(?:clean_avg\d+|noisy_\d+)_ISO(?P<iso>\d+)_EXP(?P<exp>[\d.]+)ms_(?P<group>\d+_\d{8}_\d{6}_\d+)\.(?:raw16|dng)$')


def parse_name(path):
    match = NAME.fullmatch(Path(path).name)
    if match is None:
        raise ValueError(f'Unrecognized RAW filename: {path}')
    return match['group'], float(match['iso']) * float(match['exp'])


def split_groups(groups, val_groups=2, seed=2022):
    groups = sorted(groups)
    if not 0 < val_groups < len(groups):
        raise ValueError('Need nonempty training and validation groups.')
    random.Random(seed).shuffle(groups)
    return set(groups[val_groups:]), set(groups[:val_groups])


def prepare(root, output, val_groups=2, seed=2022):
    root, output = Path(root).resolve(), Path(output).resolve()
    for folder in ('RAW_clean', 'RAW_noise'):
        with zipfile.ZipFile(root / (folder + '.zip')) as archive:
            for member in archive.infolist():
                target = (root / member.filename).resolve()
                if root not in target.parents:
                    raise ValueError('Unsafe archive path.')
                if not member.is_dir() and (not target.exists() or target.stat().st_size != member.file_size):
                    archive.extract(member, root)
    clean = {}
    for path in sorted((root / 'RAW_clean').glob('*.raw16')):
        group, exposure = parse_name(path)
        if group in clean:
            raise ValueError(f'Duplicate clean group: {group}')
        clean[group] = (path, exposure)
    noisy = sorted((root / 'RAW_noise').glob('*.dng'))
    counts = {group: 0 for group in clean}
    for path in noisy:
        group, _ = parse_name(path)
        if group not in clean:
            raise ValueError(f'No clean for {path}')
        counts[group] += 1
    if any(count != 3 for count in counts.values()):
        raise ValueError(f'Expected three noisy images per clean: {counts}')
    train, val = split_groups(clean, val_groups, seed)
    (output / 'packed').mkdir(parents=True, exist_ok=True)
    records = {'train': [], 'val': []}
    metadata = {}
    saved_clean = set()
    for path in noisy:
        group, exposure = parse_name(path)
        gt, gt_exposure = clean[group]
        with rawpy.imread(str(path)) as raw:
            mosaic = raw.raw_image_visible
            pattern = raw.raw_pattern.copy()
            black = np.asarray(raw.black_level_per_channel, dtype=np.float32)
            white = get_white_level_per_channel(raw)
            if np.any(white <= black):
                raise ValueError(f'Invalid RAW levels: {path}')
            signature = (mosaic.shape, pattern.tolist(), black.tolist(), white.tolist())
            if group in metadata and signature != metadata[group]:
                raise ValueError(f'Inconsistent metadata within group {group}')
            metadata[group] = signature
            packed = pack_raw_bayer(mosaic, pattern)
            lq_rel = 'packed/' + path.stem + '.npy'
            gt_rel = 'packed/' + gt.stem + '.npy'
            np.save(output / lq_rel, packed)
            if group not in saved_clean:
                # User confirmed clean is only averaged; same Bayer/levels as noisy.
                flat = np.fromfile(gt, dtype='<u2')
                if flat.size != mosaic.size:
                    raise ValueError(f'Clean size disagrees with DNG visible size: {gt}')
                np.save(output / gt_rel, pack_raw_bayer(flat.reshape(mosaic.shape), pattern))
                saved_clean.add(group)
            wb = np.asarray(raw.camera_whitebalance, dtype=np.float32)
            if wb[1] <= 0:
                raise ValueError(f'Invalid white balance: {path}')
            wb /= wb[1]
            if wb[3] == 0:
                wb[3] = wb[1]
            record = dict(group=group, lq_path=str(path), gt_path=str(gt),
                          lq_packed=lq_rel, gt_packed=gt_rel, shape=list(packed.shape),
                          ratio=gt_exposure/exposure, black_level=black.tolist(),
                          white_level=white.tolist(), wb=wb.tolist())
            records['train' if group in train else 'val'].append(record)
    for split, pairs in records.items():
        payload = dict(seed=seed, split=split, groups=sorted(train if split == 'train' else val),
                       clean_format='little-endian uint16; averaged only; same DNG levels and Bayer',
                       pairs=pairs)
        (output / (split + '.json')).write_text(json.dumps(payload, indent=2) + '\n')
        print(f'{split}: {len(payload["groups"])} groups, {len(pairs)} pairs')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='../../Datasets/mine-mobile-image-dataset')
    parser.add_argument('--output', default='../../Datasets/mine-mobile-image-dataset/prepared')
    parser.add_argument('--val-groups', type=int, default=2)
    parser.add_argument('--seed', type=int, default=2022)
    args = parser.parse_args()
    prepare(args.root, args.output, args.val_groups, args.seed)
