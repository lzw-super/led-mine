import json

import numpy as np
import pytest
import torch

from led.data.mobile_paired_raw_dataset import MobilePairedRAWDataset, crop_starts
from scripts.data_preparation.prepare_mobile_pairs import parse_name, split_groups
from led.utils.options import _merge_from_base, yaml_load


def test_pairing_and_group_split():
    a, ev_a = parse_name('clean_avg5_ISO200_EXP15000ms_001_20261023_191930_203.raw16')
    b, ev_b = parse_name('noisy_1_ISO6400_EXP30ms_001_20261023_191930_203.dng')
    assert a == b and ev_a / ev_b == 15.625
    train, val = split_groups(map(str, range(25)))
    assert len(train) == 23 and len(val) == 2 and not train & val
    assert (train, val) == split_groups(map(str, range(25)))
    with pytest.raises(ValueError):
        parse_name('bad.dng')


def test_edge_coverage():
    assert crop_starts(256, 256, 128) == [0]
    assert crop_starts(600, 256, 128) == [0, 128, 256, 344]
    with pytest.raises(ValueError):
        crop_starts(128, 256, 128)


def dataset(tmp_path, phase):
    # After black subtraction and ratio, both images must be exactly aligned.
    gt = (np.arange(4*12*14).reshape(4, 12, 14) % 100 * 2 + 64).astype('uint16')
    lq = ((gt-64)//2 + 64).astype('uint16')
    np.save(tmp_path/'lq.npy', lq)
    np.save(tmp_path/'gt.npy', gt)
    pair = dict(lq_packed='lq.npy', gt_packed='gt.npy', shape=list(gt.shape),
                ratio=2, black_level=[64]*4, white_level=[1023]*4,
                wb=[1]*4, lq_path='noisy.dng', gt_path='clean.raw16')
    (tmp_path/'pairs.json').write_text(json.dumps(dict(pairs=[pair])))
    return MobilePairedRAWDataset(dict(manifest=str(tmp_path/'pairs.json'), phase=phase,
                                      crop_size=8, stride=4, use_hflip=True, use_vflip=True))


def test_random_paired_crops_and_flips(tmp_path):
    ds = dataset(tmp_path, 'train')
    origins = set()
    for _ in range(25):
        sample = ds[0]
        assert sample['lq'].shape == (4, 8, 8)
        torch.testing.assert_close(sample['lq'], sample['gt'])
        origins.add(tuple(sample['crop_origin'].tolist()))
    assert len(origins) > 1
    edge = ds[len(ds)-1]
    assert edge['crop_origin'].tolist() == [4, 6]
    torch.testing.assert_close(edge['lq'], edge['gt'])


def test_validation_is_fixed(tmp_path):
    ds = dataset(tmp_path, 'val')
    assert len(ds) == 5
    torch.testing.assert_close(ds[0]['lq'], ds[0]['lq'])
    assert ds[4]['crop_origin'].tolist() == [2, 3]


def test_configs_differ_only_in_loss_and_name():
    configs = []
    for suffix in ['L1', 'L1_SSIM']:
        path = f'options/LED/finetune/Mobile_UNet_{suffix}.yaml'
        configs.append(_merge_from_base(yaml_load(path), path))
    a, b = configs
    assert 'noise_g' not in a and a['repnr_opt'] is None
    assert b['train'].pop('ssim_opt')['loss_weight'] == .1
    for c in configs:
        c.pop('base', None)
        c.pop('name')
    assert a == b
