"""Guard the controlled training comparison and RAW model integration."""
from pathlib import Path

import os
import unittest
import torch

from led.archs import build_network
from led.utils.options import _merge_from_base, yaml_load


ROOT = Path(__file__).resolve().parents[1]
LABELS = ['LearningDWT_Haar_L3_RepNCB_W32', 'SplitterNet', 'MRLFN_Paper_S2D_K4_N4_D32']


def load(label):
    path = f'options/LED/pretrain/CVPR20_ELD_{label}.yaml'
    return _merge_from_base(yaml_load(path), path)


def check_reference(label):
    os.chdir(ROOT)
    baseline, opt = load('Setting'), load(label)
    excluded = {'name', 'base', 'network_g', 'repnr_opt'}
    assert {k: v for k, v in opt.items() if k not in excluded} == {
        k: v for k, v in baseline.items() if k not in excluded}
    assert opt['repnr_opt'] is None
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        network = build_network(opt['network_g'])
        sample = torch.rand(1, 4, 32, 48)
        output = network(sample)
        assert output.shape == sample.shape
        assert torch.isfinite(output).all()
        # Compare the adapter against the underlying reference RAW convention.
        order = [0, 1, 3, 2]
        expected = network.network(sample[:, order])[:, order]
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        (output - sample).abs().mean().backward()
        grads = [p.grad for p in network.parameters() if p.requires_grad]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
        settings = dict(opt['train']['optim_g'])
        settings.pop('type')
        torch.optim.Adam(network.parameters(), **settings).step()
    finally:
        torch.set_num_threads(previous_threads)


class TestSIDReferenceArchitectures(unittest.TestCase):
    def test_training_settings_and_backward(self):
        for label in LABELS:
            with self.subTest(architecture=label):
                check_reference(label)


if __name__ == "__main__":
    unittest.main()
