import tempfile
import unittest
from pathlib import Path

import torch

from scripts.prepare_led_deploy_checkpoint import prepare_deploy_checkpoint
from led.archs.mrlfn_arch import MRLFN


ROOT = Path(__file__).resolve().parents[1]
LIVE_SOURCE = ROOT / 'experiments/LED_Pretrain_CVPR20_ELD_Setting/models/net_g_245000.pth'
ARCHIVED_SOURCES = sorted(ROOT.glob(
    'experiments/LED_Pretrain_CVPR20_ELD_Setting_archived_*/models/net_g_245000.pth'))
SOURCE = (
    LIVE_SOURCE if LIVE_SOURCE.is_file()
    else ARCHIVED_SOURCES[-1] if ARCHIVED_SOURCES
    else LIVE_SOURCE)
OPTIONS = ROOT / 'options/base/network_g/repnr_unet.yaml'


class PrepareDeployCheckpointTest(unittest.TestCase):
    def test_repnr_params_are_deployed_and_equivalent(self):
        self.assertTrue(SOURCE.is_file(), f'Missing source checkpoint: {SOURCE}')
        self.assertTrue(OPTIONS.is_file(), f'Missing network options: {OPTIONS}')

        with tempfile.TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / 'deployed.pth'
            result = prepare_deploy_checkpoint(SOURCE, output, OPTIONS, param_key='auto')
            saved = torch.load(output, map_location='cpu')

        self.assertEqual(result['conversion_mode'], 'repnr_to_deploy')
        self.assertEqual(result['source_param_key'], 'params')
        self.assertIn('params_deploy', saved)
        self.assertTrue(all('.main_weight' not in key for key in saved['params_deploy']))
        self.assertLessEqual(result['max_abs_error'], 1e-5)

    def test_deployed_params_are_rewrapped_without_repnr_conversion(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            first_output = tmp_dir / 'first_deployed.pth'
            second_output = tmp_dir / 'second_deployed.pth'
            prepare_deploy_checkpoint(SOURCE, first_output, OPTIONS, param_key='auto')
            result = prepare_deploy_checkpoint(first_output, second_output, OPTIONS, param_key='auto')
            first_saved = torch.load(first_output, map_location='cpu')
            second_saved = torch.load(second_output, map_location='cpu')

        self.assertEqual(result['conversion_mode'], 'already_deployed')
        self.assertEqual(result['source_param_key'], 'params_deploy')
        self.assertEqual(first_saved['params_deploy'].keys(), second_saved['params_deploy'].keys())
        for key in first_saved['params_deploy']:
            self.assertTrue(torch.equal(first_saved['params_deploy'][key], second_saved['params_deploy'][key]))

    def test_mrlfn_training_params_are_fused_to_deployment_state(self):
        options = ROOT / 'options/base/network_g/mrlfn.yaml'
        torch.manual_seed(2026)
        training_network = MRLFN().eval()
        sample = torch.rand(1, 4, 64, 64)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            source = tmp_dir / 'mrlfn_training.pth'
            output = tmp_dir / 'mrlfn_deployed.pth'
            torch.save({'params': training_network.state_dict()}, source)
            result = prepare_deploy_checkpoint(source, output, options, param_key='params')
            saved = torch.load(output, map_location='cpu')

        deployed_network = MRLFN(deploy=True).eval()
        deployed_network.load_state_dict(saved['params_deploy'], strict=True)
        with torch.no_grad():
            training_output = training_network(sample)
            deployed_output = deployed_network(sample)

        self.assertEqual(result['conversion_mode'], 'reparameterized_to_deploy')
        self.assertLessEqual(result['max_abs_error'], 2e-6)
        self.assertTrue(any('.reparam_conv.weight' in key for key in saved['params_deploy']))
        self.assertFalse(any('.expand_conv.' in key for key in saved['params_deploy']))
        torch.testing.assert_close(training_output, deployed_output, rtol=1e-5, atol=2e-6)
