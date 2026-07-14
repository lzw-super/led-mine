import tempfile
import unittest
from pathlib import Path

import torch

from scripts.prepare_led_deploy_checkpoint import prepare_deploy_checkpoint


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'experiments/LED_Pretrain_CVPR20_ELD_Setting/models/net_g_245000.pth'
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
