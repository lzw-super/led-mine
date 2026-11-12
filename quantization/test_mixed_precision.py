"""Activation precision ablation: ranges, gradients, opt1 aliases and export."""
import tempfile
import unittest
from pathlib import Path
import torch
from quantization.qat_model import QATModel, LearnedFakeQuant
from quantization.static_ptq import StaticPTQ, qdq
from quantization.train_qat import build_mobile, ROOT


class MixedTests(unittest.TestCase):
    def test_real_models(self):
        torch.set_num_threads(4)
        for model, profile in [(m,p) for m in ('prelu','relu') for p in ('phase_exp1','phase_exp2','phase_exp3','phase_exp4','phase_exp5','phase_exp6')]:
            with self.subTest(model=model, profile=profile), tempfile.TemporaryDirectory() as tmp:
                mobile, _, _, _ = build_mobile(model, torch.device('cpu'))
                ptq = torch.load(ROOT/'quantization/artifacts/sid_w8a8_20261031'/(model+'_w8a8.pth'), map_location='cpu')
                q8 = QATModel(mobile, ptq, 'opt1')
                q = QATModel(mobile, ptq, 'opt1', 16, profile)
                self.assertEqual(len(q.activation_log_scales), 40 if profile == 'phase_exp1' else (38 if profile in ('phase_exp5','phase_exp6') else 39))
                self.assertEqual(q.node_qmax('raw'),32767)
                self.assertEqual(q.node_qmax('conv_transpose2d_2'),32767)
                self.assertEqual(q.node_qmax('conv2d_3'),32767 if profile in ('phase_exp5','phase_exp6') else 127)
                if profile in ('phase_exp5','phase_exp6'):
                    torch.testing.assert_close(q.activation_scale('conv2d_3'), q.activation_scale('conv_transpose2d_2'), rtol=0, atol=0)
                self.assertEqual(q.node_qmax('conv2d_4'),127 if profile == 'phase_exp1' else 32767)
                self.assertEqual(q.node_qmax('head'),32767 if profile in ('phase_exp3','phase_exp5') else 127)
                self.assertEqual(q.node_qmax('cat_4'),127 if profile in ('phase_exp1','phase_exp2') else 32767)
                self.assertEqual(q.node_qmax('conv_transpose2d_3'),q.node_qmax('fusion'))
                for name in q.activation_log_scales:
                    torch.testing.assert_close(q.activation_scale(name)*q.node_qmax(name), ptq['activation_scales'][name]*127)
                for name in q.weight_log_scales:
                    torch.testing.assert_close(q.weight_scale(name), q8.weight_scale(name), rtol=0, atol=0)
                if profile != 'phase_exp1':
                    torch.testing.assert_close(q.activation_scale('conv2d_4'), q.activation_scale('raw'), rtol=0, atol=0)
                x = torch.randn(1,4,16,24)*.03+.04
                opt = torch.optim.Adam(q.parameters(), lr=1e-5)
                q(x).square().mean().backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in q.scale_parameters()))
                opt.step(); q.clamp_scales()
                out = Path(tmp)/'w8a16.pth'
                runner = q.export_qdq(out, {})
                loaded = StaticPTQ(mobile); loaded.load_artifact(out)
                self.assertEqual(loaded.activation_bits, 16)
                self.assertEqual(loaded.activation_bit_map, q.activation_bit_map)
                self.assertEqual(loaded.mixed_profile, profile)
                for record in loaded.integer_weights.values():
                    self.assertEqual(record['weight_int8'].dtype, torch.int8)
                    self.assertEqual(record['bias_int32'].dtype, torch.int32)
                for probe in (x, -x, x*30):
                    with torch.no_grad():
                        torch.testing.assert_close(q(probe), runner.run(probe), rtol=1e-5, atol=1e-6)
                        torch.testing.assert_close(loaded.run(probe), runner.run(probe), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
