"""Activation precision ablation: ranges, gradients, opt1 aliases and export."""
import tempfile
import unittest
from pathlib import Path
import torch
from quantization.qat_model import QATModel, LearnedFakeQuant
from quantization.static_ptq import StaticPTQ, qdq
from quantization.train_qat import build_mobile, ROOT


class W8A16Tests(unittest.TestCase):
    def test_range_and_gradient(self):
        x = torch.tensor([-40000., -128., .7, 128., 40000.], requires_grad=True)
        s = torch.tensor(1., requires_grad=True)
        y = LearnedFakeQuant.apply(x, s, 32767)
        torch.testing.assert_close(y, torch.tensor([-32767., -128., 1., 128., 32767.]))
        y.sum().backward()
        torch.testing.assert_close(x.grad, torch.tensor([0., 1., 1., 1., 0.]))
        self.assertTrue(torch.isfinite(s.grad))
        values = torch.linspace(-1, 1, 1000)
        self.assertLess(float((qdq(values, 1/32767, 32767)-values).square().mean()),
                        float((qdq(values, 1/127)-values).square().mean()) )

    def test_real_models(self):
        torch.set_num_threads(4)
        for model in ('prelu', 'relu'):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as tmp:
                mobile, _, _, _ = build_mobile(model, torch.device('cpu'))
                ptq = torch.load(ROOT/'quantization/artifacts/sid_w8a8_20261031'/(model+'_w8a8.pth'), map_location='cpu')
                q8 = QATModel(mobile, ptq, 'opt1')
                q = QATModel(mobile, ptq, 'opt1', 16)
                self.assertEqual(q.policy, q8.policy)
                self.assertEqual(len(q.activation_log_scales), 38)
                for name in q.activation_log_scales:
                    torch.testing.assert_close(q.activation_scale(name)*32767, q8.activation_scale(name)*127)
                for name in q.weight_log_scales:
                    torch.testing.assert_close(q.weight_scale(name), q8.weight_scale(name), rtol=0, atol=0)
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
                self.assertEqual(loaded.qmax, 32767)
                for record in loaded.integer_weights.values():
                    self.assertEqual(record['weight_int8'].dtype, torch.int8)
                    self.assertEqual(record['bias_int32'].dtype, torch.int32)
                for probe in (x, -x, x*30):
                    with torch.no_grad():
                        torch.testing.assert_close(q(probe), runner.run(probe), rtol=1e-5, atol=1e-6)
                        torch.testing.assert_close(loaded.run(probe), runner.run(probe), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
