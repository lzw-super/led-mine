"""Real model graph checks, encoding inheritance, backward and export parity."""
import tempfile
import unittest
from pathlib import Path
import torch
from quantization.train_qat import build_mobile, ROOT
from quantization.qat_model import QATModel
from quantization.static_ptq import StaticPTQ


class BoundaryTests(unittest.TestCase):
    def test_real_models(self):
        torch.set_num_threads(4)
        for model in ('prelu', 'relu'):
            mobile, _, _, _ = build_mobile(model, torch.device('cpu'))
            artifact_path = ROOT/'quantization/artifacts/sid_w8a8_20261031'/(model+'_w8a8.pth')
            artifact = torch.load(artifact_path, map_location='cpu')
            for profile, count in [('baseline', 53), ('opt1', 38), ('opt2', 47), ('opt12', 32)]:
                with self.subTest(model=model, profile=profile), tempfile.TemporaryDirectory() as tmp:
                    q = QATModel(mobile, artifact, profile)
                    self.assertEqual(len(q.activation_log_scales), count)
                    if profile in ('opt1', 'opt12'):
                        self.assertIs(q.activation_log_scales['raw'], q.activation_log_scales[q.policy['aliases']['conv2d_4']])
                        torch.testing.assert_close(q.activation_scale('conv_transpose2d_3'), q.activation_scale('fusion'), rtol=0, atol=0)
                        self.assertNotIn('hf_blocks_0_reparam_conv', q.activation_log_scales)
                    if profile in ('opt2', 'opt12'):
                        self.assertNotIn('conv2d', q.activation_log_scales)
                        self.assertNotIn('cat_1', q.activation_log_scales)
                        self.assertIn('getitem_1', q.activation_log_scales)
                    x = torch.randn(1, 4, 16, 24)*.03+.04
                    if profile == 'baseline':
                        old = StaticPTQ(mobile); old.load_artifact(artifact_path)
                        torch.testing.assert_close(q(x), old.run(x), rtol=1e-5, atol=1e-6)
                    optimizer = torch.optim.Adam(q.parameters(), lr=1e-5)
                    q(x).square().mean().backward()
                    self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in q.scale_parameters()))
                    optimizer.step(); q.clamp_scales()
                    path = Path(tmp)/'q.pth'
                    runner = q.export_qdq(path, {})
                    loaded = StaticPTQ(mobile); loaded.load_artifact(path)
                    for probe in (x, x*30, -x):
                        with torch.no_grad():
                            torch.testing.assert_close(q(probe), runner.run(probe), rtol=1e-5, atol=1e-6)
                            torch.testing.assert_close(loaded.run(probe), runner.run(probe), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
