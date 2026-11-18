"""New checkpoint topology and the explicit pre-S2D A8 cast."""
import tempfile
import unittest
from pathlib import Path
import torch
from torch import nn, fx
from quantization.train_qat import build_mobile
from quantization.qat_model import QATModel
from quantization.static_ptq import StaticPTQ


class LLStemNoActTests(unittest.TestCase):
    def test_model_and_three_precisions(self):
        torch.set_num_threads(4)
        mobile, _, _, _ = build_mobile('relu_llstem_noact', torch.device('cpu'))
        self.assertIsInstance(mobile.ll_layers[1], nn.Identity)
        self.assertFalse(any(isinstance(m,nn.PReLU) for m in mobile.modules()))
        x = torch.randn(1,4,16,24)*.05+.05
        with tempfile.TemporaryDirectory() as directory:
            ptq = StaticPTQ(mobile)
            with torch.no_grad():ptq.run(x);ptq.freeze()
            seed_path = Path(directory)/'seed.pth';ptq.export(seed_path,{})
            artifact = torch.load(seed_path)
            for bits, profile in [(8,'none'),(16,'none'),(16,'phase_exp3_pres2d')]:
                with self.subTest(bits=bits,profile=profile):
                    q = QATModel(mobile,artifact,'opt1',bits,profile)
                    self.assertEqual(q.policy['aliases']['ll_layers_1'],'ll_layers_0')
                    self.assertNotIn('ll_layers_0', q.policy['removed'])
                    self.assertEqual(len(q.activation_log_scales),39 if profile!='none' else 38)
                    if profile != 'none':
                        nodes = {n.name:n for n in q.graph.graph.nodes}
                        self.assertEqual(nodes['phase2_input_a8'].args[0].name,'conv_transpose2d_2')
                        self.assertEqual(nodes['conv2d_3'].args[0].name,'phase2_input_a8')
                        self.assertEqual(q.node_qmax('conv_transpose2d_2'),32767)
                        for name in ['phase2_input_a8','conv2d_3','stem','blocks_4_activation']:
                            self.assertEqual(q.node_qmax(name),127)
                        for name in ['conv2d_4','head','cat_4','fusion','conv_transpose2d_3']:
                            self.assertEqual(q.node_qmax(name),32767)
                        self.assertEqual(q.policy['aliases']['conv2d_3'],'phase2_input_a8')
                        # Moving a tensor-wise quantizer through a permutation must
                        # preserve initial numerical results given identical encodings.
                        old = QATModel(mobile,artifact,'opt1',16,'phase_exp3')
                        torch.testing.assert_close(q(x),old(x),rtol=0,atol=0)
                    with torch.no_grad():
                        torch.testing.assert_close(q.graph(x),mobile(x),rtol=0,atol=0)
                    opt = torch.optim.Adam(q.parameters(),lr=1e-5)
                    q(x).square().mean().backward()
                    self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in q.scale_parameters()))
                    opt.step();q.clamp_scales()
                    out=Path(directory)/'export.pth';runner=q.export_qdq(out,{})
                    loaded=StaticPTQ(mobile);loaded.load_artifact(out)
                    for probe in (x,-x,x*30):
                        with torch.no_grad():
                            torch.testing.assert_close(q(probe),runner.run(probe),rtol=1e-5,atol=1e-6)
                            torch.testing.assert_close(loaded.run(probe),runner.run(probe),rtol=0,atol=0)
                    if profile!='none':
                        # Serialized data itself proves the physical cast ordering.
                        self.assertIn('phase2_input_a8',torch.load(out)['graph'])
                        loaded.load_artifact(seed_path)
                        torch.testing.assert_close(loaded.run(x),ptq.run(x),rtol=0,atol=0)


if __name__=='__main__':unittest.main()
