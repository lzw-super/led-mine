import tempfile
from pathlib import Path
import unittest
import torch
from torch import nn
from quantization.static_ptq import StaticPTQ
from quantization.qat_model import QATModel,LearnedFakeQuant


class QATTests(unittest.TestCase):
    def test_ste_and_scale_gradients(self):
        x=torch.tensor([-200.,-.7,.7,200.],requires_grad=True)
        s=torch.tensor(1.,requires_grad=True)
        LearnedFakeQuant.apply(x,s).sum().backward()
        torch.testing.assert_close(x.grad,torch.tensor([0.,1.,1.,0.]))
        self.assertTrue(torch.isfinite(s.grad))
        x2=torch.tensor([.7],requires_grad=True);s2=torch.tensor(1.,requires_grad=True)
        LearnedFakeQuant.apply(x2,s2).sum().backward()
        self.assertGreater(float(s2.grad),0.)

    def test_training_export_and_reload(self):
        torch.manual_seed(2)
        model=nn.Sequential(nn.Conv2d(4,4,3,padding=1),nn.PReLU(4),nn.Conv2d(4,4,1)).eval()
        x=torch.randn(1,4,8,8)
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'ptq.pth'
            ptq=StaticPTQ(model)
            with torch.no_grad():ptq.run(x);ptq.freeze();ptq.export(p,{})
            q=QATModel(model,torch.load(p))
            torch.testing.assert_close(q(x),ptq.run(x),rtol=1e-5,atol=1e-6)
            before=q.graph.get_submodule('0').weight.detach().clone()
            optimizer=torch.optim.Adam(q.parameters(),lr=1e-3)
            optimizer.zero_grad();q(x).square().mean().backward()
            self.assertTrue(all(v.grad is not None and torch.isfinite(v.grad).all() for v in q.scale_parameters()))
            optimizer.step();q.clamp_scales()
            self.assertFalse(torch.equal(before,q.graph.get_submodule('0').weight))
            q.freeze_scales();self.assertTrue(all(not v.requires_grad for v in q.scale_parameters()))
            out=Path(directory)/'qat.pth';runner=q.export_qdq(out,{})
            torch.testing.assert_close(q(x),runner.run(x),rtol=1e-5,atol=1e-6)
            loaded=StaticPTQ(model);loaded.load_artifact(out)
            torch.testing.assert_close(loaded.run(x),runner.run(x),rtol=0,atol=0)

if __name__=='__main__':unittest.main()
