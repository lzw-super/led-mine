import tempfile
from pathlib import Path
import unittest
import torch
from torch import nn
from quantization.static_ptq import StaticPTQ, qdq


class PTQTests(unittest.TestCase):
    def test_symmetric_rounding_and_saturation(self):
        x=torch.tensor([-200.,-1.5,-.5,0.,.5,1.5,200.])
        torch.testing.assert_close(qdq(x,torch.tensor(1.)),torch.tensor([-127.,-2.,0.,0.,0.,2.,127.]))

    def test_channel_scales_freeze_and_roundtrip(self):
        torch.manual_seed(4)
        model=nn.Sequential(nn.Conv2d(2,2,1),nn.ReLU()).eval()
        with torch.no_grad():
            model[0].weight[0]*=.01
        q=StaticPTQ(model)
        x=torch.randn(1,2,8,8)
        with torch.inference_mode():
            q.run(x);q.freeze();expected=q.run(x)
            scales={k:v.clone() for k,v in q.scales.items()}
            q.run(x*100)
            for k,v in scales.items():torch.testing.assert_close(v,q.scales[k])
            record=q.integer_weights['0']
            self.assertEqual(record['weight_int8'].dtype,torch.int8)
            self.assertEqual(record['bias_int32'].dtype,torch.int32)
            self.assertNotEqual(float(record['weight_scale'][0]),float(record['weight_scale'][1]))
            self.assertTrue(all(float(v)>0 for v in scales.values()))
            with tempfile.TemporaryDirectory() as d:
                p=Path(d)/'q.pth';q.export(p,{'test':True})
                loaded=StaticPTQ(model);loaded.load_artifact(p)
                torch.testing.assert_close(loaded.run(x),expected,rtol=0,atol=0)

if __name__=='__main__':unittest.main()
