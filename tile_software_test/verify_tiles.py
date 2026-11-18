"""Small FP32 numerical check; does not replace full-dataset/FPGA validation."""
import hashlib
import io
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
import yaml
from led.archs.sid_reference_arch import SIDReferenceDenoiser

OUT = Path(__file__).resolve().parent
EXP = ROOT/'experiments/LED_Pretrain_CVPR20_ELD_LearningDWT_Haar_L3_RepNCB_W32_ReLU_LLStemNoAct'
torch.set_num_threads(4)
torch.manual_seed(2026)
options = yaml.safe_load(next(EXP.glob('*.yaml')).read_text())['network_g'].copy()
options.pop('type')
model = SIDReferenceDenoiser(**options).eval()
blob = (EXP/'models/net_g_255000.pth').read_bytes()
weights = torch.load(io.BytesIO(blob),map_location='cpu')['params']
removed = [k for k in weights if k.endswith(('total_ops','total_params'))]
model.load_state_dict({k:v for k,v in weights.items() if k not in removed},strict=True)
results = []
with torch.inference_mode():
    for deploy in [False,True]:
        net = model.deploy() if deploy else model
        for height,width in [(192,256),(197,259)]:
            x = torch.rand(1,4,height,width)
            ref = net(x)
            for halo in [48,56,64]:
                y = torch.empty_like(x)
                for top in range(0,height,64):
                    bottom = min(top+64,height)
                    for left in range(0,width,64):
                        right = min(left+64,width)
                        a,b = max(0,top-halo),min(height,bottom+halo)
                        c,d = max(0,left-halo),min(width,right+halo)
                        patch = net(x[...,a:b,c:d])
                        y[...,top:bottom,left:right] = patch[...,top-a:bottom-a,left-c:right-c]
                diff = (y-ref).abs()
                passed = torch.allclose(y,ref,rtol=1e-5,atol=1e-6)
                results.append(dict(deploy=deploy,hw=[height,width],core=64,halo=halo,max_abs=diff.max().item(),mean_abs=diff.mean().item(),allclose=passed))
                if halo >= 56:
                    assert passed, results[-1]
report = dict(checkpoint='net_g_255000.pth',sha256=hashlib.sha256(blob).hexdigest(),torch=torch.__version__,device='cpu',dtype='float32',seed=2026,removed_profile_buffers=removed,results=results)
(OUT/'numerical_check.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
