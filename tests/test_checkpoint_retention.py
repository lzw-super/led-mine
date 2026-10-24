from types import SimpleNamespace
import torch
from led.utils.checkpoint_retention import CheckpointRetention
from led.utils.options import yaml_load, _merge_from_base


def make_model(tmp_path):
    root=tmp_path/'models';root.mkdir(exist_ok=True)
    states=tmp_path/'states';states.mkdir(exist_ok=True)
    net=torch.nn.Linear(1,1)
    def save(epoch, step):
        torch.save({'params':net.state_dict()},root/f'net_g_{step}.pth')
        torch.save({'epoch':epoch,'iter':step},states/f'{step}.state')
    model=SimpleNamespace(net_g=net,save=save,get_bare_model=lambda n:n,metric_results={})
    opt={'path':{'models':str(root),'training_states':str(states)},
         'logger':{'checkpoint_retention':{'dataset':'val'}}}
    return model,opt,root,states


def test_retains_latest_two_numbered_and_best_across_resume(tmp_path):
    model,opt,root,states=make_model(tmp_path)
    policy=CheckpointRetention(model,opt)
    for step,score in [(1,30),(2,35),(3,32),(4,34)]:
        model.net_g.weight.data.fill_(step)
        policy.save(0,step)
        model.metric_results={'psnr':score}
        policy.consider_best('val',step)
    assert {p.name for p in root.iterdir()}=={'net_g_3.pth','net_g_4.pth','net_g_latest.pth','net_g_best_psnr.pth'}
    assert {p.name for p in states.iterdir()}=={'3.state','4.state'}
    best=torch.load(root/'net_g_best_psnr.pth')
    assert best['selection']['iteration']==2
    assert best['params']['weight'].item()==2
    assert torch.load(root/'net_g_latest.pth')['params']['weight'].item()==4
    resumed=CheckpointRetention(model,opt)
    for score in [34,35,float('nan')]:
        model.metric_results={'psnr':score};resumed.consider_best('val',5)
        assert torch.load(root/'net_g_best_psnr.pth')['selection']['iteration']==2
    model.metric_results={'psnr':36};resumed.consider_best('other',6)
    assert resumed.best==35
    resumed.consider_best('val',6)
    assert torch.load(root/'net_g_best_psnr.pth')['selection']['iteration']==6


def test_scratch_configs():
    configs=[]
    for suffix in ['L1','L1_SSIM']:
        path=f'options/LED/train/Mobile_UNet_Scratch_{suffix}.yaml'
        opt=_merge_from_base(yaml_load(path),path)
        assert opt['path']['pretrain_network_g'] is None
        assert opt['path']['resume_state'] is None
        assert opt['repnr_opt'] is None and 'noise_g' not in opt
        assert opt['train']['optim_g']['lr']==1e-4
        assert opt['train']['total_iter']==257600
        configs.append(opt)
    a,b=configs
    assert b['train'].pop('ssim_opt')['loss_weight']==.3
    for opt in configs:
        opt.pop('base');opt.pop('name')
    assert a==b
