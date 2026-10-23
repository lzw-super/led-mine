from types import SimpleNamespace

import numpy as np
import pytest
import torch
from skimage.metrics import structural_similarity

from led.losses import build_loss
from led.models.raw_image_denoising_model import RAWImageDenoisingModel, LEDFinetuneModel
from led.utils.options import yaml_load, _merge_from_base


def test_ssim_matches_independent_reference():
    torch.manual_seed(7)
    x = torch.rand(2, 4, 21, 23, dtype=torch.float64)
    y = (x * .85 + .1 * torch.rand_like(x)).clamp(0, 1)
    loss = build_loss(dict(type='SSIMLoss'))
    expected = np.mean([structural_similarity(a.numpy(), b.numpy(), data_range=1,
                        gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
                        for batch_x, batch_y in zip(x, y) for a, b in zip(batch_x, batch_y)])
    assert loss(x, y).item() == pytest.approx(1 - expected, abs=1e-10)


def test_identity_and_out_of_range_gradients():
    loss = build_loss(dict(type='SSIMLoss', loss_weight=.1))
    for value in [0., .3, 1.]:
        x = torch.full((1, 4, 13, 13), value)
        assert abs(loss(x, x).item()) < 1e-6
    x = torch.full((1, 4, 13, 13), 1.2, requires_grad=True)
    loss(x, torch.ones_like(x)).backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


@pytest.mark.parametrize('model_class', [RAWImageDenoisingModel, LEDFinetuneModel])
@pytest.mark.parametrize('enabled', [False, True])
def test_training_step_uses_optional_ssim(model_class, enabled):
    torch.manual_seed(12)
    net = torch.nn.Conv2d(4, 4, 1)
    optimizer = torch.optim.SGD(net.parameters(), lr=.01)
    state = SimpleNamespace(net_g=net, optimizer_g=optimizer, cur_optimizer=optimizer,
                            align_iter=100, ema_decay=0,
                            lq=torch.rand(1, 4, 16, 16), gt=torch.rand(1, 4, 16, 16),
                            cri_pix=build_loss(dict(type='L1Loss')),
                            cri_ssim=build_loss(dict(type='SSIMLoss', loss_weight=.1)) if enabled else None,
                            reduce_loss_dict=lambda x: x)
    output = net(state.lq)
    expected = state.cri_pix(output, state.gt)
    if enabled:
        expected = expected + state.cri_ssim(output, state.gt)
    grads = torch.autograd.grad(expected, tuple(net.parameters()))
    before = [p.detach().clone() for p in net.parameters()]
    model_class.optimize_parameters(state, 1)
    for parameter, old, grad in zip(net.parameters(), before, grads):
        torch.testing.assert_close(parameter, old - .01 * grad)
    assert ('l_ssim' in state.log_dict) == enabled


def test_ablation_config_preserves_baseline():
    path = 'options/LED/pretrain/CVPR20_ELD_Setting_L1_SSIM.yaml'
    opt = _merge_from_base(yaml_load(path), path)
    assert opt['train']['pixel_opt']['type'] == 'L1Loss'
    assert opt['train']['ssim_opt']['loss_weight'] == .1
    original = 'options/LED/pretrain/CVPR20_ELD_Setting.yaml'
    assert 'ssim_opt' not in _merge_from_base(yaml_load(original), original)['train']


def test_small_crops_rejected():
    with pytest.raises(ValueError, match='Spatial'):
        build_loss(dict(type='SSIMLoss'))(torch.zeros(1, 4, 8, 8), torch.zeros(1, 4, 8, 8))
