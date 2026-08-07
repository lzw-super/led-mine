import torch

from led.archs import build_network
from led.archs.mrlfn_arch import MRLFN, ReparamConv3x3
from led.losses import build_loss
from led.models.lr_scheduler import CosineAnnealingRestartLR
from led.utils.options import _merge_from_base, yaml_load


def test_reparam_conv_fuses_to_one_equivalent_3x3():
    torch.manual_seed(2026)
    block = ReparamConv3x3(8).eval()
    sample = torch.randn(2, 8, 17, 19)

    with torch.no_grad():
        training_output = block(sample)
        block.switch_to_deploy()
        deployed_output = block(sample)

    assert block.deploy_mode
    assert isinstance(block.reparam_conv, torch.nn.Conv2d)
    assert block.reparam_conv.kernel_size == (3, 3)
    assert not hasattr(block, 'expand_conv')
    torch.testing.assert_close(training_output, deployed_output, rtol=1e-5, atol=2e-6)


def test_mrlfn_deploy_copy_preserves_output_and_loads_strictly():
    torch.manual_seed(2026)
    network = MRLFN(feature_channels=8, num_blocks=2).eval()
    sample = torch.randn(1, 4, 31, 37)

    with torch.no_grad():
        training_output = network(sample)
        deployed = network.deploy().eval()
        deployed_output = deployed(sample)

    # deploy() must not destroy branches used by an ongoing training run.
    assert hasattr(network.blocks[0].conv1, 'expand_conv')
    assert all(module.deploy_mode for module in deployed.modules() if isinstance(module, ReparamConv3x3))
    torch.testing.assert_close(training_output, deployed_output, rtol=1e-5, atol=2e-6)
    assert deployed_output.shape == sample.shape

    fresh_deployed = MRLFN(feature_channels=8, num_blocks=2, deploy=True).eval()
    fresh_deployed.load_state_dict(deployed.state_dict(), strict=True)
    with torch.no_grad():
        reloaded_output = fresh_deployed(sample)
    torch.testing.assert_close(deployed_output, reloaded_output, rtol=0, atol=0)


def test_registry_builds_training_and_deploy_networks():
    options = {
        'type': 'MRLFN',
        'in_channels': 4,
        'out_channels': 4,
        'feature_channels': 8,
        'num_blocks': 1,
        'deploy': False,
    }
    network = build_network(options)
    assert isinstance(network, MRLFN)
    assert network(torch.rand(1, 4, 16, 16)).shape == (1, 4, 16, 16)


def test_raw_reconstruction_chromatic_loss_matches_sid_channel_formula():
    loss = build_loss({
        'type': 'RawReconstructionChromaticLoss',
        'raw_weight': 0.6,
        'chromatic_weight': 0.4,
        'reduction': 'mean',
        'channel_order': [0, 1, 2, 3],
    })
    target = torch.zeros(1, 4, 2, 2)
    pred = torch.zeros_like(target, requires_grad=True)
    pred.data[:, 0] = 1

    # RAW mean is 1/4; only R-G changes, so chromatic loss is 1.
    actual = loss(pred, target)
    expected = torch.tensor(0.6 * 0.25 + 0.4)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert pred.grad is not None


def test_chromatic_loss_ignores_common_brightness_offset():
    loss = build_loss({'type': 'RawReconstructionChromaticLoss'})
    target = torch.zeros(2, 4, 3, 3)
    pred = torch.ones_like(target)
    torch.testing.assert_close(loss(pred, target), torch.tensor(0.6))


def test_sid_training_config_does_not_mix_scheduler_arguments():
    path = 'options/SID/SID_SonyA7S2_CVPR26_MRLFN_Setting.yaml'
    options = _merge_from_base(yaml_load(path), path)
    scheduler_options = dict(options['train']['scheduler'])

    assert scheduler_options == {
        'type': 'CosineAnnealingRestartLR',
        'periods': [257600],
        'restart_weights': [1],
        'eta_min': 1e-7,
    }
    assert options['model_complexity'] == {
        'input_shape': [1, 4, 256, 256],
        'raw_pixel_scale': 2,
    }

    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([parameter], lr=1e-4)
    scheduler_options.pop('type')
    scheduler = CosineAnnealingRestartLR(optimizer, **scheduler_options)
    assert scheduler.get_last_lr() == [1e-4]


def test_led_virtual_camera_training_config_uses_sid_clean_raw():
    path = 'options/LED/pretrain/MRLFN_CVPR20_ELD_Setting.yaml'
    options = _merge_from_base(yaml_load(path), path)

    assert options['datasets']['train']['type'] == 'RAWGTDataset'
    assert options['datasets']['train']['dataroot'] == 'datasets/ICCV23-LED/Sony_train_long_patches'
    assert options['datasets']['train']['ratio_range'] == [100, 300]
    assert options['datasets']['train']['batch_size_per_gpu'] == 1
    assert options['noise_g']['type'] == 'VirtualNoisyPairGenerator'
    assert options['noise_g']['virtual_camera_count'] == 5
    assert options['noise_g']['noise_type'] == 'ptrqc'
    assert options['network_g']['type'] == 'MRLFN'
    assert options['train']['pixel_opt']['type'] == 'RawReconstructionChromaticLoss'
    assert options['train']['scheduler'] == {
        'type': 'HandieLR',
        'milestones': [128800, 231840],
        'lrs': [5e-5, 1e-5],
    }
    assert options['model_complexity'] == {
        'input_shape': [1, 4, 512, 512],
        'raw_pixel_scale': 2,
    }
