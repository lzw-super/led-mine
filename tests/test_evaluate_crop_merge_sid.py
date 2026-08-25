import torch

from led.archs.mrlfn_arch import MRLFN, ReparamConv3x3
from led.utils.tiled_inference import tiled_inference
from scripts.evaluate_crop_merge_sid import load_led_network, parse_tile_overlap_pair


def test_parse_tile_overlap_pair():
    assert parse_tile_overlap_pair('80:24') == (80, 24)


def test_loader_converts_mrlfn_training_checkpoint_to_deployment_graph(tmp_path):
    torch.manual_seed(2026)
    training_network = MRLFN(feature_channels=8, num_blocks=2).eval()
    checkpoint_path = tmp_path / 'mrlfn_training.pth'
    options_path = tmp_path / 'mrlfn.yaml'
    torch.save({'params': training_network.state_dict()}, checkpoint_path)
    options_path.write_text(
        'network_g:\n'
        '  type: MRLFN\n'
        '  in_channels: 4\n'
        '  out_channels: 4\n'
        '  feature_channels: 8\n'
        '  num_blocks: 2\n'
        '  bias: true\n'
        '  deploy: false\n',
        encoding='utf-8')

    deployed_network, metadata = load_led_network(
        checkpoint_path, options_path, 'params', torch.device('cpu'))
    sample = torch.rand(1, 4, 31, 37)
    with torch.inference_mode():
        expected = training_network(sample)
        actual = deployed_network(sample)

    assert metadata['load_mode'] == 'reparameterized_to_deploy'
    assert all(
        module.deploy_mode
        for module in deployed_network.modules()
        if isinstance(module, ReparamConv3x3))
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)

    tiled = tiled_inference(
        deployed_network,
        sample,
        tile_size=24,
        overlap=18,
        blend='crop',
        alignment=1)
    torch.testing.assert_close(tiled, actual, rtol=1e-5, atol=2e-6)
