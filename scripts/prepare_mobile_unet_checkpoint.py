"""Average LED camera alignments, fuse RepNR and verify the plain UNet initialization."""
import argparse
from pathlib import Path

import torch
from led.archs import build_network
from led.archs.repnr_utils import build_repnr_arch_from_base
from led.utils.options import yaml_load
from scripts.prepare_led_deploy_checkpoint import select_state_dict, strip_thop_profile_buffers


def prepare(source, output):
    if Path(source).resolve() == Path(output).resolve():
        raise ValueError('Source checkpoint must be preserved.')
    torch.set_num_threads(4)
    options = yaml_load('options/base/network_g/repnr_unet.yaml')
    net = build_repnr_arch_from_base(build_network(options['network_g']), **options['repnr_opt'])
    _, state = select_state_dict(torch.load(source, map_location='cpu', weights_only=False), 'params')
    state, _ = strip_thop_profile_buffers(state)
    net.load_state_dict(state, strict=True)
    net.generalize()
    net.eval()
    sample = torch.rand(1, 4, 64, 64, generator=torch.Generator().manual_seed(2022))
    with torch.no_grad():
        expected = net(sample)
        plain = net.deploy().eval()
        actual = plain(sample)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    error = (expected - actual).abs().max().item()
    torch.save(dict(params=plain.state_dict(), source_checkpoint=str(Path(source).resolve()),
                    initialization='mean of 5 LED alignment branches, fused into plain UNet',
                    fusion_max_abs_error=error), output)
    print(f'Saved {output}; fusion max abs error = {error:.8g}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='experiments/LED_Pretrain_CVPR20_ELD_Setting_archived_20260726_230625/models/net_g_latest.pth')
    parser.add_argument('--output', default='pretrained/mobile/led_generalized_unet.pth')
    args = parser.parse_args()
    prepare(args.source, args.output)
