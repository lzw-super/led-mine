"""Run LED inference on headerless Bayer BIN files and save RGB PNGs.

Supported formats are a tightly packed 8-bit Bayer frame (RAW8) and standard
MIPI RAW10 (four 10-bit pixels stored in five bytes).  BIN files do not carry
camera metadata, so their geometry, Bayer pattern, black/white levels and
display gamma must be supplied explicitly or accepted from the defaults.
"""

import argparse
import glob
import os
import sys
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from led.archs import build_network
from led.utils.options import yaml_load


BAYER_OFFSETS = {
    # Output channel order is always [R, G1, B, G2], matching LED packed RAW.
    'RGGB': ((0, 0), (0, 1), (1, 1), (1, 0)),
    'BGGR': ((1, 1), (0, 1), (0, 0), (1, 0)),
    'GRBG': ((0, 1), (0, 0), (1, 0), (1, 1)),
    'GBRG': ((1, 0), (0, 0), (0, 1), (1, 1)),
}
BAYER_TO_BGR = {
    'RGGB': cv2.COLOR_BayerRG2BGR,
    'BGGR': cv2.COLOR_BayerBG2BGR,
    'GRBG': cv2.COLOR_BayerGR2BGR,
    'GBRG': cv2.COLOR_BayerGB2BGR,
}


def load_network(net, load_path, strict=True, param_key='params'):
    load_net = torch.load(load_path, map_location='cpu')
    if param_key is not None:
        if param_key not in load_net and 'params' in load_net:
            param_key = 'params'
            print('Loading: requested parameter key does not exist; using params.')
        load_net = load_net[param_key]
    print(f'Loading {net.__class__.__name__} from {load_path}, parameter key: [{param_key}].')
    for key, value in deepcopy(load_net).items():
        if key.startswith('module.'):
            load_net[key[7:]] = value
            load_net.pop(key)
    net.load_state_dict(load_net, strict=strict)


def get_available_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def unpack_mipi_raw10(buffer, height, width):
    """Decode standard MIPI RAW10: every five bytes contain four pixels."""
    packed = buffer.reshape(-1, 5).astype(np.uint16)
    b0, b1, b2, b3, b4 = (packed[:, index] for index in range(5))
    pixels = np.empty(height * width, dtype=np.uint16)
    pixels[0::4] = (b0 << 2) | (b4 & 0x03)
    pixels[1::4] = (b1 << 2) | ((b4 >> 2) & 0x03)
    pixels[2::4] = (b2 << 2) | ((b4 >> 4) & 0x03)
    pixels[3::4] = (b3 << 2) | ((b4 >> 6) & 0x03)
    return pixels.reshape(height, width)


def read_bin(path, height, width, bin_format, raw8_scale):
    """Read a RAW8 or MIPI RAW10 Bayer frame into the common 10-bit domain."""
    buffer = np.fromfile(path, dtype=np.uint8)
    raw8_size = height * width
    raw10_size = raw8_size * 5 // 4

    if bin_format == 'auto':
        if buffer.size == raw8_size:
            bin_format = 'raw8'
        elif buffer.size == raw10_size:
            bin_format = 'mipi_raw10'
        else:
            raise ValueError(
                f'{path}: {buffer.size} bytes do not match RAW8 ({raw8_size}) '
                f'or MIPI RAW10 ({raw10_size}) for {width}x{height}.')

    if bin_format == 'raw8':
        if buffer.size != raw8_size:
            raise ValueError(f'{path}: RAW8 needs {raw8_size} bytes, got {buffer.size}.')
        raw = buffer.reshape(height, width).astype(np.float32) * raw8_scale
    else:
        if buffer.size != raw10_size:
            raise ValueError(f'{path}: MIPI RAW10 needs {raw10_size} bytes, got {buffer.size}.')
        raw = unpack_mipi_raw10(buffer, height, width).astype(np.float32)

    return raw, bin_format


def pack_bayer(raw, bayer):
    """Pack a 2-D Bayer mosaic into LED's [R, G1, B, G2] layout."""
    if raw.shape[0] % 2 or raw.shape[1] % 2:
        raise ValueError(f'Bayer dimensions must be even, got {raw.shape}.')
    return np.stack([raw[row::2, col::2] for row, col in BAYER_OFFSETS[bayer]], axis=0)


def depack_bayer(packed, bayer):
    """Restore a 2-D Bayer mosaic from LED's [R, G1, B, G2] layout."""
    _, height, width = packed.shape
    raw = np.empty((height * 2, width * 2), dtype=packed.dtype)
    for channel, (row, col) in enumerate(BAYER_OFFSETS[bayer]):
        raw[row::2, col::2] = packed[channel]
    return raw


def result_to_bgr(result, bayer, output_bps, gamma):
    """Demosaic normalized model output for display without camera metadata."""
    mosaic = depack_bayer(result.numpy()[0], bayer).clip(0, 1)
    max_value = 255 if output_bps == 8 else 65535
    mosaic = np.rint(mosaic * max_value).astype(np.uint8 if output_bps == 8 else np.uint16)
    bgr = cv2.cvtColor(mosaic, BAYER_TO_BGR[bayer]).astype(np.float32) / max_value
    if gamma > 0:
        bgr = np.power(np.clip(bgr, 0, 1), 1.0 / gamma)
    return np.rint(np.clip(bgr, 0, 1) * max_value).astype(mosaic.dtype)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        description='Run LED inference on headerless RAW8 or MIPI RAW10 Bayer BIN files.')
    parser.add_argument('-p', '--pretrained_network', required=True)
    parser.add_argument('--data_path', required=True, help='Directory containing BIN files.')
    parser.add_argument('--save_path', required=True, help='Directory for output PNG files.')
    parser.add_argument('-opt', '--network_options', default='options/base/network_g/unet.yaml')
    parser.add_argument('--ratio', type=float, default=1.0, help='Additional digital gain before inference.')
    parser.add_argument('--bps', '--output_bps', type=int, choices=(8, 16), default=16)
    parser.add_argument('--led', action='store_true', help='Load params_deploy rather than params.')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--bin_format', choices=('auto', 'raw8', 'mipi_raw10'), default='auto')
    parser.add_argument('--bayer', choices=tuple(BAYER_OFFSETS), default='BGGR')
    parser.add_argument('--black_level', type=float, default=64.0)
    parser.add_argument('--white_level', type=float, default=1023.0)
    parser.add_argument('--raw8_scale', type=float, default=4.0,
                        help='Scale RAW8 values into the RAW10-domain metadata scale.')
    parser.add_argument('--gamma', type=float, default=2.2,
                        help='Display gamma; set to 0 to save linear demosaicked output.')
    parser.add_argument('--save_comparison', action='store_true',
                        help='Save input RGB on the left and denoised RGB on the right.')
    args = parser.parse_args()

    if args.width <= 0 or args.height <= 0 or args.width % 4 or args.height % 2:
        parser.error('width must be positive and divisible by 4; height must be positive and even.')
    if args.ratio <= 0:
        parser.error('ratio must be greater than zero.')
    if args.white_level <= args.black_level:
        parser.error('white_level must be greater than black_level.')
    if args.raw8_scale <= 0:
        parser.error('raw8_scale must be greater than zero.')
    if args.gamma < 0:
        parser.error('gamma must be zero or greater.')

    bin_paths = sorted(glob.glob(os.path.join(args.data_path, '*')))
    bin_paths = [path for path in bin_paths if os.path.isfile(path) and path.lower().endswith('.bin')]
    if not bin_paths:
        parser.error(f'No .bin files found directly under {args.data_path}.')
    os.makedirs(args.save_path, exist_ok=True)

    print('Building network...')
    network = build_network(yaml_load(args.network_options)['network_g'])
    load_network(network, args.pretrained_network, param_key='params_deploy' if args.led else 'params')
    device = get_available_device()
    network = network.to(device).eval()

    black_level = torch.full((1, 4, 1, 1), args.black_level, dtype=torch.float32)
    white_level = torch.full((1, 4, 1, 1), args.white_level, dtype=torch.float32)
    print(f'Processing {len(bin_paths)} BIN files as {args.bin_format}, {args.width}x{args.height}, '
          f'{args.bayer}; black/white={args.black_level}/{args.white_level}.')

    for path in bin_paths:
        raw, detected_format = read_bin(path, args.height, args.width, args.bin_format, args.raw8_scale)
        packed = torch.from_numpy(pack_bayer(raw, args.bayer)).unsqueeze(0).contiguous()
        network_input = (packed - black_level) / (white_level - black_level)
        network_input = torch.clamp(network_input * args.ratio, max=1.0).to(device)
        result = torch.clamp(network(network_input), 0, 1).cpu()
        bgr = result_to_bgr(result, args.bayer, args.bps, args.gamma)
        if args.save_comparison:
            input_bgr = result_to_bgr(network_input.cpu().clip(0, 1), args.bayer, args.bps, args.gamma)
            bgr = cv2.hconcat([input_bgr, bgr])
            output_name = os.path.basename(path) + '.comparison.png'
        else:
            output_name = os.path.basename(path) + '.png'
        output_path = os.path.join(args.save_path, output_name)
        if not cv2.imwrite(output_path, bgr):
            raise OSError(f'Failed to write {output_path}.')
        print(f'{os.path.basename(path)} ({detected_format}) -> {output_path}')


if __name__ == '__main__':
    main()
