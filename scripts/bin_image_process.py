"""Run LED inference on headerless Bayer BIN/RAW files and save RGB PNGs.

Supported formats are a tightly packed 8-bit Bayer frame (RAW8) and standard
MIPI RAW10 (four 10-bit pixels stored in five bytes), plus unpacked 16-bit
little/big-endian Bayer frames.  Such files do not carry camera metadata, so
their geometry, Bayer pattern, black/white levels and display gamma must be
supplied explicitly or accepted from the defaults.
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
    # OpenCV's Bayer conversion constants name the input relative to RGB
    # ordering even when the requested output is BGR.  These mappings produce
    # BGR=[blue, green, red] for the stated sensor CFA (and match CANS).
    'RGGB': cv2.COLOR_BayerBG2BGR,
    'BGGR': cv2.COLOR_BayerRG2BGR,
    'GRBG': cv2.COLOR_BayerGB2BGR,
    'GBRG': cv2.COLOR_BayerGR2BGR,
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
    """Read a headerless Bayer file into the metadata scale selected by the user."""
    buffer = np.fromfile(path, dtype=np.uint8)
    pixel_count = height * width
    format_sizes = {
        'raw8': pixel_count,
        'raw16le': pixel_count * 2,
        'raw16be': pixel_count * 2,
    }
    if pixel_count % 4 == 0:
        format_sizes['mipi_raw10'] = pixel_count * 5 // 4

    if bin_format == 'auto':
        matched_formats = [name for name, size in format_sizes.items() if buffer.size == size]
        # RAW16LE and RAW16BE have the same byte size.  Little endian is the
        # conventional/default choice, while big endian must be requested.
        if 'raw16le' in matched_formats:
            bin_format = 'raw16le'
        elif len(matched_formats) == 1:
            bin_format = matched_formats[0]
        else:
            expected_sizes = ', '.join(
                f'{name} ({size})' for name, size in format_sizes.items())
            raise ValueError(
                f'{path}: {buffer.size} bytes do not match {expected_sizes} '
                f'for {width}x{height}.')

    if bin_format == 'raw8':
        if buffer.size != format_sizes['raw8']:
            raise ValueError(f'{path}: RAW8 needs {format_sizes["raw8"]} bytes, got {buffer.size}.')
        raw = buffer.reshape(height, width).astype(np.float32) * raw8_scale
    elif bin_format == 'mipi_raw10':
        if 'mipi_raw10' not in format_sizes:
            raise ValueError(f'{path}: MIPI RAW10 needs a pixel count divisible by 4.')
        if buffer.size != format_sizes['mipi_raw10']:
            raise ValueError(
                f'{path}: MIPI RAW10 needs {format_sizes["mipi_raw10"]} bytes, '
                f'got {buffer.size}.')
        raw = unpack_mipi_raw10(buffer, height, width).astype(np.float32)
    elif bin_format in ('raw16le', 'raw16be'):
        if buffer.size != format_sizes[bin_format]:
            raise ValueError(
                f'{path}: {bin_format} needs {format_sizes[bin_format]} bytes, '
                f'got {buffer.size}.')
        dtype = '<u2' if bin_format == 'raw16le' else '>u2'
        raw = np.frombuffer(buffer.tobytes(), dtype=dtype).reshape(height, width).astype(np.float32)
    else:
        raise ValueError(f'Unsupported headerless Bayer format: {bin_format}.')

    return raw, bin_format


def parse_extensions(value):
    """Convert a comma-separated extension list to normalized suffixes."""
    extensions = []
    for extension in value.split(','):
        extension = extension.strip().lower().lstrip('.')
        if extension:
            extensions.append(f'.{extension}')
    if not extensions:
        raise ValueError('at least one filename extension is required')
    return frozenset(extensions)


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


def gray_world_gains(packed_raw, clip=8.0):
    """Estimate [R, G1, B, G2] display WB gains from a packed Bayer image."""
    channel_means = np.asarray(packed_raw, dtype=np.float32).reshape(4, -1).mean(axis=1)
    green_mean = max(float((channel_means[1] + channel_means[3]) * 0.5), 1e-6)
    gains = np.array([
        green_mean / max(float(channel_means[0]), 1e-6),
        1.0,
        green_mean / max(float(channel_means[2]), 1e-6),
        1.0,
    ], dtype=np.float32)
    return np.clip(gains, 1.0 / clip, clip)


def apply_ccm_to_bgr(bgr, ccm):
    """Apply a row-major RGB 3x3 color-correction matrix to a BGR image."""
    if ccm is None:
        return bgr
    rgb = bgr[..., ::-1]
    corrected_rgb = np.matmul(rgb, ccm.T)
    return corrected_rgb[..., ::-1]


def result_to_bgr(result, bayer, output_bps, gamma, wb_gains=None, ccm=None):
    """Render packed linear Bayer output using WB, demosaic, optional CCM and gamma."""
    packed = result.numpy()[0].clip(0, 1).astype(np.float32)
    if wb_gains is not None:
        packed = np.clip(packed * wb_gains.reshape(4, 1, 1), 0, 1)
    mosaic = depack_bayer(packed, bayer)
    max_value = 255 if output_bps == 8 else 65535
    mosaic = np.rint(mosaic * max_value).astype(np.uint8 if output_bps == 8 else np.uint16)
    bgr = cv2.cvtColor(mosaic, BAYER_TO_BGR[bayer]).astype(np.float32) / max_value
    bgr = np.clip(apply_ccm_to_bgr(bgr, ccm), 0, 1)
    if gamma > 0:
        bgr = np.power(np.clip(bgr, 0, 1), 1.0 / gamma)
    return np.rint(np.clip(bgr, 0, 1) * max_value).astype(mosaic.dtype)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        description='Run LED inference on headerless RAW8, MIPI RAW10, or RAW16 Bayer files.')
    parser.add_argument('-p', '--pretrained_network', required=True)
    parser.add_argument('--data_path', required=True, help='Directory containing headerless Bayer files.')
    parser.add_argument('--save_path', required=True, help='Directory for output PNG files.')
    parser.add_argument('-opt', '--network_options', default='options/base/network_g/unet.yaml')
    parser.add_argument('--ratio', type=float, default=1.0, help='Additional digital gain before inference.')
    parser.add_argument('--bps', '--output_bps', type=int, choices=(8, 16), default=16)
    parser.add_argument('--led', action='store_true', help='Load params_deploy rather than params.')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--bin_format', choices=('auto', 'raw8', 'mipi_raw10', 'raw16le', 'raw16be'),
                        default='auto')
    parser.add_argument('--extensions', default='bin',
                        help='Comma-separated input extensions without dots; default: %(default)s.')
    parser.add_argument('--bayer', choices=tuple(BAYER_OFFSETS), default='BGGR')
    parser.add_argument('--black_level', type=float, default=64.0)
    parser.add_argument('--white_level', type=float, default=1023.0)
    parser.add_argument('--raw8_scale', type=float, default=4.0,
                        help='Scale RAW8 values into the RAW10-domain metadata scale.')
    parser.add_argument('--isp_wb', choices=('none', 'grayworld', 'manual'), default='none',
                        help='Display white balance for headerless Bayer input; default: %(default)s.')
    parser.add_argument('--wb_gains', type=float, nargs=4, metavar=('R', 'G1', 'B', 'G2'),
                        help='Manual display WB gains in LED packed-channel order; requires --isp_wb manual.')
    parser.add_argument('--wb_clip', type=float, default=8.0,
                        help='Maximum/minimum reciprocal gain allowed by gray-world WB; default: %(default)s.')
    parser.add_argument('--ccm', type=float, nargs=9,
                        metavar=('M00', 'M01', 'M02', 'M10', 'M11', 'M12', 'M20', 'M21', 'M22'),
                        help='Optional row-major RGB 3x3 display color-correction matrix.')
    parser.add_argument('--gamma', type=float, default=2.2,
                        help='Display gamma; set to 0 to save linear demosaicked output.')
    parser.add_argument('--save_comparison', action='store_true',
                        help='Save input RGB on the left and denoised RGB on the right.')
    args = parser.parse_args()

    if args.width <= 0 or args.height <= 0 or args.width % 2 or args.height % 2:
        parser.error('width and height must be positive even numbers.')
    if args.bin_format == 'mipi_raw10' and args.width % 4:
        parser.error('MIPI RAW10 width must be divisible by 4.')
    if args.ratio <= 0:
        parser.error('ratio must be greater than zero.')
    if args.white_level <= args.black_level:
        parser.error('white_level must be greater than black_level.')
    if args.raw8_scale <= 0:
        parser.error('raw8_scale must be greater than zero.')
    if args.gamma < 0:
        parser.error('gamma must be zero or greater.')
    if args.wb_clip < 1:
        parser.error('wb_clip must be at least 1.')
    if args.isp_wb == 'manual' and args.wb_gains is None:
        parser.error('--isp_wb manual requires four --wb_gains values.')

    try:
        extensions = parse_extensions(args.extensions)
    except ValueError as error:
        parser.error(str(error))
    input_paths = sorted(glob.glob(os.path.join(args.data_path, '*')))
    input_paths = [path for path in input_paths
                   if os.path.isfile(path) and Path(path).suffix.lower() in extensions]
    if not input_paths:
        parser.error(f'No {", ".join(sorted(extensions))} files found directly under {args.data_path}.')
    os.makedirs(args.save_path, exist_ok=True)

    print('Building network...')
    network = build_network(yaml_load(args.network_options)['network_g'])
    if args.led and callable(getattr(network, 'switch_to_deploy', None)):
        network.switch_to_deploy()
    load_network(network, args.pretrained_network, param_key='params_deploy' if args.led else 'params')
    device = get_available_device()
    network = network.to(device).eval()

    black_level = torch.full((1, 4, 1, 1), args.black_level, dtype=torch.float32)
    white_level = torch.full((1, 4, 1, 1), args.white_level, dtype=torch.float32)
    manual_wb_gains = (np.asarray(args.wb_gains, dtype=np.float32)
                       if args.wb_gains is not None else None)
    ccm = np.asarray(args.ccm, dtype=np.float32).reshape(3, 3) if args.ccm is not None else None
    print(f'Processing {len(input_paths)} headerless Bayer files as {args.bin_format}, {args.width}x{args.height}, '
          f'{args.bayer}; black/white={args.black_level}/{args.white_level}; '
          f'ISP WB={args.isp_wb}, CCM={"custom" if ccm is not None else "identity"}, gamma={args.gamma}.')

    for path in input_paths:
        raw, detected_format = read_bin(path, args.height, args.width, args.bin_format, args.raw8_scale)
        packed = torch.from_numpy(pack_bayer(raw, args.bayer)).unsqueeze(0).contiguous()
        network_input_cpu = torch.clamp(
            ((packed - black_level) / (white_level - black_level)) * args.ratio, max=1.0)
        if args.isp_wb == 'grayworld':
            wb_gains = gray_world_gains(network_input_cpu.numpy()[0], args.wb_clip)
        elif args.isp_wb == 'manual':
            wb_gains = manual_wb_gains
        else:
            wb_gains = None
        network_input = network_input_cpu.to(device)
        result = torch.clamp(network(network_input), 0, 1).cpu()
        bgr = result_to_bgr(result, args.bayer, args.bps, args.gamma, wb_gains, ccm)
        if args.save_comparison:
            input_bgr = result_to_bgr(network_input_cpu, args.bayer, args.bps, args.gamma, wb_gains, ccm)
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
