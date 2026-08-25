"""Profile MRLFN depth/width choices for resource-constrained deployment.

The default configurations form three roughly equal deployment-parameter
groups around 100K, 125K, and 175K.  This makes the cost of trading channel
width ``d`` for block depth ``N`` visible without conflating it with a large
change in model capacity.
"""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import torch
import torch.utils.benchmark

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from led.archs.mrlfn_arch import MRLFN  # noqa: E402
from led.utils.model_complexity import _compute_complexity  # noqa: E402

DEFAULT_CONFIGS = (
    (3, 32),
    (6, 24),
    (14, 16),
    (4, 32),
    (8, 24),
    (16, 16),
    (6, 32),
    (10, 24),
    (24, 16),
)


def parse_config(value):
    """Parse a configuration written as ``Nxd``, for example ``4x32``."""
    try:
        num_blocks, feature_channels = (int(item) for item in value.lower().split('x', 1))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(f'Expected Nxd, for example 4x32, but got {value!r}.') from error
    if num_blocks <= 0 or feature_channels <= 0:
        raise argparse.ArgumentTypeError('N and d must both be positive.')
    return num_blocks, feature_channels


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--configs',
        nargs='+',
        type=parse_config,
        default=DEFAULT_CONFIGS,
        help='MRLFN Nxd pairs. Defaults to equal-budget 100K/125K/175K groups.')
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dtypes', nargs='+', choices=('fp32', 'fp16'), default=('fp32', 'fp16'))
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--repeats', type=int, default=100)
    parser.add_argument(
        '--no-latency', action='store_true', help='Only report parameters, THOP MACs, receptive field, and storage.')
    parser.add_argument('--json', action='store_true', help='Print JSON instead of a Markdown table.')
    return parser.parse_args()


def percentile(sorted_values, fraction):
    index = max(0, math.ceil(fraction * len(sorted_values)) - 1)
    return sorted_values[index]


def benchmark_cuda(model, sample, warmup, repeats):
    device = sample.device
    with torch.inference_mode():
        for _ in range(warmup):
            output = model(sample)
        torch.cuda.synchronize(device)
        del output

        torch.cuda.reset_peak_memory_stats(device)
        baseline_memory = torch.cuda.memory_allocated(device)
        output = model(sample)
        torch.cuda.synchronize(device)
        peak_delta = torch.cuda.max_memory_allocated(device) - baseline_memory
        del output

        timings = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = model(sample)
            end.record()
            end.synchronize()
            timings.append(start.elapsed_time(end))
        del output

    timings.sort()
    return {
        'median_ms': statistics.median(timings),
        'p90_ms': percentile(timings, 0.9),
        'peak_transient_mib': peak_delta / (1024**2),
    }


def benchmark_cpu(model, sample, warmup, repeats):
    timer = torch.utils.benchmark.Timer(stmt='model(sample)', globals={'model': model, 'sample': sample})
    with torch.inference_mode():
        for _ in range(warmup):
            model(sample)
        measurement = timer.timeit(repeats)
    return {'median_ms': measurement.median * 1000, 'p90_ms': None, 'peak_transient_mib': None}


def profile_config(num_blocks, feature_channels, args):
    training_model = MRLFN(feature_channels=feature_channels, num_blocks=num_blocks, deploy=False).eval()
    deployed_model = training_model.deploy().eval()
    train_parameters = sum(parameter.numel() for parameter in training_model.parameters())
    deploy_parameters, macs = _compute_complexity(deployed_model, (1, 4, args.height, args.width))

    row = {
        'N': num_blocks,
        'd': feature_channels,
        'train_parameters': train_parameters,
        'deploy_parameters': deploy_parameters,
        'gmacs': macs / 1e9,
        'receptive_field': 6 * num_blocks + 5,
        'spatial_conv_layers': 3 * num_blocks + 2,
        'activation_traffic_proxy_Nd': num_blocks * feature_channels,
        'int16_weight_kib': deploy_parameters * 2 / 1024,
        'int16_one_feature_mib': (args.height * args.width * feature_channels * 2 / (1024**2)),
    }

    if args.no_latency:
        return row

    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable.')
    if device.type == 'cpu' and 'fp16' in args.dtypes:
        raise ValueError('CPU Conv2d does not provide a representative fp16 benchmark; use --dtypes fp32.')

    del training_model, deployed_model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    for dtype_name in args.dtypes:
        dtype = torch.float32 if dtype_name == 'fp32' else torch.float16
        model = MRLFN(
            feature_channels=feature_channels,
            num_blocks=num_blocks,
            deploy=True,
        ).eval()
        model = model.to(device=device, dtype=dtype)
        sample = torch.randn(1, 4, args.height, args.width, device=device, dtype=dtype)
        if device.type == 'cuda':
            metrics = benchmark_cuda(model, sample, args.warmup, args.repeats)
        else:
            metrics = benchmark_cpu(model, sample, args.warmup, args.repeats)
        for key, value in metrics.items():
            row[f'{dtype_name}_{key}'] = value
        del model, sample
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return row


def format_optional(value, digits=3):
    return '-' if value is None else f'{value:.{digits}f}'


def print_markdown(rows, dtypes):
    latency_columns = []
    for dtype_name in dtypes:
        latency_columns.extend((
            f'{dtype_name} median ms',
            f'{dtype_name} peak MiB',
        ))
    headers = [
        'N',
        'd',
        'deploy params',
        'GMACs',
        'RF',
        'INT16 weights KiB',
        'INT16 feature MiB',
        'N×d traffic proxy',
    ] + latency_columns
    print('| ' + ' | '.join(headers) + ' |')
    print('| ' + ' | '.join(['---:'] * len(headers)) + ' |')
    for row in rows:
        values = [
            str(row['N']),
            str(row['d']),
            f"{row['deploy_parameters']:,}",
            f"{row['gmacs']:.3f}",
            f"{row['receptive_field']}×{row['receptive_field']}",
            f"{row['int16_weight_kib']:.1f}",
            f"{row['int16_one_feature_mib']:.1f}",
            str(row['activation_traffic_proxy_Nd']),
        ]
        for dtype_name in dtypes:
            values.extend((
                format_optional(row.get(f'{dtype_name}_median_ms')),
                format_optional(row.get(f'{dtype_name}_peak_transient_mib'), 1),
            ))
        print('| ' + ' | '.join(values) + ' |')


def main():
    args = parse_args()
    if args.height <= 0 or args.width <= 0:
        raise ValueError('--height and --width must be positive.')
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError('--warmup must be non-negative and --repeats must be positive.')

    torch.manual_seed(2026)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    rows = [profile_config(num_blocks, feature_channels, args) for num_blocks, feature_channels in args.configs]
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print_markdown(rows, () if args.no_latency else args.dtypes)


if __name__ == '__main__':
    main()
