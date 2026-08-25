"""Compare full-frame LED inference with memory-bounded crop-merge inference."""

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from led.archs import build_network  # noqa: E402
from led.archs.repnr_utils import build_repnr_arch_from_base  # noqa: E402
from led.data import build_dataset  # noqa: E402
from led.utils.model_complexity import _compute_complexity  # noqa: E402
from led.utils.options import yaml_load  # noqa: E402
from led.utils.tiled_inference import analyze_tiling, tiled_inference  # noqa: E402
from scripts.prepare_led_deploy_checkpoint import (  # noqa: E402
    is_repnr_state_dict,
    select_state_dict,
    strip_thop_profile_buffers,
)

DEFAULT_CHECKPOINT = ROOT / 'experiments/LED_Pretrain_CVPR20_ELD_Setting/models/net_g_latest.pth'
DEFAULT_NETWORK_OPTIONS = ROOT / 'options/base/network_g/repnr_unet.yaml'
DEFAULT_DATASET_OPTIONS = ROOT / 'options/base/dataset/test/SID_SonyA7S2_val_split.yaml'
DEFAULT_RESULTS = ROOT / 'results/crop_merge_sid'
OVERLAP_METHOD_SCAN_OVERLAPS = (128, 160, 192)
OVERLAP_METHOD_SCAN_METHODS = ('crop', 'hann', 'uniform')


def parse_tile_overlap_pair(value: str) -> Tuple[int, int]:
    """Parse a TILE:OVERLAP pair for non-Cartesian configuration scans."""
    try:
        tile_text, overlap_text = value.split(':', maxsplit=1)
        tile_size = int(tile_text)
        overlap = int(overlap_text)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            f'Expected TILE:OVERLAP, but received {value!r}.') from error
    if tile_size <= 0 or overlap < 0 or overlap >= tile_size:
        raise argparse.ArgumentTypeError(
            'TILE:OVERLAP must satisfy TILE > OVERLAP >= 0.')
    return tile_size, overlap


def resolve_checkpoint_path(checkpoint_path: Path) -> Path:
    """Resolve the live checkpoint or its newest experiment archive."""
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if checkpoint_path.is_file():
        return checkpoint_path
    if checkpoint_path == DEFAULT_CHECKPOINT.resolve():
        archived = sorted(
            ROOT.glob(
                'experiments/LED_Pretrain_CVPR20_ELD_Setting_archived_*/'
                'models/net_g_latest.pth'),
            key=lambda path: path.parent.parent.name)
        if archived:
            resolved = archived[-1].resolve()
            print(
                f'Default checkpoint was archived; using {resolved}',
                flush=True)
            return resolved
    raise FileNotFoundError(f'Checkpoint does not exist: {checkpoint_path}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Evaluate full-frame and overlapped crop-merge inference on the SID Sony validation sets. '
            'The reported RAW PSNR follows the LED validation convention (2-pixel crop).'))
    parser.add_argument('-p', '--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('-opt', '--network-options', type=Path, default=DEFAULT_NETWORK_OPTIONS)
    parser.add_argument('--dataset-options', type=Path, default=DEFAULT_DATASET_OPTIONS)
    parser.add_argument(
        '--dataset-root', type=Path, default=None,
        help=(
            'Override datasets.*.dataroot. Use SID/Sony_npy for npz or '
            'SID/Sony for original short/long ARW files.'))
    parser.add_argument(
        '--input-format', choices=['npz', 'raw'], default='npz',
        help='Read cached full-frame NPZ files or decode original short/long RAW pairs.')
    parser.add_argument('--ratios', nargs='+', type=int, default=[100, 250, 300])
    parser.add_argument('--tile-size', type=int, default=512)
    parser.add_argument(
        '--tile-sizes', nargs='+', type=int, default=None,
        help='Evaluate multiple tile sizes. This overrides --tile-size.')
    parser.add_argument(
        '--tile-overlap-pairs',
        nargs='+',
        type=parse_tile_overlap_pair,
        default=None,
        metavar='TILE:OVERLAP',
        help=(
            'Evaluate explicit tile/overlap pairs instead of the Cartesian product '
            'from --tile-sizes and --overlaps, for example 80:24 80:32 100:40.'))
    parser.add_argument(
        '--overlaps', nargs='+', type=int, default=[192],
        help='Shared input pixels. In crop mode, the context halo on each side is overlap/2.')
    parser.add_argument(
        '--methods', nargs='+', choices=['crop', 'hann', 'uniform'], default=['crop'],
        help='crop discards unreliable patch borders; hann/uniform blend complete outputs.')
    parser.add_argument(
        '--scan-overlap-methods', action='store_true',
        help=(
            'Run the 3x3 preset: overlaps 128/160/192 crossed with '
            'crop/hann/uniform. This overrides --overlaps and --methods.'))
    parser.add_argument('--alignment', type=int, default=16, help='Network downsampling factor.')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--amp', action='store_true', help='Use CUDA autocast for both full and tiled inference.')
    parser.add_argument(
        '--max-images-per-ratio', type=int, default=0,
        help='Zero evaluates every image; a positive value is useful for a quick strategy sweep.')
    parser.add_argument('--skip-full', action='store_true', help='Skip full-frame reference inference.')
    parser.add_argument(
        '--no-illumination-correct', action='store_true',
        help='Disable the global illuminance correction used by the LED SID config.')
    parser.add_argument('--crop-border', type=int, default=2)
    parser.add_argument('--param-key', default='auto')
    parser.add_argument(
        '--no-compute-profile', action='store_true',
        help='Do not use THOP to estimate MACs/FLOPs from processed patch pixels.')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_RESULTS)
    return parser.parse_args()


def load_led_network(
        checkpoint_path: Path,
        options_path: Path,
        param_key: str,
        device: torch.device) -> Tuple[torch.nn.Module, Dict[str, object]]:
    """Load a regular, RepNR, or generic reparameterizable deployment network."""
    options = yaml_load(str(options_path))
    if not isinstance(options, Mapping) or 'network_g' not in options:
        raise KeyError(f'network_g is missing from {options_path}.')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f'Checkpoint root must be a mapping: {checkpoint_path}')
    selected_key, state_dict = select_state_dict(checkpoint, param_key)
    state_dict, removed_keys = strip_thop_profile_buffers(state_dict)

    base_network = build_network(deepcopy(options['network_g']))
    if is_repnr_state_dict(state_dict):
        if 'repnr_opt' not in options:
            raise KeyError(f'RepNR checkpoint requires repnr_opt in {options_path}.')
        repnr_network = build_repnr_arch_from_base(
            base_network, **deepcopy(options['repnr_opt']))
        repnr_network.load_state_dict(state_dict, strict=True)
        # RAWBaseModel.nondist_validation() averages the five virtual-camera
        # branches before SID inference. Do the same before fusing RepNR.
        repnr_network.generalize()
        network = repnr_network.eval().deploy()
        mode = 'repnr_generalize_then_deploy'
    elif selected_key.endswith('_deploy'):
        switch_to_deploy = getattr(base_network, 'switch_to_deploy', None)
        if callable(switch_to_deploy):
            switch_to_deploy()
        base_network.load_state_dict(state_dict, strict=True)
        network = base_network
        mode = 'already_deployed'
    else:
        base_network.load_state_dict(state_dict, strict=True)
        deploy = getattr(base_network, 'deploy', None)
        if callable(deploy):
            network = base_network.eval().deploy()
            mode = 'reparameterized_to_deploy'
        else:
            network = base_network
            mode = 'base_to_deploy'

    network = network.eval().to(device)
    return network, {
        'checkpoint_key': selected_key,
        'load_mode': mode,
        'removed_thop_profile_keys': removed_keys,
    }


def build_sid_datasets(
        options_path: Path,
        ratios: Iterable[int],
        dataset_root: Optional[Path],
        input_format: str) -> List[Tuple[int, torch.utils.data.Dataset]]:
    options = yaml_load(str(options_path))
    if not isinstance(options, Mapping) or 'datasets' not in options:
        raise KeyError(f'datasets is missing from {options_path}.')
    datasets = []
    for ratio in ratios:
        key = f'val_{ratio}'
        if key not in options['datasets']:
            raise KeyError(f'{key!r} is not defined in {options_path}.')
        dataset_options = deepcopy(options['datasets'][key])
        dataset_options['phase'] = 'val'
        dataset_options['postfix'] = 'npz' if input_format == 'npz' else None
        if dataset_root is not None:
            dataset_options['dataroot'] = str(dataset_root.expanduser().resolve())
        datasets.append((ratio, build_dataset(dataset_options)))
    return datasets


def illuminance_correct(predict: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Match LED's one-scalar, full-image illuminance correction."""
    if predict.shape[0] != 1:
        return torch.cat(
            [illuminance_correct(predict[index:index + 1], target[index:index + 1])
             for index in range(predict.shape[0])],
            dim=0)
    predict = predict.clamp(0, 1)
    valid = target != 1
    predict_valid = predict[valid]
    target_valid = target[valid]
    denominator = torch.dot(predict_valid, predict_valid)
    if denominator == 0:
        return predict
    return torch.dot(predict_valid, target_valid) / denominator * predict


def calculate_psnr(
        image: torch.Tensor, target: torch.Tensor, crop_border: int = 2) -> float:
    if image.shape != target.shape:
        raise ValueError(f'PSNR shape mismatch: {image.shape} != {target.shape}.')
    if crop_border:
        image = image[..., crop_border:-crop_border, crop_border:-crop_border]
        target = target[..., crop_border:-crop_border, crop_border:-crop_border]
    mse = torch.mean((image.double() - target.double()).square()).item()
    return math.inf if mse == 0 else 10 * math.log10(1.0 / mse)


def _difference_metrics(
        image: torch.Tensor, reference: torch.Tensor, crop_border: int) -> Dict[str, float]:
    if crop_border:
        image = image[..., crop_border:-crop_border, crop_border:-crop_border]
        reference = reference[..., crop_border:-crop_border, crop_border:-crop_border]
    difference = image.double() - reference.double()
    mse = difference.square().mean().item()
    return {
        'psnr_vs_full_db': math.inf if mse == 0 else 10 * math.log10(1.0 / mse),
        'mae_vs_full': difference.abs().mean().item(),
        'max_abs_vs_full': difference.abs().max().item(),
    }


def _timed_call(function, device: torch.device) -> Tuple[torch.Tensor, float, Optional[float]]:
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    result = function()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        peak_memory_mb = None
    return result, time.perf_counter() - start, peak_memory_mb


def _full_inference(
        network: torch.nn.Module,
        image: torch.Tensor,
        device: torch.device,
        use_amp: bool) -> torch.Tensor:
    amp_enabled = use_amp and device.type == 'cuda'
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=amp_enabled):
        return network(image.to(device)).float().cpu()


def profile_compute(
        network: torch.nn.Module,
        disabled: bool) -> Tuple[int, Optional[float]]:
    """Return parameter count and deployment MACs per packed input pixel."""
    parameter_count = sum(parameter.numel() for parameter in network.parameters())
    if disabled:
        return parameter_count, None
    probe_size = 256
    _, probe_macs = _compute_complexity(network, (1, 4, probe_size, probe_size))
    return parameter_count, probe_macs / (probe_size * probe_size)


def _serialize_patch_shapes(patch_shapes: Tuple[Tuple[int, int, int], ...]) -> str:
    return ';'.join(
        f'{height}x{width}:{count}' for height, width, count in patch_shapes)


def _finite_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    finite_values = [value for value in values if value is not None and math.isfinite(value)]
    return sum(finite_values) / len(finite_values) if finite_values else None


SUMMARY_FIELDS = (
    'scope', 'ratio', 'method', 'tile_size', 'overlap', 'image_count',
    'input_psnr_db', 'full_denoise_gain_db', 'tiled_denoise_gain_db',
    'full_psnr_db', 'tiled_psnr_db', 'psnr_change_db', 'psnr_loss_db',
    'psnr_vs_full_db', 'mae_vs_full', 'max_abs_vs_full',
    'tile_count', 'processed_packed_pixels', 'processed_pixel_ratio',
    'full_estimated_gmacs', 'tiled_estimated_gmacs',
    'full_estimated_gflops', 'tiled_estimated_gflops',
    'full_throughput_gflops_s', 'tiled_throughput_gflops_s',
    'full_time_s', 'tiled_time_s', 'full_peak_memory_mb', 'tiled_peak_memory_mb',
    'memory_reduction_percent', 'time_ratio', 'data_load_time_s',
    'full_end_to_end_time_s', 'tiled_end_to_end_time_s',
)


def summarize(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['ratio'], row['method'], row['tile_size'], row['overlap'])].append(row)
        grouped[('all', row['method'], row['tile_size'], row['overlap'])].append(row)

    summaries = []
    for (ratio, method, tile_size, overlap), group in grouped.items():
        full_psnr = _finite_mean(row['full_psnr_db'] for row in group)
        tiled_psnr = _finite_mean(row['tiled_psnr_db'] for row in group)
        psnr_change = _finite_mean(row['psnr_change_db'] for row in group)
        summary = {
            'scope': 'ratio' if ratio != 'all' else 'all',
            'ratio': ratio,
            'method': method,
            'tile_size': tile_size,
            'overlap': overlap,
            'image_count': len(group),
            'input_psnr_db': _finite_mean(row['input_psnr_db'] for row in group),
            'full_denoise_gain_db': _finite_mean(
                row['full_denoise_gain_db'] for row in group),
            'tiled_denoise_gain_db': _finite_mean(
                row['tiled_denoise_gain_db'] for row in group),
            'full_psnr_db': full_psnr,
            'tiled_psnr_db': tiled_psnr,
            'psnr_change_db': psnr_change,
            'psnr_loss_db': -psnr_change if psnr_change is not None else None,
            'psnr_vs_full_db': _finite_mean(row['psnr_vs_full_db'] for row in group),
            'mae_vs_full': _finite_mean(row['mae_vs_full'] for row in group),
            'max_abs_vs_full': max(
                (row['max_abs_vs_full'] for row in group if row['max_abs_vs_full'] is not None),
                default=None),
            'tile_count': _finite_mean(row['tile_count'] for row in group),
            'processed_packed_pixels': _finite_mean(
                row['processed_packed_pixels'] for row in group),
            'processed_pixel_ratio': _finite_mean(
                row['processed_pixel_ratio'] for row in group),
            'full_estimated_gmacs': _finite_mean(
                row['full_estimated_gmacs'] for row in group),
            'tiled_estimated_gmacs': _finite_mean(
                row['tiled_estimated_gmacs'] for row in group),
            'full_estimated_gflops': _finite_mean(
                row['full_estimated_gflops'] for row in group),
            'tiled_estimated_gflops': _finite_mean(
                row['tiled_estimated_gflops'] for row in group),
            'full_throughput_gflops_s': _finite_mean(
                row['full_throughput_gflops_s'] for row in group),
            'tiled_throughput_gflops_s': _finite_mean(
                row['tiled_throughput_gflops_s'] for row in group),
            'full_time_s': _finite_mean(row['full_time_s'] for row in group),
            'tiled_time_s': _finite_mean(row['tiled_time_s'] for row in group),
            'full_peak_memory_mb': _finite_mean(row['full_peak_memory_mb'] for row in group),
            'tiled_peak_memory_mb': _finite_mean(row['tiled_peak_memory_mb'] for row in group),
            'memory_reduction_percent': _finite_mean(
                row['memory_reduction_percent'] for row in group),
            'time_ratio': _finite_mean(row['time_ratio'] for row in group),
            'data_load_time_s': _finite_mean(row['data_load_time_s'] for row in group),
            'full_end_to_end_time_s': _finite_mean(
                row['full_end_to_end_time_s'] for row in group),
            'tiled_end_to_end_time_s': _finite_mean(
                row['tiled_end_to_end_time_s'] for row in group),
        }
        summaries.append(summary)
    return sorted(summaries, key=lambda item: (
        item['scope'] != 'ratio', str(item['ratio']), item['method'],
        item['tile_size'], item['overlap']))


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return 'inf' if value > 0 else '-inf'
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def write_reports(
        output_dir: Path,
        rows: List[Dict[str, object]],
        summaries: List[Dict[str, object]],
        metadata: Dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'details.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / 'summary.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summaries)
    payload = {'metadata': metadata, 'summary': summaries, 'details': rows}
    with (output_dir / 'report.json').open('w', encoding='utf-8') as file:
        json.dump(_json_safe(payload), file, ensure_ascii=False, indent=2)


def _format_number(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return 'n/a'
    if math.isinf(value):
        return 'inf'
    return f'{value:.{digits}f}'


def main() -> None:
    args = parse_args()
    if args.max_images_per_ratio < 0:
        raise ValueError('--max-images-per-ratio cannot be negative.')
    if args.tile_overlap_pairs is not None and args.scan_overlap_methods:
        raise ValueError(
            '--tile-overlap-pairs and --scan-overlap-methods cannot be used together.')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable.')

    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    network, load_metadata = load_led_network(
        checkpoint_path,
        args.network_options.expanduser().resolve(),
        args.param_key,
        device)
    parameter_count, macs_per_packed_pixel = profile_compute(
        network, args.no_compute_profile)
    datasets = build_sid_datasets(
        args.dataset_options, args.ratios, args.dataset_root, args.input_format)
    rows: List[Dict[str, object]] = []
    methods = (
        OVERLAP_METHOD_SCAN_METHODS if args.scan_overlap_methods
        else args.methods)
    if args.tile_overlap_pairs is not None:
        tile_overlap_pairs = list(dict.fromkeys(args.tile_overlap_pairs))
        configurations = [
            (tile_size, method, overlap)
            for tile_size, overlap in tile_overlap_pairs
            for method in methods
        ]
        tile_sizes = list(dict.fromkeys(pair[0] for pair in tile_overlap_pairs))
        overlaps = list(dict.fromkeys(pair[1] for pair in tile_overlap_pairs))
    else:
        tile_sizes = args.tile_sizes if args.tile_sizes is not None else [args.tile_size]
        overlaps = (
            OVERLAP_METHOD_SCAN_OVERLAPS if args.scan_overlap_methods
            else args.overlaps)
        tile_overlap_pairs = None
        configurations = [
            (tile_size, method, overlap)
            for tile_size in tile_sizes
            for method in methods
            for overlap in overlaps
        ]
    print(
        f'Evaluating {len(configurations)} tiled configurations: '
        + ', '.join(
            f'{method}/tile{tile_size}/overlap{overlap}'
            for tile_size, method, overlap in configurations),
        flush=True)

    for ratio, dataset in datasets:
        image_count = len(dataset)
        if args.max_images_per_ratio:
            image_count = min(image_count, args.max_images_per_ratio)
        for index in range(image_count):
            load_start = time.perf_counter()
            sample = dataset[index]
            data_load_time = time.perf_counter() - load_start
            image = sample['lq'].unsqueeze(0)
            target = sample['gt'].unsqueeze(0)
            image_name = Path(sample['lq_path']).name
            input_metric_output = (
                image if args.no_illumination_correct
                else illuminance_correct(image, target))
            input_psnr = calculate_psnr(input_metric_output, target, args.crop_border)
            print(
                f'[{ratio}x {index + 1}/{image_count}] {image_name} '
                f'packed={image.shape[-2]}x{image.shape[-1]} '
                f'sensor={image.shape[-2] * 2}x{image.shape[-1] * 2} '
                f'load={data_load_time:.3f}s',
                flush=True)

            full_output = None
            full_psnr = None
            full_denoise_gain = None
            full_time = None
            full_peak_memory = None
            if not args.skip_full:
                full_output, full_time, full_peak_memory = _timed_call(
                    lambda: _full_inference(network, image, device, args.amp), device)
                full_metric_output = (
                    full_output if args.no_illumination_correct
                    else illuminance_correct(full_output, target))
                full_psnr = calculate_psnr(full_metric_output, target, args.crop_border)
                full_denoise_gain = full_psnr - input_psnr

            for tile_size, method, overlap in configurations:
                tiling_stats = analyze_tiling(
                    image.shape[-2],
                    image.shape[-1],
                    tile_size=tile_size,
                    overlap=overlap,
                    blend=method,
                    alignment=args.alignment)
                tiled_output, tiled_time, tiled_peak_memory = _timed_call(
                    lambda tile_size=tile_size, method=method, overlap=overlap: tiled_inference(
                        network,
                        image,
                        tile_size=tile_size,
                        overlap=overlap,
                        blend=method,
                        alignment=args.alignment,
                        inference_device=device,
                        accumulation_device=torch.device('cpu'),
                        use_amp=args.amp),
                    device)
                tiled_metric_output = (
                    tiled_output if args.no_illumination_correct
                    else illuminance_correct(tiled_output, target))
                tiled_psnr = calculate_psnr(tiled_metric_output, target, args.crop_border)
                tiled_denoise_gain = tiled_psnr - input_psnr
                if full_output is None:
                    difference = {
                        'psnr_vs_full_db': None,
                        'mae_vs_full': None,
                        'max_abs_vs_full': None,
                    }
                    psnr_change = None
                else:
                    difference = _difference_metrics(
                        tiled_output, full_output, args.crop_border)
                    psnr_change = tiled_psnr - full_psnr

                if macs_per_packed_pixel is None:
                    full_gmacs = None
                    tiled_gmacs = None
                    full_gflops = None
                    tiled_gflops = None
                    full_throughput = None
                    tiled_throughput = None
                else:
                    full_gmacs = (
                        macs_per_packed_pixel * tiling_stats.output_pixels / 1e9)
                    tiled_gmacs = (
                        macs_per_packed_pixel * tiling_stats.processed_pixels / 1e9)
                    full_gflops = 2 * full_gmacs
                    tiled_gflops = 2 * tiled_gmacs
                    full_throughput = (
                        full_gflops / full_time if full_time is not None else None)
                    tiled_throughput = tiled_gflops / tiled_time
                memory_reduction = (
                    100 * (1 - tiled_peak_memory / full_peak_memory)
                    if tiled_peak_memory is not None and full_peak_memory is not None
                    else None)
                time_ratio = (
                    tiled_time / full_time if full_time is not None else None)

                row = {
                    'input_format': args.input_format,
                    'ratio': ratio,
                    'image_index': index,
                    'image_name': image_name,
                    'lq_path': sample['lq_path'],
                    'gt_path': sample['gt_path'],
                    'height': image.shape[-2],
                    'width': image.shape[-1],
                    'sensor_height': image.shape[-2] * 2,
                    'sensor_width': image.shape[-1] * 2,
                    'method': method,
                    'tile_size': tile_size,
                    'overlap': overlap,
                    'halo': overlap // 2 if method == 'crop' else None,
                    'input_psnr_db': input_psnr,
                    'full_denoise_gain_db': full_denoise_gain,
                    'tiled_denoise_gain_db': tiled_denoise_gain,
                    'full_psnr_db': full_psnr,
                    'tiled_psnr_db': tiled_psnr,
                    'psnr_change_db': psnr_change,
                    'psnr_loss_db': -psnr_change if psnr_change is not None else None,
                    **difference,
                    'tile_count': tiling_stats.tile_count,
                    'patch_shapes': _serialize_patch_shapes(tiling_stats.patch_shapes),
                    'output_packed_pixels': tiling_stats.output_pixels,
                    'processed_packed_pixels': tiling_stats.processed_pixels,
                    'processed_pixel_ratio': tiling_stats.processed_pixel_ratio,
                    'full_estimated_gmacs': full_gmacs,
                    'tiled_estimated_gmacs': tiled_gmacs,
                    'full_estimated_gflops': full_gflops,
                    'tiled_estimated_gflops': tiled_gflops,
                    'full_throughput_gflops_s': full_throughput,
                    'tiled_throughput_gflops_s': tiled_throughput,
                    'full_time_s': full_time,
                    'tiled_time_s': tiled_time,
                    'data_load_time_s': data_load_time,
                    'full_end_to_end_time_s': (
                        data_load_time + full_time if full_time is not None else None),
                    'tiled_end_to_end_time_s': data_load_time + tiled_time,
                    'time_ratio': time_ratio,
                    'full_peak_memory_mb': full_peak_memory,
                    'tiled_peak_memory_mb': tiled_peak_memory,
                    'memory_reduction_percent': memory_reduction,
                }
                rows.append(row)
                print(
                    f"  {method:7s} tile={tile_size:4d} overlap={overlap:3d} "
                    f"patches={tiling_stats.tile_count:2d} "
                    f"work={tiling_stats.processed_pixel_ratio:.3f}x: "
                    f"input={input_psnr:.4f} dB, "
                    f"full={_format_number(full_psnr)} dB, "
                    f"tiled={_format_number(tiled_psnr)} dB, "
                    f"change={_format_number(psnr_change, 6)} dB, "
                    f"time={tiled_time:.3f}s, "
                    f"memory={_format_number(tiled_peak_memory, 1)} MiB",
                    flush=True)

            del image, target, full_output
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    if not rows:
        raise RuntimeError('No SID samples were evaluated.')
    summaries = summarize(rows)
    metadata = {
        'checkpoint': str(checkpoint_path),
        'network_options': str(args.network_options.expanduser().resolve()),
        'dataset_options': str(args.dataset_options.expanduser().resolve()),
        'dataset_root': str(args.dataset_root.expanduser().resolve()) if args.dataset_root else None,
        'input_format': args.input_format,
        'scan_overlap_methods': args.scan_overlap_methods,
        'tile_sizes': list(tile_sizes),
        'overlaps': list(overlaps),
        'tile_overlap_pairs': (
            [list(pair) for pair in tile_overlap_pairs]
            if tile_overlap_pairs is not None else None),
        'methods': list(methods),
        'configuration_count': len(configurations),
        'device': str(device),
        'amp': args.amp,
        'illumination_correct': not args.no_illumination_correct,
        'crop_border': args.crop_border,
        'parameter_count': parameter_count,
        'macs_per_packed_pixel': macs_per_packed_pixel,
        'flops_per_packed_pixel': (
            2 * macs_per_packed_pixel if macs_per_packed_pixel is not None else None),
        'macs_per_sensor_raw_pixel': (
            macs_per_packed_pixel / 4 if macs_per_packed_pixel is not None else None),
        'flops_per_sensor_raw_pixel': (
            macs_per_packed_pixel / 2 if macs_per_packed_pixel is not None else None),
        **load_metadata,
    }
    output_dir = args.output_dir.expanduser().resolve()
    write_reports(output_dir, rows, summaries, metadata)

    print('\nOverall summary', flush=True)
    for summary in summaries:
        if summary['scope'] != 'all':
            continue
        print(
            f"  {summary['method']:7s} tile={summary['tile_size']} overlap={summary['overlap']}: "
            f"patches={_format_number(summary['tile_count'], 1)}, "
            f"work={_format_number(summary['processed_pixel_ratio'], 3)}x, "
            f"input={_format_number(summary['input_psnr_db'])} dB, "
            f"full={_format_number(summary['full_psnr_db'])} dB, "
            f"tiled={_format_number(summary['tiled_psnr_db'])} dB, "
            f"loss={_format_number(summary['psnr_loss_db'], 6)} dB, "
            f"time={_format_number(summary['tiled_time_s'], 3)}s, "
            f"peak={_format_number(summary['tiled_peak_memory_mb'], 1)} MiB",
            flush=True)
    print(f'Reports: {output_dir}', flush=True)


if __name__ == '__main__':
    main()
