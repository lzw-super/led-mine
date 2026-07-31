"""Benchmark equal-shape SID tiles batched through LED and merged on the GPU."""

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from led.utils.tiled_inference import (  # noqa: E402
    analyze_batched_tiling,
    batched_tiled_inference,
    crop_to_batchable_tiles,
)
from scripts.evaluate_crop_merge_sid import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_DATASET_OPTIONS,
    DEFAULT_NETWORK_OPTIONS,
    _difference_metrics,
    _json_safe,
    build_sid_datasets,
    calculate_psnr,
    illuminance_correct,
    load_led_network,
    profile_compute,
    resolve_checkpoint_path,
)

DEFAULT_RESULTS = ROOT / 'results/crop_merge_sid_gpu_batch'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Crop SID RAW pairs to an aligned equal-tile grid, batch spatial '
            'tiles through LED, and merge crop/Hann/uniform outputs on the GPU.'))
    parser.add_argument('-p', '--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('-opt', '--network-options', type=Path, default=DEFAULT_NETWORK_OPTIONS)
    parser.add_argument('--dataset-options', type=Path, default=DEFAULT_DATASET_OPTIONS)
    parser.add_argument('--dataset-root', type=Path, default=None)
    parser.add_argument('--input-format', choices=['npz', 'raw'], default='raw')
    parser.add_argument('--ratios', nargs='+', type=int, default=[100, 250, 300])
    parser.add_argument('--tile-size', type=int, default=512)
    parser.add_argument('--overlap', type=int, default=192)
    parser.add_argument(
        '--methods',
        nargs='+',
        choices=['crop', 'hann', 'uniform'],
        default=['crop', 'hann', 'uniform'])
    parser.add_argument(
        '--tile-batch-sizes',
        nargs='+',
        type=int,
        default=[1, 2, 4, 8, 0],
        help='Spatial tiles per network launch. Zero means all tiles in one launch.')
    parser.add_argument('--alignment', type=int, default=16)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--max-images-per-ratio', type=int, default=0)
    parser.add_argument('--crop-border', type=int, default=2)
    parser.add_argument('--no-illumination-correct', action='store_true')
    parser.add_argument('--no-compute-profile', action='store_true')
    parser.add_argument('--param-key', default='auto')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_RESULTS)
    return parser.parse_args()


def _timed_gpu_call(function, device: torch.device) -> Tuple[torch.Tensor, float, Optional[float]]:
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


def _full_gpu_inference(
        network: torch.nn.Module,
        image: torch.Tensor,
        device: torch.device,
        use_amp: bool,
) -> torch.Tensor:
    amp_enabled = use_amp and device.type == 'cuda'
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=amp_enabled):
        return network(image.to(device))


def _warm_up(
        network: torch.nn.Module,
        image: torch.Tensor,
        tile_size: int,
        device: torch.device,
        use_amp: bool,
) -> None:
    probe = image[..., :tile_size, :tile_size].to(device)
    amp_enabled = use_amp and device.type == 'cuda'
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=amp_enabled):
        network(probe)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


def _finite_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    values = [value for value in values if value is not None and math.isfinite(value)]
    return sum(values) / len(values) if values else None


def _finite_sum(values: Iterable[Optional[float]]) -> Optional[float]:
    values = [value for value in values if value is not None and math.isfinite(value)]
    return sum(values) if values else None


def _psnr_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    values = [value for value in values if value is not None]
    finite_values = [value for value in values if math.isfinite(value)]
    if finite_values:
        return sum(finite_values) / len(finite_values)
    return math.inf if values else None


def _effective_batch_size(requested: int, tile_count: int) -> int:
    return tile_count if requested == 0 else min(requested, tile_count)


def summarize(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped = defaultdict(list)
    for row in rows:
        key = (row['ratio'], row['method'], row['tile_batch_size'])
        grouped[key].append(row)
        grouped[('all', row['method'], row['tile_batch_size'])].append(row)

    summaries = []
    for (ratio, method, tile_batch_size), group in grouped.items():
        tiled_psnr = _finite_mean(row['tiled_psnr_db'] for row in group)
        full_psnr = _finite_mean(row['full_psnr_db'] for row in group)
        summary = {
            'scope': 'ratio' if ratio != 'all' else 'all',
            'ratio': ratio,
            'method': method,
            'tile_batch_size': tile_batch_size,
            'effective_tile_batch_size': group[0]['effective_tile_batch_size'],
            'image_count': len(group),
            'cropped_height': group[0]['cropped_height'],
            'cropped_width': group[0]['cropped_width'],
            'crop_retained_percent': _finite_mean(
                row['crop_retained_percent'] for row in group),
            'tile_count': group[0]['tile_count'],
            'network_launch_count': group[0]['network_launch_count'],
            'processed_pixel_ratio': group[0]['processed_pixel_ratio'],
            'input_psnr_db': _finite_mean(row['input_psnr_db'] for row in group),
            'full_psnr_db': full_psnr,
            'tiled_psnr_db': tiled_psnr,
            'psnr_change_db': (
                tiled_psnr - full_psnr
                if tiled_psnr is not None and full_psnr is not None else None),
            'psnr_vs_full_db': _psnr_mean(row['psnr_vs_full_db'] for row in group),
            'mae_vs_full': _finite_mean(row['mae_vs_full'] for row in group),
            'max_abs_vs_full': max(row['max_abs_vs_full'] for row in group),
            'estimated_gmacs': group[0]['estimated_gmacs'],
            'full_time_s': _finite_mean(row['full_time_s'] for row in group),
            'tiled_time_s': _finite_mean(row['tiled_time_s'] for row in group),
            'total_full_time_s': _finite_sum(row['full_time_s'] for row in group),
            'total_tiled_time_s': _finite_sum(row['tiled_time_s'] for row in group),
            'data_load_time_s': _finite_mean(row['data_load_time_s'] for row in group),
            'total_data_load_time_s': _finite_sum(
                row['data_load_time_s'] for row in group),
            'total_full_end_to_end_time_s': _finite_sum(
                row['data_load_time_s'] + row['full_time_s'] for row in group),
            'total_tiled_end_to_end_time_s': _finite_sum(
                row['data_load_time_s'] + row['tiled_time_s'] for row in group),
            'time_ratio_vs_full': _finite_mean(
                row['time_ratio_vs_full'] for row in group),
            'speedup_vs_batch1': _finite_mean(
                row['speedup_vs_batch1'] for row in group),
            'throughput_tflops_s': _finite_mean(
                row['throughput_tflops_s'] for row in group),
            'full_peak_memory_mb': _finite_mean(
                row['full_peak_memory_mb'] for row in group),
            'tiled_peak_memory_mb': _finite_mean(
                row['tiled_peak_memory_mb'] for row in group),
        }
        summaries.append(summary)
    return sorted(
        summaries,
        key=lambda row: (
            row['scope'] != 'ratio',
            str(row['ratio']),
            row['method'],
            row['effective_tile_batch_size']))


def write_reports(
        output_dir: Path,
        rows: List[Dict[str, object]],
        summaries: List[Dict[str, object]],
        metadata: Dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'details.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / 'summary.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    with (output_dir / 'report.json').open('w', encoding='utf-8') as file:
        json.dump(
            _json_safe({'metadata': metadata, 'summary': summaries, 'details': rows}),
            file,
            ensure_ascii=False,
            indent=2)


def main() -> None:
    args = parse_args()
    if args.max_images_per_ratio < 0:
        raise ValueError('--max-images-per-ratio cannot be negative.')
    if any(batch_size < 0 for batch_size in args.tile_batch_sizes):
        raise ValueError('--tile-batch-sizes values must be zero or positive.')
    if len(set(args.tile_batch_sizes)) != len(args.tile_batch_sizes):
        raise ValueError('--tile-batch-sizes must not contain duplicates.')

    device = torch.device(args.device)
    if device.type != 'cuda':
        raise ValueError('This benchmark requires a CUDA device.')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable.')
    checkpoint = resolve_checkpoint_path(args.checkpoint)
    network, load_metadata = load_led_network(
        checkpoint,
        args.network_options.expanduser().resolve(),
        args.param_key,
        device)
    parameter_count, macs_per_packed_pixel = profile_compute(
        network, args.no_compute_profile)
    datasets = build_sid_datasets(
        args.dataset_options,
        args.ratios,
        args.dataset_root,
        args.input_format)

    rows: List[Dict[str, object]] = []
    warmed_up = False
    for ratio, dataset in datasets:
        image_count = len(dataset)
        if args.max_images_per_ratio:
            image_count = min(image_count, args.max_images_per_ratio)
        for image_index in range(image_count):
            load_start = time.perf_counter()
            sample = dataset[image_index]
            data_load_time = time.perf_counter() - load_start
            original_image = sample['lq'].unsqueeze(0)
            original_target = sample['gt'].unsqueeze(0)
            image, crop = crop_to_batchable_tiles(
                original_image,
                tile_size=args.tile_size,
                overlap=args.overlap,
                alignment=args.alignment)
            top, bottom, left, right = crop.crop_box
            target = original_target[..., top:bottom, left:right]
            stats = analyze_batched_tiling(
                *image.shape[-2:],
                tile_size=args.tile_size,
                overlap=args.overlap,
                alignment=args.alignment)
            retained_percent = (
                100 * crop.cropped_height * crop.cropped_width
                / (crop.original_height * crop.original_width))
            if not warmed_up:
                _warm_up(network, image, args.tile_size, device, args.amp)
                warmed_up = True

            input_metric = (
                image if args.no_illumination_correct
                else illuminance_correct(image, target))
            input_psnr = calculate_psnr(input_metric, target, args.crop_border)
            full_gpu, full_time, full_peak_memory = _timed_gpu_call(
                lambda: _full_gpu_inference(network, image, device, args.amp),
                device)
            full_output = full_gpu.float().cpu()
            del full_gpu
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            full_metric = (
                full_output if args.no_illumination_correct
                else illuminance_correct(full_output, target))
            full_psnr = calculate_psnr(full_metric, target, args.crop_border)
            estimated_gmacs = (
                macs_per_packed_pixel * stats.processed_pixels / 1e9
                if macs_per_packed_pixel is not None else None)

            print(
                f'[{ratio}x {image_index + 1}/{image_count}] '
                f'{Path(sample["lq_path"]).name} '
                f'original={crop.original_height}x{crop.original_width} '
                f'cropped={crop.cropped_height}x{crop.cropped_width} '
                f'tiles={stats.tile_count} retained={retained_percent:.2f}% '
                f'full={full_time:.3f}s/{full_peak_memory:.1f}MiB',
                flush=True)

            image_rows = []
            for method in args.methods:
                batch1_time = None
                method_rows = []
                for requested_batch_size in args.tile_batch_sizes:
                    effective_batch_size = _effective_batch_size(
                        requested_batch_size, stats.tile_count)
                    output_gpu, tiled_time, tiled_peak_memory = _timed_gpu_call(
                        lambda method=method, requested_batch_size=requested_batch_size:
                        batched_tiled_inference(
                            network,
                            image,
                            tile_size=args.tile_size,
                            overlap=args.overlap,
                            blend=method,
                            alignment=args.alignment,
                            tile_batch_size=requested_batch_size,
                            inference_device=device,
                            accumulation_device=device,
                            use_amp=args.amp),
                        device)
                    tiled_output = output_gpu.float().cpu()
                    del output_gpu
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    tiled_metric = (
                        tiled_output if args.no_illumination_correct
                        else illuminance_correct(tiled_output, target))
                    tiled_psnr = calculate_psnr(
                        tiled_metric, target, args.crop_border)
                    difference = _difference_metrics(
                        tiled_output, full_output, args.crop_border)
                    if effective_batch_size == 1:
                        batch1_time = tiled_time
                    speedup_vs_batch1 = (
                        batch1_time / tiled_time if batch1_time is not None else None)
                    throughput_tflops = (
                        2 * estimated_gmacs / tiled_time / 1000
                        if estimated_gmacs is not None else None)
                    row = {
                        'input_format': args.input_format,
                        'ratio': ratio,
                        'image_index': image_index,
                        'image_name': Path(sample['lq_path']).name,
                        'lq_path': sample['lq_path'],
                        'gt_path': sample['gt_path'],
                        'original_height': crop.original_height,
                        'original_width': crop.original_width,
                        'cropped_height': crop.cropped_height,
                        'cropped_width': crop.cropped_width,
                        'crop_top': top,
                        'crop_bottom': crop.original_height - bottom,
                        'crop_left': left,
                        'crop_right': crop.original_width - right,
                        'crop_retained_percent': retained_percent,
                        'method': method,
                        'tile_size': args.tile_size,
                        'overlap': args.overlap,
                        'stride': args.tile_size - args.overlap,
                        'tile_batch_size': requested_batch_size,
                        'effective_tile_batch_size': effective_batch_size,
                        'tile_count': stats.tile_count,
                        'network_launch_count': math.ceil(
                            stats.tile_count / effective_batch_size),
                        'processed_packed_pixels': stats.processed_pixels,
                        'processed_pixel_ratio': stats.processed_pixel_ratio,
                        'estimated_gmacs': estimated_gmacs,
                        'input_psnr_db': input_psnr,
                        'full_psnr_db': full_psnr,
                        'tiled_psnr_db': tiled_psnr,
                        'psnr_change_db': tiled_psnr - full_psnr,
                        **difference,
                        'data_load_time_s': data_load_time,
                        'full_time_s': full_time,
                        'tiled_time_s': tiled_time,
                        'time_ratio_vs_full': tiled_time / full_time,
                        'speedup_vs_batch1': speedup_vs_batch1,
                        'throughput_tflops_s': throughput_tflops,
                        'full_peak_memory_mb': full_peak_memory,
                        'tiled_peak_memory_mb': tiled_peak_memory,
                    }
                    image_rows.append(row)
                    method_rows.append(row)
                    print(
                        f'  {method:7s} batch={effective_batch_size:2d} '
                        f'launches={row["network_launch_count"]:2d} '
                        f'PSNR={tiled_psnr:.4f} '
                        f'vs_full={difference["psnr_vs_full_db"]:.2f}dB '
                        f'time={tiled_time:.3f}s '
                        f'peak={tiled_peak_memory:.1f}MiB',
                        flush=True)
                for row in method_rows:
                    row['speedup_vs_batch1'] = (
                        batch1_time / row['tiled_time_s']
                        if batch1_time is not None else None)
            rows.extend(image_rows)
            del original_image, original_target, image, target, full_output
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    if not rows:
        raise RuntimeError('No SID samples were evaluated.')
    summaries = summarize(rows)
    metadata = {
        'checkpoint': str(checkpoint),
        'network_options': str(args.network_options.expanduser().resolve()),
        'dataset_options': str(args.dataset_options.expanduser().resolve()),
        'dataset_root': (
            str(args.dataset_root.expanduser().resolve())
            if args.dataset_root else None),
        'input_format': args.input_format,
        'tile_size': args.tile_size,
        'overlap': args.overlap,
        'methods': args.methods,
        'tile_batch_sizes': args.tile_batch_sizes,
        'device': str(device),
        'amp': args.amp,
        'illumination_correct': not args.no_illumination_correct,
        'crop_border': args.crop_border,
        'parameter_count': parameter_count,
        'macs_per_packed_pixel': macs_per_packed_pixel,
        **load_metadata,
    }
    output_dir = args.output_dir.expanduser().resolve()
    write_reports(output_dir, rows, summaries, metadata)

    print('\nOverall summary', flush=True)
    for summary in summaries:
        if summary['scope'] == 'all':
            print(
                f'  {summary["method"]:7s} '
                f'batch={summary["effective_tile_batch_size"]:2d} '
                f'launches={summary["network_launch_count"]:2d} '
                f'PSNR={summary["tiled_psnr_db"]:.4f} '
                f'vs_full={summary["psnr_vs_full_db"]:.2f}dB '
                f'total={summary["total_tiled_time_s"]:.3f}s '
                f'avg/image={summary["tiled_time_s"]:.3f}s '
                f'speedup={summary["speedup_vs_batch1"] or float("nan"):.2f}x '
                f'peak={summary["tiled_peak_memory_mb"]:.1f}MiB',
                flush=True)
    print(f'Reports: {output_dir}', flush=True)


if __name__ == '__main__':
    main()
