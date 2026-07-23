"""Memory-bounded tiled inference for spatially aligned image restoration.

The default ``crop`` mode implements overlap-crop (also called halo exchange):
neighbouring input patches overlap, but only the reliable centre of every
network output is copied to the final image.  ``hann`` and ``uniform`` modes
instead overlap-add complete patch outputs with normalized weights.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class Tile:
    """Coordinates of one input tile and the output region retained from it."""

    input_box: Tuple[int, int, int, int]
    output_box: Tuple[int, int, int, int]
    crop_box: Tuple[int, int, int, int]


def _validate_common(image: torch.Tensor, tile_size: int, overlap: int, alignment: int) -> None:
    if image.ndim != 4:
        raise ValueError(f'Expected a BCHW tensor, but received shape {tuple(image.shape)}.')
    if tile_size <= 0:
        raise ValueError('tile_size must be positive.')
    if overlap < 0 or overlap >= tile_size:
        raise ValueError('overlap must satisfy 0 <= overlap < tile_size.')
    if alignment <= 0:
        raise ValueError('alignment must be positive.')
    if tile_size % alignment != 0 or overlap % alignment != 0:
        raise ValueError('tile_size and overlap must both be multiples of alignment.')


def _symmetric_pad_to_alignment(
        image: torch.Tensor, alignment: int) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    height, width = image.shape[-2:]
    pad_height = -height % alignment
    pad_width = -width % alignment
    top = pad_height // 2
    bottom = pad_height - top
    left = pad_width // 2
    right = pad_width - left
    if pad_height == 0 and pad_width == 0:
        return image, (0, 0, 0, 0)

    # Reflection matches UNetArch._check_and_padding. Very small synthetic
    # inputs cannot be reflected by a large amount, so replicate is a safe
    # fallback for that uncommon case.
    mode = 'reflect'
    if top >= height or bottom >= height or left >= width or right >= width:
        mode = 'replicate'
    return F.pad(image, (left, right, top, bottom), mode=mode), (left, right, top, bottom)


def _remove_padding(image: torch.Tensor, padding: Tuple[int, int, int, int]) -> torch.Tensor:
    left, right, top, bottom = padding
    height_end = image.shape[-2] - bottom if bottom else image.shape[-2]
    width_end = image.shape[-1] - right if right else image.shape[-1]
    return image[..., top:height_end, left:width_end]


def _model_device(model: torch.nn.Module) -> torch.device:
    parameter = next(model.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(model.buffers(), None)
    return buffer.device if buffer is not None else torch.device('cpu')


def _run_patch(
        model: torch.nn.Module,
        patch: torch.Tensor,
        inference_device: torch.device,
        use_amp: bool,
) -> torch.Tensor:
    patch = patch.to(inference_device, non_blocking=inference_device.type == 'cuda')
    amp_enabled = use_amp and inference_device.type == 'cuda'
    with torch.inference_mode(), torch.autocast(
            device_type=inference_device.type, enabled=amp_enabled):
        output = model(patch)
    if output.shape[-2:] != patch.shape[-2:]:
        raise ValueError(
            'Tiled inference currently requires a shape-preserving model, '
            f'but input/output spatial shapes are {patch.shape[-2:]} and {output.shape[-2:]}.')
    return output


def _to_accumulation_device(output: torch.Tensor, device: torch.device) -> torch.Tensor:
    # CPU overlap-add kernels have incomplete/slow float16 support in the
    # PyTorch versions used by LED. Accumulating in float32 is also safer.
    dtype = torch.float32 if device.type == 'cpu' and output.dtype in (
        torch.float16, torch.bfloat16) else output.dtype
    return output.to(device=device, dtype=dtype)


def build_overlap_crop_tiles(
        height: int, width: int, tile_size: int, overlap: int) -> List[Tile]:
    """Build halo tiles whose non-overlapping centre regions cover the image."""
    if height <= 0 or width <= 0:
        raise ValueError('height and width must be positive.')
    if tile_size <= 0 or overlap < 0 or overlap >= tile_size:
        raise ValueError('Expected tile_size > overlap >= 0.')
    if overlap % 2 != 0:
        raise ValueError('crop mode requires an even overlap so it can use equal halos.')
    halo = overlap // 2
    core_size = tile_size - overlap
    tiles = []
    for output_top in range(0, height, core_size):
        output_bottom = min(output_top + core_size, height)
        input_top = max(0, output_top - halo)
        input_bottom = min(height, output_bottom + halo)
        crop_top = output_top - input_top
        crop_bottom = crop_top + output_bottom - output_top
        for output_left in range(0, width, core_size):
            output_right = min(output_left + core_size, width)
            input_left = max(0, output_left - halo)
            input_right = min(width, output_right + halo)
            crop_left = output_left - input_left
            crop_right = crop_left + output_right - output_left
            tiles.append(Tile(
                input_box=(input_top, input_bottom, input_left, input_right),
                output_box=(output_top, output_bottom, output_left, output_right),
                crop_box=(crop_top, crop_bottom, crop_left, crop_right),
            ))
    return tiles


def _sliding_starts(length: int, tile_size: int, overlap: int) -> List[int]:
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def _axis_hann_window(
        starts: List[int],
        ends: List[int],
        index: int,
        device: torch.device,
        dtype: torch.dtype) -> torch.Tensor:
    start, end = starts[index], ends[index]
    window = torch.ones(end - start, device=device, dtype=dtype)
    if index > 0:
        left_overlap = max(0, ends[index - 1] - start)
        phase = (torch.arange(left_overlap, device=device, dtype=dtype) + 0.5) / left_overlap
        window[:left_overlap] *= torch.sin(phase * (torch.pi / 2)).square()
    if index + 1 < len(starts):
        right_overlap = max(0, end - starts[index + 1])
        phase = (torch.arange(right_overlap, device=device, dtype=dtype) + 0.5) / right_overlap
        window[-right_overlap:] *= torch.cos(phase * (torch.pi / 2)).square()
    return window


def _weighted_overlap_add(
        model: torch.nn.Module,
        image: torch.Tensor,
        tile_size: int,
        overlap: int,
        blend: str,
        inference_device: torch.device,
        accumulation_device: torch.device,
        use_amp: bool,
) -> torch.Tensor:
    height, width = image.shape[-2:]
    y_starts = _sliding_starts(height, tile_size, overlap)
    x_starts = _sliding_starts(width, tile_size, overlap)
    y_ends = [min(start + tile_size, height) for start in y_starts]
    x_ends = [min(start + tile_size, width) for start in x_starts]

    output_sum: Optional[torch.Tensor] = None
    weight_sum: Optional[torch.Tensor] = None
    for y_index, (top, bottom) in enumerate(zip(y_starts, y_ends)):
        for x_index, (left, right) in enumerate(zip(x_starts, x_ends)):
            patch = image[..., top:bottom, left:right]
            patch_output = _to_accumulation_device(
                _run_patch(model, patch, inference_device, use_amp), accumulation_device)
            if output_sum is None:
                output_sum = torch.zeros(
                    (*patch_output.shape[:-2], height, width),
                    device=accumulation_device,
                    dtype=patch_output.dtype)
                weight_sum = torch.zeros(
                    (1, 1, height, width), device=accumulation_device, dtype=patch_output.dtype)

            if blend == 'hann':
                y_weight = _axis_hann_window(
                    y_starts, y_ends, y_index, accumulation_device, patch_output.dtype)
                x_weight = _axis_hann_window(
                    x_starts, x_ends, x_index, accumulation_device, patch_output.dtype)
                weight = y_weight[:, None] * x_weight[None, :]
            else:
                weight = torch.ones(
                    (bottom - top, right - left),
                    device=accumulation_device,
                    dtype=patch_output.dtype)
            weight = weight[None, None]
            output_sum[..., top:bottom, left:right] += patch_output * weight
            weight_sum[..., top:bottom, left:right] += weight

    if output_sum is None or weight_sum is None:
        raise RuntimeError('No inference tiles were generated.')
    return output_sum / weight_sum.clamp_min(torch.finfo(weight_sum.dtype).eps)


def tiled_inference(
        model: torch.nn.Module,
        image: torch.Tensor,
        tile_size: int = 512,
        overlap: int = 192,
        blend: str = 'crop',
        alignment: int = 16,
        inference_device: Optional[torch.device] = None,
        accumulation_device: torch.device = torch.device('cpu'),
        use_amp: bool = False,
) -> torch.Tensor:
    """Run a shape-preserving model patch by patch.

    Args:
        model: Denoising/restoration network.
        image: BCHW input. It may stay in CPU memory while patches stream to
            ``inference_device``.
        tile_size: Maximum input patch side length.
        overlap: Number of shared pixels between adjacent input patches.
        blend: ``crop`` discards an overlap/2 halo from internal patch edges;
            ``hann`` uses raised-cosine overlap-add; ``uniform`` averages.
        alignment: Network downsampling factor. Coordinates and patch shapes
            remain aligned to this value to preserve pooling phase.
        inference_device: Device used by the model. Defaults to its parameters.
        accumulation_device: Device holding the assembled full output.
        use_amp: Use CUDA autocast for each patch.

    Returns:
        The assembled BCHW output on ``accumulation_device``.
    """
    if blend not in ('crop', 'hann', 'uniform'):
        raise ValueError("blend must be one of 'crop', 'hann', or 'uniform'.")
    _validate_common(image, tile_size, overlap, alignment)
    if blend == 'crop' and (overlap // 2) % alignment != 0:
        raise ValueError('crop mode requires overlap/2 to be a multiple of alignment.')

    inference_device = (
        torch.device(inference_device) if inference_device is not None else _model_device(model))
    accumulation_device = torch.device(accumulation_device)
    padded, padding = _symmetric_pad_to_alignment(image, alignment)

    if blend == 'crop':
        tiles = build_overlap_crop_tiles(*padded.shape[-2:], tile_size, overlap)
        assembled: Optional[torch.Tensor] = None
        for tile in tiles:
            input_top, input_bottom, input_left, input_right = tile.input_box
            output_top, output_bottom, output_left, output_right = tile.output_box
            crop_top, crop_bottom, crop_left, crop_right = tile.crop_box
            patch = padded[..., input_top:input_bottom, input_left:input_right]
            patch_output = _to_accumulation_device(
                _run_patch(model, patch, inference_device, use_amp), accumulation_device)
            if assembled is None:
                assembled = torch.empty(
                    (*patch_output.shape[:-2], *padded.shape[-2:]),
                    device=accumulation_device,
                    dtype=patch_output.dtype)
            assembled[..., output_top:output_bottom, output_left:output_right] = (
                patch_output[..., crop_top:crop_bottom, crop_left:crop_right])
        if assembled is None:
            raise RuntimeError('No inference tiles were generated.')
    else:
        assembled = _weighted_overlap_add(
            model, padded, tile_size, overlap, blend, inference_device, accumulation_device, use_amp)

    return _remove_padding(assembled, padding)
