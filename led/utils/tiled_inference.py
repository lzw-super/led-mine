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


@dataclass(frozen=True)
class TilingStats:
    """Static work estimate for one image and crop-merge configuration."""

    original_height: int
    original_width: int
    padded_height: int
    padded_width: int
    tile_count: int
    output_pixels: int
    processed_pixels: int
    processed_pixel_ratio: float
    patch_shapes: Tuple[Tuple[int, int, int], ...]


@dataclass(frozen=True)
class BatchTileCrop:
    """Aligned image crop whose sliding-window tiles all have one shape."""

    crop_box: Tuple[int, int, int, int]
    original_height: int
    original_width: int
    cropped_height: int
    cropped_width: int
    tile_rows: int
    tile_columns: int
    tile_count: int


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


def build_batch_tile_crop(
        height: int,
        width: int,
        tile_size: int = 512,
        overlap: int = 192,
        alignment: int = 16,
) -> BatchTileCrop:
    """Find the largest aligned crop covered by equal sliding-window tiles.

    The cropped side length has the form ``tile_size + n * stride`` where
    ``stride = tile_size - overlap``. The top/left crop offsets remain aligned
    to the network downsampling phase.
    """
    if height <= 0 or width <= 0:
        raise ValueError('height and width must be positive.')
    dummy = torch.empty((1, 1, height, width), device='meta')
    _validate_common(dummy, tile_size, overlap, alignment)
    if height < tile_size or width < tile_size:
        raise ValueError(
            f'Image {height}x{width} is smaller than tile_size={tile_size}.')

    stride = tile_size - overlap

    def crop_axis(length: int) -> Tuple[int, int, int, int]:
        steps = (length - tile_size) // stride
        cropped_length = tile_size + steps * stride
        removed = length - cropped_length
        start = (removed // 2 // alignment) * alignment
        end = start + cropped_length
        return start, end, cropped_length, steps + 1

    top, bottom, cropped_height, tile_rows = crop_axis(height)
    left, right, cropped_width, tile_columns = crop_axis(width)
    return BatchTileCrop(
        crop_box=(top, bottom, left, right),
        original_height=height,
        original_width=width,
        cropped_height=cropped_height,
        cropped_width=cropped_width,
        tile_rows=tile_rows,
        tile_columns=tile_columns,
        tile_count=tile_rows * tile_columns,
    )


def crop_to_batchable_tiles(
        image: torch.Tensor,
        tile_size: int = 512,
        overlap: int = 192,
        alignment: int = 16,
) -> Tuple[torch.Tensor, BatchTileCrop]:
    """Crop a BCHW image so all sliding-window patches are equally sized."""
    _validate_common(image, tile_size, overlap, alignment)
    crop = build_batch_tile_crop(
        *image.shape[-2:], tile_size=tile_size, overlap=overlap, alignment=alignment)
    top, bottom, left, right = crop.crop_box
    return image[..., top:bottom, left:right], crop


def analyze_batched_tiling(
        height: int,
        width: int,
        tile_size: int = 512,
        overlap: int = 192,
        alignment: int = 16,
) -> TilingStats:
    """Analyze an already cropped, equal-shape sliding-window tile grid."""
    dummy = torch.empty((1, 1, height, width), device='meta')
    _validate_common(dummy, tile_size, overlap, alignment)
    stride = tile_size - overlap
    if height < tile_size or width < tile_size:
        raise ValueError(
            f'Image {height}x{width} is smaller than tile_size={tile_size}.')
    if (height - tile_size) % stride or (width - tile_size) % stride:
        raise ValueError(
            'Batched tiled inference requires each side to equal '
            'tile_size + n * (tile_size - overlap). Use '
            'crop_to_batchable_tiles() first.')

    tile_rows = (height - tile_size) // stride + 1
    tile_columns = (width - tile_size) // stride + 1
    tile_count = tile_rows * tile_columns
    output_pixels = height * width
    processed_pixels = tile_count * tile_size * tile_size
    return TilingStats(
        original_height=height,
        original_width=width,
        padded_height=height,
        padded_width=width,
        tile_count=tile_count,
        output_pixels=output_pixels,
        processed_pixels=processed_pixels,
        processed_pixel_ratio=processed_pixels / output_pixels,
        patch_shapes=((tile_size, tile_size, tile_count),),
    )


def analyze_tiling(
        height: int,
        width: int,
        tile_size: int = 512,
        overlap: int = 192,
        blend: str = 'crop',
        alignment: int = 16,
) -> TilingStats:
    """Count patches and processed pixels without running the network.

    ``patch_shapes`` contains ``(height, width, count)`` tuples. For a
    fully-convolutional network whose coordinates are alignment-safe, MACs are
    proportional to ``processed_pixels``.
    """
    if height <= 0 or width <= 0:
        raise ValueError('height and width must be positive.')
    if blend not in ('crop', 'hann', 'uniform'):
        raise ValueError("blend must be one of 'crop', 'hann', or 'uniform'.")
    dummy = torch.empty((1, 1, height, width), device='meta')
    _validate_common(dummy, tile_size, overlap, alignment)
    if blend == 'crop':
        if overlap % 2:
            raise ValueError('crop mode requires an even overlap.')
        if (overlap // 2) % alignment != 0:
            raise ValueError('crop mode requires overlap/2 to be a multiple of alignment.')

    padded_height = height + (-height % alignment)
    padded_width = width + (-width % alignment)
    shapes = {}
    processed_pixels = 0
    if blend == 'crop':
        boxes = [
            tile.input_box
            for tile in build_overlap_crop_tiles(
                padded_height, padded_width, tile_size, overlap)
        ]
    else:
        y_starts = _sliding_starts(padded_height, tile_size, overlap)
        x_starts = _sliding_starts(padded_width, tile_size, overlap)
        boxes = [
            (
                top,
                min(top + tile_size, padded_height),
                left,
                min(left + tile_size, padded_width),
            )
            for top in y_starts
            for left in x_starts
        ]

    for top, bottom, left, right in boxes:
        patch_shape = (bottom - top, right - left)
        shapes[patch_shape] = shapes.get(patch_shape, 0) + 1
        processed_pixels += patch_shape[0] * patch_shape[1]
    output_pixels = padded_height * padded_width
    patch_shapes = tuple(
        (patch_height, patch_width, count)
        for (patch_height, patch_width), count in sorted(shapes.items()))
    return TilingStats(
        original_height=height,
        original_width=width,
        padded_height=padded_height,
        padded_width=padded_width,
        tile_count=len(boxes),
        output_pixels=output_pixels,
        processed_pixels=processed_pixels,
        processed_pixel_ratio=processed_pixels / output_pixels,
        patch_shapes=patch_shapes,
    )


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


def _batch_crop_intervals(
        starts: List[int], tile_size: int, length: int) -> List[Tuple[int, int, int, int]]:
    """Map equal input windows to disjoint output intervals at overlap midpoints."""
    ends = [start + tile_size for start in starts]
    boundaries = [0]
    boundaries.extend(
        (ends[index] + starts[index + 1]) // 2
        for index in range(len(starts) - 1))
    boundaries.append(length)
    return [
        (
            boundaries[index],
            boundaries[index + 1],
            boundaries[index] - start,
            boundaries[index + 1] - start,
        )
        for index, start in enumerate(starts)
    ]


def batched_tiled_inference(
        model: torch.nn.Module,
        image: torch.Tensor,
        tile_size: int = 512,
        overlap: int = 192,
        blend: str = 'crop',
        alignment: int = 16,
        tile_batch_size: int = 0,
        inference_device: Optional[torch.device] = None,
        accumulation_device: Optional[torch.device] = None,
        use_amp: bool = False,
) -> torch.Tensor:
    """Run equal-shape tiles in parallel batches and merge on one device.

    ``image`` must first be cropped with :func:`crop_to_batchable_tiles`.
    ``tile_batch_size=0`` sends every spatial tile through the network in one
    batch. A positive value chunks the tile batch to fit the target hardware.
    Unlike :func:`tiled_inference`, accumulation defaults to the inference
    device so patch merge is also performed on the GPU.
    """
    if blend not in ('crop', 'hann', 'uniform'):
        raise ValueError("blend must be one of 'crop', 'hann', or 'uniform'.")
    _validate_common(image, tile_size, overlap, alignment)
    if blend == 'crop':
        if overlap % 2:
            raise ValueError('crop mode requires an even overlap.')
        if (overlap // 2) % alignment != 0:
            raise ValueError('crop mode requires overlap/2 to be a multiple of alignment.')
    if tile_batch_size < 0:
        raise ValueError('tile_batch_size must be zero or positive.')

    stats = analyze_batched_tiling(
        *image.shape[-2:],
        tile_size=tile_size,
        overlap=overlap,
        alignment=alignment)
    inference_device = (
        torch.device(inference_device) if inference_device is not None else _model_device(model))
    accumulation_device = (
        inference_device if accumulation_device is None
        else torch.device(accumulation_device))
    stride = tile_size - overlap
    height, width = image.shape[-2:]
    y_starts = list(range(0, height - tile_size + 1, stride))
    x_starts = list(range(0, width - tile_size + 1, stride))
    boxes = [
        (top, top + tile_size, left, left + tile_size, y_index, x_index)
        for y_index, top in enumerate(y_starts)
        for x_index, left in enumerate(x_starts)
    ]
    if len(boxes) != stats.tile_count:
        raise RuntimeError('The batched tile grid is inconsistent with its workload analysis.')

    batch_size = image.shape[0]
    tiles_per_batch = tile_batch_size or len(boxes)
    assembled: Optional[torch.Tensor] = None
    output_sum: Optional[torch.Tensor] = None
    weight_sum: Optional[torch.Tensor] = None
    y_crop_intervals = _batch_crop_intervals(y_starts, tile_size, height)
    x_crop_intervals = _batch_crop_intervals(x_starts, tile_size, width)
    y_ends = [start + tile_size for start in y_starts]
    x_ends = [start + tile_size for start in x_starts]

    for chunk_start in range(0, len(boxes), tiles_per_batch):
        chunk = boxes[chunk_start:chunk_start + tiles_per_batch]
        patch_batch = torch.cat(
            [image[..., top:bottom, left:right]
             for top, bottom, left, right, _, _ in chunk],
            dim=0)
        patch_outputs = _to_accumulation_device(
            _run_patch(model, patch_batch, inference_device, use_amp),
            accumulation_device)
        patch_outputs = patch_outputs.reshape(
            len(chunk), batch_size, *patch_outputs.shape[1:])

        if blend == 'crop':
            if assembled is None:
                assembled = torch.empty(
                    (batch_size, *patch_outputs.shape[2:-2], height, width),
                    device=accumulation_device,
                    dtype=patch_outputs.dtype)
            for patch_output, (_, _, _, _, y_index, x_index) in zip(
                    patch_outputs, chunk):
                output_top, output_bottom, crop_top, crop_bottom = (
                    y_crop_intervals[y_index])
                output_left, output_right, crop_left, crop_right = (
                    x_crop_intervals[x_index])
                assembled[..., output_top:output_bottom, output_left:output_right] = (
                    patch_output[..., crop_top:crop_bottom, crop_left:crop_right])
        else:
            if output_sum is None:
                output_sum = torch.zeros(
                    (batch_size, *patch_outputs.shape[2:-2], height, width),
                    device=accumulation_device,
                    dtype=patch_outputs.dtype)
                weight_sum = torch.zeros(
                    (1, 1, height, width),
                    device=accumulation_device,
                    dtype=patch_outputs.dtype)
            for patch_output, (top, bottom, left, right, y_index, x_index) in zip(
                    patch_outputs, chunk):
                if blend == 'hann':
                    y_weight = _axis_hann_window(
                        y_starts, y_ends, y_index, accumulation_device, patch_output.dtype)
                    x_weight = _axis_hann_window(
                        x_starts, x_ends, x_index, accumulation_device, patch_output.dtype)
                    weight = y_weight[:, None] * x_weight[None, :]
                else:
                    weight = torch.ones(
                        (tile_size, tile_size),
                        device=accumulation_device,
                        dtype=patch_output.dtype)
                weight = weight[None, None]
                output_sum[..., top:bottom, left:right] += patch_output * weight
                weight_sum[..., top:bottom, left:right] += weight

    if blend == 'crop':
        if assembled is None:
            raise RuntimeError('No inference tiles were generated.')
        return assembled
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
