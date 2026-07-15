"""Utilities for logging neural-network parameter and MAC complexity."""

import torch

from .dist_util import master_only
from .logger import get_root_logger


_THOP_PROFILE_BUFFER_NAMES = ('total_ops', 'total_params')


def _remove_thop_profile_buffers(net):
    """Remove temporary buffers left by THOP on modules without counting hooks.

    THOP only removes these buffers from modules that have registered counting
    hooks.  Containers and custom NAFNet modules have no hook, so their
    ``total_ops``/``total_params`` buffers would otherwise leak into a later
    checkpoint's state dict.
    """
    for module in net.modules():
        for name in _THOP_PROFILE_BUFFER_NAMES:
            module._buffers.pop(name, None)


@master_only
def log_model_complexity(net, input_shape, raw_pixel_scale=None):
    """Profile ``net`` once and write its parameter count and MACs to the log.

    Args:
        net (torch.nn.Module): A bare network (not DDP/DataParallel wrapped).
        input_shape (list[int] | tuple[int, ...]): Dummy input shape, including
            batch and channel dimensions.
        raw_pixel_scale (int | None): Number of sensor RAW pixels represented
            by one spatial dimension of an input pixel. For 2x2 Bayer-packed
            RAW, set this to ``2`` to log MAC/RAW-pixel.

    Notes:
        THOP reports multiply-accumulate operations (MACs). One MAC is two
        arithmetic operations/FLOPs under the hardware-budget convention.
    """
    if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 4:
        raise ValueError(
            'model_complexity.input_shape must be [batch, channels, height, width], '
            f'but got {input_shape!r}.')
    if any(not isinstance(value, int) or value <= 0 for value in input_shape):
        raise ValueError(
            'model_complexity.input_shape must contain only positive integers, '
            f'but got {input_shape!r}.')
    if raw_pixel_scale is not None and (
            not isinstance(raw_pixel_scale, int) or raw_pixel_scale <= 0):
        raise ValueError(
            'model_complexity.raw_pixel_scale must be a positive integer or null, '
            f'but got {raw_pixel_scale!r}.')

    try:
        from thop import profile
    except ImportError as error:
        raise ImportError(
            'Model complexity logging requires thop. Install dependencies with '
            '`pip install -r requirements.txt`.') from error

    try:
        parameter = next(net.parameters())
    except StopIteration:
        parameter = None

    device = parameter.device if parameter is not None else torch.device('cpu')
    dtype = parameter.dtype if parameter is not None else torch.float32
    dummy_input = torch.zeros(*input_shape, device=device, dtype=dtype)
    parameter_count = sum(parameter.numel() for parameter in net.parameters())
    was_training = net.training

    try:
        _remove_thop_profile_buffers(net)
        net.eval()
        with torch.no_grad():
            macs, _ = profile(net, inputs=(dummy_input,), verbose=False)
    finally:
        _remove_thop_profile_buffers(net)
        net.train(was_training)

    logger = get_root_logger()
    logger.info(
        'Model complexity (input: %s): parameters: %s (%.3f M); '
        'MACs: %s (%.3f G, THOP); FLOPs: %.3f G (2 FLOPs/MAC).',
        tuple(input_shape),
        f'{parameter_count:,d}',
        parameter_count / 1e6,
        f'{int(macs):,d}',
        macs / 1e9,
        2 * macs / 1e9)
    if raw_pixel_scale is not None:
        batch_size, _, height, width = input_shape
        raw_pixel_count = batch_size * (height * raw_pixel_scale) * (width * raw_pixel_scale)
        logger.info(
            'Model complexity normalized: %.1f MAC/RAW-pixel '
            '(raw_pixel_scale=%d).', macs / raw_pixel_count, raw_pixel_scale)
