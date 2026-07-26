"""Utilities for logging neural-network parameter and MAC complexity."""

from copy import deepcopy

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


def _validate_input_shape(input_shape):
    if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 4:
        raise ValueError(
            'model_complexity.input_shape must be [batch, channels, height, width], '
            f'but got {input_shape!r}.')
    if any(not isinstance(value, int) or value <= 0 for value in input_shape):
        raise ValueError(
            'model_complexity.input_shape must contain only positive integers, '
            f'but got {input_shape!r}.')


def _validate_raw_pixel_scale(raw_pixel_scale):
    if raw_pixel_scale is not None and (
            not isinstance(raw_pixel_scale, int) or raw_pixel_scale <= 0):
        raise ValueError(
            'model_complexity.raw_pixel_scale must be a positive integer or null, '
            f'but got {raw_pixel_scale!r}.')


def _compute_complexity(net, input_shape):
    """Count parameters and MACs of ``net`` for a dummy input of ``input_shape``.

    Args:
        net (torch.nn.Module): A bare network. Left in its original
            train/eval state on return.
        input_shape (list[int] | tuple[int, ...]): Dummy input shape,
            including batch and channel dimensions.

    Returns:
        (int, float): ``(parameter_count, macs)``. MACs are reported by THOP
        (multiply-accumulate operations).
    """
    _validate_input_shape(input_shape)

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

    return parameter_count, float(macs)


def _log_complexity(label, input_shape, parameter_count, macs, raw_pixel_scale=None):
    """Write the parameter/MAC/FLOP summary line(s) for one network variant."""
    _validate_raw_pixel_scale(raw_pixel_scale)
    logger = get_root_logger()
    logger.info(
        '%s (input: %s): parameters: %s (%.3f M); '
        'MACs: %s (%.3f G, THOP); FLOPs: %.3f G (2 FLOPs/MAC).',
        label,
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
            '%s normalized: %.1f MAC/RAW-pixel (raw_pixel_scale=%d).',
            label, macs / raw_pixel_count, raw_pixel_scale)


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
    parameter_count, macs = _compute_complexity(net, input_shape)
    _log_complexity('Model complexity', input_shape, parameter_count, macs, raw_pixel_scale)


@master_only
def log_repnr_deployed_complexity(net, input_shape, raw_pixel_scale=None):
    """Fuse a RepNR reparameterizable ``net`` to deployment form and log it.

    The training-time RepNR network carries multiple alignment/auxiliary
    branches that are mathematically folded into a single ``nn.Conv2d`` per
    block at deployment via ``net.deploy()``. ``log_model_complexity`` reports
    the heavier training-state cost; this function profiles the fused network
    so the log also carries the real deployment-state cost (the numbers a
    downstream deployment/export would see).

    The fusion runs on a deep-copied, CPU-side clone, so the live training
    network (parameters, optimizer state, DDP wrappers, ``cur_branch``) is
    never touched and no extra GPU memory is held beyond the call.

    Args:
        net (torch.nn.Module): A bare RepNR network exposing a callable
            ``deploy()`` (e.g. ``RepNRBase``). Non-reparameterizable networks
            are skipped with a log line.
        input_shape (list[int] | tuple[int, ...]): Same convention as
            :func:`log_model_complexity`.
        raw_pixel_scale (int | None): Same convention as
            :func:`log_model_complexity`.
    """
    logger = get_root_logger()

    if not callable(getattr(net, 'deploy', None)):
        logger.info(
            'Deployed-state model complexity skipped: %s is not a RepNR '
            'reparameterizable network (no deploy()).',
            type(net).__name__)
        return

    logger.info(
        'Fusing RepNR branches into single convolutions for deployment-state '
        'complexity (evaluated on a CPU clone; training state untouched).')
    # deepcopy + cpu: reuse the up-to-date trained weights without mutating the
    # live training network or allocating GPU memory for the probe.
    deployed_clone = deepcopy(net).cpu().eval()
    try:
        deployed_net = deployed_clone.deploy()
    except Exception as error:  # pragma: no cover - defensive: never block training
        logger.warning(
            'Failed to fuse RepNR network for deployment-state complexity: %s', error)
        return

    parameter_count, macs = _compute_complexity(deployed_net, input_shape)
    _log_complexity('Deployed-state model complexity', input_shape,
                    parameter_count, macs, raw_pixel_scale)
