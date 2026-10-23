"""Differentiable single-scale SSIM on each packed RAW channel.

Uses Wang et al.'s Gaussian window, population covariance, and valid borders.
The objective is 1 - SSIM (not half that value). No display ISP or clipping is
applied: clipping model predictions would suppress out-of-range gradients.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from led.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class SSIMLoss(nn.Module):
    def __init__(self, loss_weight=1.0, data_range=1.0, window_size=11, sigma=1.5):
        super().__init__()
        if not isinstance(window_size, int) or window_size < 3 or window_size % 2 == 0:
            raise ValueError('window_size must be an odd integer >= 3.')
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError('sigma must be finite and positive.')
        if not math.isfinite(data_range) or data_range <= 0:
            raise ValueError('data_range must be finite and positive.')
        if not math.isfinite(loss_weight) or loss_weight < 0:
            raise ValueError('loss_weight must be finite and nonnegative.')
        self.loss_weight = loss_weight
        self.data_range = data_range
        self.window_size = window_size
        coords = torch.arange(window_size, dtype=torch.float64) - window_size // 2
        gaussian = torch.exp(-coords.square() / (2 * sigma ** 2))
        gaussian /= gaussian.sum()
        self.register_buffer('window', (gaussian[:, None] * gaussian[None, :])[None, None])

    def forward(self, pred, target):
        if pred.ndim != 4 or pred.shape != target.shape:
            raise ValueError('pred and target must have matching NCHW shapes.')
        if min(pred.shape[-2:]) < self.window_size:
            raise ValueError('Spatial dimensions must be >= window_size.')
        if not pred.is_floating_point() or not target.is_floating_point():
            raise ValueError('SSIM requires floating-point inputs.')
        # Local second moments are fragile in half precision. Keep float64 for
        # reference/gradient checks and use float32 for FP16/BF16 training.
        with torch.autocast(device_type=pred.device.type, enabled=False):
            dtype = torch.float64 if pred.dtype == torch.float64 else torch.float32
            x, y = pred.to(dtype), target.to(dtype)
            channels = x.shape[1]
            kernel = self.window.to(device=x.device, dtype=dtype).expand(channels, 1, -1, -1)
            def average(z):
                return F.conv2d(z, kernel, groups=channels)
            mx, my = average(x), average(y)
            vx = average(x.square()) - mx.square()
            vy = average(y.square()) - my.square()
            cov = average(x * y) - mx * my
            c1, c2 = (0.01 * self.data_range) ** 2, (0.03 * self.data_range) ** 2
            ssim = ((2 * mx * my + c1) * (2 * cov + c2) /
                    ((mx.square() + my.square() + c1) * (vx + vy + c2)))
            return self.loss_weight * (1 - ssim.mean())
