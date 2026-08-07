"""Composite RAW-domain objective used by the mRLFB denoising paper."""

from torch import nn

from led.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class RawReconstructionChromaticLoss(nn.Module):
    """Weighted RAW reconstruction and chromatic-aberration objective.

    SID tensors in this repository are packed as ``[R, G1, B, G2]``.  The
    channel order remains configurable so the loss cannot silently apply the
    Bayer formula to a differently packed dataset.
    """

    def __init__(self,
                 raw_weight=0.6,
                 chromatic_weight=0.4,
                 loss_weight=1.0,
                 reduction='mean',
                 channel_order=(0, 1, 2, 3)):
        super().__init__()
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError(f'Unsupported reduction: {reduction}.')
        if len(channel_order) != 4 or len(set(channel_order)) != 4:
            raise ValueError('channel_order must contain four distinct indices for [R, G1, B, G2].')
        if min(channel_order) < 0:
            raise ValueError('channel_order indices must be non-negative.')

        self.raw_weight = raw_weight
        self.chromatic_weight = chromatic_weight
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.channel_order = tuple(channel_order)

    def forward(self, pred, target, weight=None, **kwargs):
        if pred.shape != target.shape or pred.ndim != 4:
            raise ValueError(
                f'pred and target must have the same NCHW shape, got {pred.shape} and {target.shape}.')
        if max(self.channel_order) >= pred.shape[1]:
            raise ValueError(
                f'channel_order {self.channel_order} is invalid for {pred.shape[1]} input channels.')
        if weight is not None:
            raise ValueError('Element-wise weights are not defined for the composite chromatic loss.')

        raw_per_sample = (pred - target).abs().flatten(1).mean(1)

        red, green1, blue, green2 = self.channel_order
        pred_green = 0.5 * (pred[:, green1] + pred[:, green2])
        target_green = 0.5 * (target[:, green1] + target[:, green2])
        pred_bg = pred[:, blue] - pred_green
        target_bg = target[:, blue] - target_green
        pred_rg = pred[:, red] - pred_green
        target_rg = target[:, red] - target_green
        chromatic_per_sample = (
            (pred_bg - target_bg).abs().flatten(1).mean(1)
            + (pred_rg - target_rg).abs().flatten(1).mean(1))

        loss = self.loss_weight * (
            self.raw_weight * raw_per_sample
            + self.chromatic_weight * chromatic_per_sample)
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss
