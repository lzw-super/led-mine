"""Mobile Residual Local Feature Network for packed-RAW denoising.

This is a single-frame, patch-training adaptation of the architecture in
"Efficient Real-Time Raw-to-Raw Denoising for Extreme Low-Light Ultra HD
Video on Mobile Devices".  Space/depth transforms and temporal convolutions
are intentionally omitted; SID's four Bayer planes are consumed directly.
"""

from copy import deepcopy

import torch
from torch import nn

from led.utils.registry import ARCH_REGISTRY


class ReparamConv3x3(nn.Module):
    """Training-time three-convolution block that fuses to one 3x3 conv.

    The training graph follows Figure 7 of the paper::

        y = ReLU(x + Conv1x1_b(Conv3x3(x + Conv1x1_a(x))))

    ``Conv1x1_a`` deliberately has no bias.  A bias before a zero-padded 3x3
    convolution would make the border response spatially varying and could
    not be represented exactly by a single convolution with one bias vector.
    Biases after the padded convolution remain exactly fusible.
    """

    def __init__(self, channels, bias=True, deploy=False):
        super().__init__()
        if channels <= 0:
            raise ValueError(f'channels must be positive, got {channels}.')

        self.channels = channels
        self.deploy_mode = deploy
        self.activation = nn.ReLU(inplace=True)

        if deploy:
            self.reparam_conv = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
        else:
            self.expand_conv = nn.Conv2d(channels, channels, 1, bias=False)
            self.spatial_conv = nn.Conv2d(channels, channels, 3, 1, 1, bias=bias)
            self.project_conv = nn.Conv2d(channels, channels, 1, bias=bias)

    def forward(self, x):
        if self.deploy_mode:
            return self.activation(self.reparam_conv(x))

        local_feature = x + self.expand_conv(x)
        refined_feature = self.project_conv(self.spatial_conv(local_feature))
        return self.activation(x + refined_feature)

    def get_equivalent_kernel_bias(self):
        """Return the kernel and bias of the mathematically equivalent conv."""
        if self.deploy_mode:
            return self.reparam_conv.weight, self.reparam_conv.bias

        expand_weight = self.expand_conv.weight[:, :, 0, 0]
        spatial_weight = self.spatial_conv.weight
        project_weight = self.project_conv.weight[:, :, 0, 0]

        identity = torch.eye(
            self.channels, device=expand_weight.device, dtype=expand_weight.dtype)
        local_weight = expand_weight + identity

        # Conv3x3(Conv1x1(x)) followed by Conv1x1 is a 3x3 convolution.
        equivalent_weight = torch.einsum('omhw,mi->oihw', spatial_weight, local_weight)
        equivalent_weight = torch.einsum('om,mihw->oihw', project_weight, equivalent_weight)
        equivalent_weight[:, :, 1, 1] += identity

        if self.spatial_conv.bias is None:
            spatial_bias = spatial_weight.new_zeros(self.channels)
        else:
            spatial_bias = self.spatial_conv.bias
        if self.project_conv.bias is None:
            project_bias = spatial_weight.new_zeros(self.channels)
        else:
            project_bias = self.project_conv.bias
        equivalent_bias = torch.matmul(project_weight, spatial_bias) + project_bias
        return equivalent_weight, equivalent_bias

    @torch.no_grad()
    def switch_to_deploy(self):
        """Replace the training branches in-place with one 3x3 convolution."""
        if self.deploy_mode:
            return self

        kernel, bias = self.get_equivalent_kernel_bias()
        reparam_conv = nn.Conv2d(self.channels, self.channels, 3, 1, 1, bias=True)
        reparam_conv = reparam_conv.to(device=kernel.device, dtype=kernel.dtype)
        reparam_conv.weight.copy_(kernel)
        reparam_conv.bias.copy_(bias)
        self.reparam_conv = reparam_conv

        del self.expand_conv
        del self.spatial_conv
        del self.project_conv
        self.deploy_mode = True
        return self


class MobileResidualLocalFeatureBlock(nn.Module):
    """The paper's mobile-optimized Residual Local Feature Block (mRLFB)."""

    def __init__(self, channels, bias=True, deploy=False):
        super().__init__()
        self.conv1 = ReparamConv3x3(channels, bias=bias, deploy=deploy)
        self.conv2 = ReparamConv3x3(channels, bias=bias, deploy=deploy)
        self.conv3 = ReparamConv3x3(channels, bias=bias, deploy=deploy)
        self.linear = nn.Conv2d(channels, channels, 1, bias=bias)

    def forward(self, x):
        feature = self.conv3(self.conv2(self.conv1(x)))
        feature = self.linear(feature + x)
        return feature + x


@ARCH_REGISTRY.register()
class MRLFN(nn.Module):
    """Single-frame mRLFB network adapted to four-channel SID packed RAW.

    Args:
        in_channels: Number of packed input planes. SID uses four.
        out_channels: Number of packed output planes. SID uses four.
        feature_channels: Intermediate feature depth ``d`` from the paper.
        num_blocks: Number ``N`` of cascaded mRLFBs.
        bias: Use biases in convolutions for which exact fusion permits them.
        deploy: Build the inference graph whose reparameterized modules each
            contain only one 3x3 convolution.
    """

    def __init__(self,
                 in_channels=4,
                 out_channels=4,
                 feature_channels=32,
                 num_blocks=4,
                 bias=True,
                 deploy=False):
        super().__init__()
        if in_channels <= 0 or out_channels <= 0 or feature_channels <= 0:
            raise ValueError('Channel counts must all be positive.')
        if num_blocks <= 0:
            raise ValueError(f'num_blocks must be positive, got {num_blocks}.')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_channels = feature_channels
        self.num_blocks = num_blocks
        self.deploy_mode = deploy

        self.shallow_conv = nn.Conv2d(in_channels, feature_channels, 3, 1, 1, bias=bias)
        self.blocks = nn.Sequential(*[
            MobileResidualLocalFeatureBlock(feature_channels, bias=bias, deploy=deploy)
            for _ in range(num_blocks)
        ])
        self.deep_fusion = nn.Conv2d(feature_channels, feature_channels, 3, 1, 1, bias=bias)
        self.shallow_fusion = nn.Conv2d(feature_channels, feature_channels, 1, bias=bias)
        self.output_conv = nn.Conv2d(2 * feature_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f'Expected NCHW input with {self.in_channels} channels, got {tuple(x.shape)}.')

        shallow_feature = self.shallow_conv(x)
        deep_feature = self.deep_fusion(self.blocks(shallow_feature))
        shallow_feature = self.shallow_fusion(shallow_feature)
        return self.output_conv(torch.cat((deep_feature, shallow_feature), dim=1))

    @torch.no_grad()
    def switch_to_deploy(self):
        """Fuse every training-time reparameterizable block in-place."""
        for module in self.modules():
            if isinstance(module, ReparamConv3x3):
                module.switch_to_deploy()
        self.deploy_mode = True
        return self

    @torch.no_grad()
    def deploy(self):
        """Return a fused copy, leaving the live training network untouched."""
        return deepcopy(self).switch_to_deploy()
