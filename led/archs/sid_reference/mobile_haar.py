"""Static, convolution-only rearrangements for the trained Haar HF-CNN model.

Input/output use LED's [R,G1,B,G2] order. Spatial dimensions must be multiples
of eight. This graph deliberately excludes padding and training auxiliaries.
"""
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from .models.fixed_raw_ops import FixedHaarConv, FixedSpaceDepth


class MobileHaarRepNCB(nn.Module):
    def __init__(self, reference):
        super().__init__()
        core = reference.network
        wavelet, refiner = core.wavelet, core.refiner
        if not (
            wavelet.wavelet == 'haar' and wavelet.levels == 3
            and wavelet.transform is None and wavelet.threshold_mode == 'hf_cnn'
            and wavelet.ll_mode == 'll_only' and not wavelet.ll_normalize
            and wavelet.ll_restorer.fusion is not None
            and refiner.s2d_factor == 2 and refiner.skip_source == 'noisy'
            and refiner.input_fusion == 'none'
            and isinstance(refiner.skip_projection, nn.Identity)
            and isinstance(refiner.stem_activation, nn.Identity)
        ):
            raise ValueError('Expected the fixed Haar L3 / LL concat / HF CNN / noisy-skip configuration')
        for processor in wavelet.processors:
            if not (processor.residual and not processor.dual
                    and hasattr(processor, 'block')
                    and processor.pointwise is None
                    and isinstance(processor.skip_projection, nn.Identity)):
                raise ValueError('Expected HF RepNCB + residual processors')

        # Fuse training branches on a copy; the original checkpoint is untouched.
        deployed = deepcopy(core).cpu().eval().switch_to_deploy()
        self.hf_blocks = nn.ModuleList([p.block for p in deployed.wavelet.processors])
        self.ll_layers = deployed.wavelet.ll_restorer.layers
        self.ll_fusion = deployed.wavelet.ll_restorer.fusion
        self.stem = deployed.refiner.stem
        self.blocks = deployed.refiner.blocks
        self.head = deployed.refiner.head
        self.fusion = deployed.refiner.fusion

        haar = FixedHaarConv().weight
        space = FixedSpaceDepth().weight
        order = [0, 1, 3, 2]
        self.register_buffer('haar_weight', haar.clone())
        # Fold LED -> reference channel permutation into the first analysis
        # convolution and the noisy-input S2D branch (input-channel axis).
        self.register_buffer('input_haar_weight', haar[:, order].clone())
        self.register_buffer('space_weight', space.clone())
        self.register_buffer('input_space_weight', space[:, order].clone())
        # ConvTranspose stores [input_channel, output_channel, H, W].
        self.register_buffer('output_space_weight', space[:, order].clone())

    def forward(self, raw):
        def visit(current, level):
            weight = self.input_haar_weight if level == 0 else self.haar_weight
            ll, hf = F.conv2d(current, weight, stride=2).split([4, 12], dim=1)
            hf = hf + self.hf_blocks[level](hf)
            if level < 2:
                ll = visit(ll, level + 1)
            else:
                ll = self.ll_fusion(torch.cat((self.ll_layers(ll), ll), dim=1))
            return F.conv_transpose2d(torch.cat((ll, hf), dim=1), self.haar_weight, stride=2)

        preliminary = visit(raw, 0)
        packed = F.conv2d(preliminary, self.space_weight, stride=2)
        features = self.head(self.blocks(self.stem(packed)))
        skip = F.conv2d(raw, self.input_space_weight, stride=2)
        fused = self.fusion(torch.cat((features, skip), dim=1))
        return F.conv_transpose2d(fused, self.output_space_weight, stride=2)
