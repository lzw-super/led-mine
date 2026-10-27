"""Reference SID architectures with LED's packed-RAW channel convention."""
from copy import deepcopy

import torch
from torch import nn

from led.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class SIDReferenceDenoiser(nn.Module):
    def __init__(self, architecture, settings):
        super().__init__()
        if architecture == 'learning_dwt_repncb':
            from .sid_reference.learning_wt.learning_dwt import learning_dwt_kwargs
            from .sid_reference.models.learning_dwt_repncb import LearningDWTRepNCB, refiner_kwargs
            self.network = LearningDWTRepNCB(
                learning_dwt_kwargs(settings), refine_config=refiner_kwargs(settings))
        elif architecture == 'splitternet':
            from .sid_reference.models.splitternet import SplitterNet
            self.network = SplitterNet(**settings)
        elif architecture == 'mrlfn':
            from .sid_reference.models.mrlfn_arch import MRLFN
            self.network = MRLFN(**settings)
        else:
            raise ValueError(f'Unknown SID reference architecture: {architecture}')
        # LED [R,G1,B,G2] <-> reference [R,G1,G2,B]. This is self-inverse.
        self.register_buffer('channel_order', torch.tensor([0, 1, 3, 2]), persistent=False)

    def forward(self, x):
        return self.network(x.index_select(1, self.channel_order)).index_select(1, self.channel_order)

    def deploy(self):
        result = deepcopy(self).eval()
        if callable(getattr(result.network, 'deploy', None)):
            result.network = result.network.deploy()
        return result
