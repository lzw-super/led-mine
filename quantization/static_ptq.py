"""Static signed-symmetric W8A8 Q/DQ accuracy simulation (not an INT8 runtime).

Weights: output-channel absmax; activations: tensor absmax on held-out training
crops. All scales frozen before test. Floating kernels implement dequantized
arithmetic; PReLU slopes and fixed Haar constants remain exact FP32. No QAT.
"""
import copy
import operator
import torch
from torch import nn, fx


def qdq(x, scale):
    return torch.clamp(torch.round(x / scale), -127, 127) * scale


class StaticPTQ(fx.Interpreter):
    def __init__(self, model):
        graph = fx.symbolic_trace(copy.deepcopy(model).eval())
        super().__init__(graph)
        self.mode = 'observe'
        self.maxima = {}
        self.scales = {}
        self.integer_weights = {}
        self.saturation = {}
        self.nodes = {n.name: n for n in graph.graph.nodes}
        # Quantize data tensors, never constant weights or tuple containers.
        self.boundaries = {n.name for n in graph.graph.nodes
                           if n.op not in ('get_attr', 'output') and
                           not (n.op == 'call_method' and n.target == 'split')}

    def run_node(self, n):
        result = super().run_node(n)
        if n.name not in self.boundaries or not isinstance(result, torch.Tensor):
            return result
        if self.mode == 'observe':
            maximum = result.detach().abs().amax()
            self.maxima[n.name] = torch.maximum(self.maxima.get(n.name, maximum), maximum)
            return result
        scale = self.scales[n.name]
        # Keep statistics on the GPU to avoid a synchronization per node.
        clipped = (result.abs() > 127 * scale).sum()
        old = self.saturation.get(n.name, (torch.zeros((), device=result.device, dtype=torch.long), 0))
        self.saturation[n.name] = (old[0] + clipped, old[1] + result.numel())
        return qdq(result, scale)

    def freeze(self):
        self.scales = {name: value.clamp_min(1e-12) / 127 for name, value in self.maxima.items()}
        for node in self.module.graph.nodes:
            if node.op != 'call_module':
                continue
            layer = self.module.get_submodule(node.target)
            if not isinstance(layer, nn.Conv2d):
                continue
            w = layer.weight.detach()
            sw = w.abs().flatten(1).amax(1).clamp_min(1e-12) / 127
            qw = torch.round(w / sw[:, None, None, None]).clamp(-127, 127).to(torch.int8)
            sx = self.scales[node.args[0].name]
            record = {'weight_int8': qw.cpu(), 'weight_scale': sw.cpu(), 'input_scale': sx.cpu()}
            with torch.no_grad():
                layer.weight.copy_(qw.to(w.dtype) * sw[:, None, None, None])
                if layer.bias is not None:
                    sb = sx * sw
                    qb = torch.round(layer.bias.detach() / sb)
                    if qb.abs().max() > torch.iinfo(torch.int32).max:
                        raise OverflowError(f'INT32 bias overflow: {node.target}')
                    layer.bias.copy_(qb * sb)
                    record.update(bias_int32=qb.to(torch.int32).cpu(), bias_scale=sb.cpu())
            self.integer_weights[node.target] = record
        self.mode = 'quantize'

    def export(self, path, metadata):
        torch.save({'format': 'signed_symmetric_w8a8_qdq_v1', 'metadata': metadata,
                    'state_dict': {k: v.cpu() for k, v in self.module.state_dict().items()},
                    'activation_scales': {k: v.cpu() for k, v in self.scales.items()},
                    'integer_convs': self.integer_weights,
                    'graph': str(self.module.graph)}, path)

    def load_artifact(self, path):
        data = torch.load(path, map_location=next(self.module.parameters()).device)
        self.module.load_state_dict(data['state_dict'], strict=True)
        if str(self.module.graph) != data['graph']:
            raise ValueError('Quantization graph differs from artifact')
        self.scales = data['activation_scales']
        self.integer_weights = data['integer_convs']
        self.mode = 'quantize'
        return data['metadata']

    def statistics(self):
        return {name: {'scale': float(scale), 'absmax': float(scale * 127),
                       'clipped': int(self.saturation[name][0]) if name in self.saturation else 0,
                       'elements': self.saturation.get(name, (0, 0))[1]}
                for name, scale in self.scales.items()}
