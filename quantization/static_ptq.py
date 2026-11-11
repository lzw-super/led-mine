"""Static signed-symmetric W8A8/W8A16 Q/DQ accuracy simulation (not an INT8 runtime).

Weights: output-channel absmax; activations: tensor absmax on held-out training
crops. All scales frozen before test. Floating kernels implement dequantized
arithmetic; PReLU slopes and fixed Haar constants remain exact FP32. No QAT.
"""
import copy
import operator
import torch
from torch import nn, fx
from quantization.boundaries import boundary_policy, scale_source


def activation_qmax(bits):
    if bits not in (8, 16):
        raise ValueError("Activation bits must be 8 or 16")
    return (1 << (bits - 1)) - 1


def qdq(x, scale, qmax=127):
    return torch.clamp(torch.round(x / scale), -qmax, qmax) * scale


class StaticPTQ(fx.Interpreter):
    def __init__(self, model, boundary_profile="baseline", activation_bits=8):
        graph = fx.symbolic_trace(copy.deepcopy(model).eval())
        super().__init__(graph)
        self.activation_bits = activation_bits
        self.qmax = activation_qmax(activation_bits)
        self.mode = 'observe'
        self.maxima = {}
        self.scales = {}
        self.integer_weights = {}
        self.saturation = {}
        self.nodes = {n.name: n for n in graph.graph.nodes}
        self.configure_boundaries(boundary_profile)

    def configure_boundaries(self, profile):
        self.policy = boundary_policy(self.module, profile)
        self.boundaries = set(self.policy['boundaries'])

    def activation_scale(self, name):
        return self.scales[scale_source(name, self.policy['aliases'])]

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
        clipped = (result.abs() > self.qmax * scale).sum()
        old = self.saturation.get(n.name, (torch.zeros((), device=result.device, dtype=torch.long), 0))
        self.saturation[n.name] = (old[0] + clipped, old[1] + result.numel())
        return qdq(result, scale, self.qmax)

    def freeze(self):
        self.scales = {name: value.clamp_min(1e-12) / self.qmax for name, value in self.maxima.items()}
        for node in self.module.graph.nodes:
            if node.op != 'call_module':
                continue
            layer = self.module.get_submodule(node.target)
            if not isinstance(layer, nn.Conv2d):
                continue
            w = layer.weight.detach()
            sw = w.abs().flatten(1).amax(1).clamp_min(1e-12) / 127
            qw = torch.round(w / sw[:, None, None, None]).clamp(-127, 127).to(torch.int8)
            sx = self.activation_scale(node.args[0].name)
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
        torch.save({'format': f'signed_symmetric_w8a{self.activation_bits}_qdq_v1', 'metadata': metadata,
                    'activation_bits': self.activation_bits, 'weight_bits': 8,
                    'state_dict': {k: v.cpu() for k, v in self.module.state_dict().items()},
                    'activation_scales': {k: v.cpu() for k, v in self.scales.items()},
                    'integer_convs': self.integer_weights,
                    'boundary_policy': self.policy, 'graph': str(self.module.graph)}, path)

    def load_artifact(self, path):
        data = torch.load(path, map_location=next(self.module.parameters()).device)
        self.activation_bits = data.get('activation_bits', 8)
        self.qmax = activation_qmax(self.activation_bits)
        if data.get('weight_bits', 8) != 8:
            raise ValueError('Expected W8 weights')
        self.module.load_state_dict(data['state_dict'], strict=True)
        if str(self.module.graph) != data['graph']:
            raise ValueError('Quantization graph differs from artifact')
        policy = data.get('boundary_policy', boundary_policy(self.module))
        self.configure_boundaries(policy['profile'])
        if policy != self.policy:
            raise ValueError('Artifact boundary policy differs from implementation')
        self.scales = data['activation_scales']
        if set(self.scales) != self.boundaries:
            raise ValueError('Activation encodings do not match boundaries')
        self.integer_weights = data['integer_convs']
        self.mode = 'quantize'
        return data['metadata']

    def statistics(self):
        return {name: {'scale': float(scale), 'absmax': float(scale * self.qmax),
                       'clipped': int(self.saturation[name][0]) if name in self.saturation else 0,
                       'elements': self.saturation.get(name, (0, 0))[1]}
                for name, scale in self.scales.items()}
