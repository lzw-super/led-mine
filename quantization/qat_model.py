"""LSQ-style signed W8A8 QAT on the exact static-PTQ FX boundaries."""
import copy
import math
import torch
from torch import nn, fx
from torch.nn import functional as F
from quantization.static_ptq import StaticPTQ


class LearnedFakeQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        u = x / scale
        ctx.save_for_backward(u, scale)
        ctx.elements_per_scale = x.numel() / scale.numel()
        return u.round().clamp(-127, 127) * scale

    @staticmethod
    def backward(ctx, grad):
        u, scale = ctx.saved_tensors
        inside = (u >= -127) & (u <= 127)
        # STE through round, clipping gradient outside the representable range.
        dx = grad * inside
        ds_local = torch.where(u < -127, -127., torch.where(u > 127, 127., u.round() - u))
        ds = (grad * ds_local).sum_to_size(scale.shape)
        ds = ds / math.sqrt(ctx.elements_per_scale * 127)
        return dx, ds


class QATInterpreter(fx.Interpreter):
    def __init__(self, owner):
        super().__init__(owner.graph)
        self.owner = owner

    def run_node(self, node):
        owner = self.owner
        if node.name in owner.weight_log_scales:
            layer = self.module.get_submodule(node.target)
            args, _ = self.fetch_args_kwargs_from_env(node)
            sw = owner.weight_scale(node.name).view(-1, 1, 1, 1)
            weight = LearnedFakeQuant.apply(layer.weight, sw)
            bias = layer.bias
            if bias is not None:
                sx = owner.activation_scale(node.args[0].name).detach()
                sb = sx * sw.detach().flatten()
                # Bias gradient is STE; bias encodings follow frozen/input scales.
                integer = (bias.detach() / sb).round().clamp(-(2**31), 2**31-1)
                bias = integer * sb + (bias - bias.detach())
            result = F.conv2d(args[0], weight, bias, layer.stride, layer.padding, layer.dilation, layer.groups)
        else:
            result = super().run_node(node)
        if node.name in owner.activation_log_scales:
            result = LearnedFakeQuant.apply(result, owner.activation_scale(node.name))
        return result


class QATModel(nn.Module):
    def __init__(self, mobile, ptq_artifact):
        super().__init__()
        traced = fx.symbolic_trace(copy.deepcopy(mobile).eval())
        if str(traced.graph) != ptq_artifact['graph']:
            raise ValueError('FP32 model and PTQ artifact graph mismatch')
        self.graph = traced
        device = next(traced.parameters()).device
        self.activation_log_scales = nn.ParameterDict({
            name: nn.Parameter(torch.zeros_like(value, device=device))
            for name, value in ptq_artifact['activation_scales'].items()})
        for name, value in ptq_artifact['activation_scales'].items():
            self.register_buffer('act_base__'+name, value.detach().to(device).clone())
        self.weight_log_scales = nn.ParameterDict()
        for node in traced.graph.nodes:
            if node.op == 'call_module' and isinstance(traced.get_submodule(node.target), nn.Conv2d):
                scale = ptq_artifact['integer_convs'][node.target]['weight_scale']
                self.weight_log_scales[node.name] = nn.Parameter(torch.zeros_like(scale, device=device))
                self.register_buffer('weight_base__'+node.name, scale.to(device).clone())
        self.engine = QATInterpreter(self)

    def activation_scale(self, name):
        return getattr(self, 'act_base__'+name) * self.activation_log_scales[name].exp()

    def weight_scale(self, name):
        return getattr(self, 'weight_base__'+name) * self.weight_log_scales[name].exp()

    def forward(self, x):
        if x.shape[-2] % 8 or x.shape[-1] % 8:
            raise ValueError('Packed RAW spatial dimensions must be multiples of eight')
        return self.engine.run(x)

    def scale_parameters(self):
        return list(self.activation_log_scales.parameters()) + list(self.weight_log_scales.parameters())

    def freeze_scales(self):
        for parameter in self.scale_parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def clamp_scales(self):
        # Numerical guard, not calibration/clipping-range search.
        for prefix, params in [('act_base__', self.activation_log_scales), ('weight_base__', self.weight_log_scales)]:
            for name, value in params.items():
                base_log = getattr(self, prefix+name).log()
                value.copy_((value+base_log).clamp(math.log(1e-12), math.log(1e3))-base_log)

    @torch.no_grad()
    def export_qdq(self, path, metadata):
        """Export learned encodings without recalibrating or replacing their scales."""
        runner = StaticPTQ(self.graph)
        if str(runner.module.graph) != str(self.graph.graph):
            raise ValueError('Export retracing changed graph')
        runner.scales = {k: self.activation_scale(k).detach().clone() for k in self.activation_log_scales}
        for node in self.graph.graph.nodes:
            if node.name not in self.weight_log_scales:
                continue
            source = self.graph.get_submodule(node.target)
            target = runner.module.get_submodule(node.target)
            sw = self.weight_scale(node.name).detach()
            qw = (source.weight / sw[:, None, None, None]).round().clamp(-127, 127).to(torch.int8)
            sx = runner.scales[node.args[0].name]
            record = dict(weight_int8=qw.cpu(), weight_scale=sw.cpu(), input_scale=sx.cpu())
            target.weight.copy_(qw.to(source.weight.dtype) * sw[:, None, None, None])
            if source.bias is not None:
                sb = sx * sw
                qb = (source.bias / sb).round()
                if (qb < -(2**31)).any() or (qb > 2**31-1).any():
                    raise OverflowError('INT32 bias overflow at '+node.target)
                target.bias.copy_(qb * sb)
                record.update(bias_int32=qb.to(torch.int32).cpu(), bias_scale=sb.cpu())
            runner.integer_weights[node.target] = record
        runner.mode = 'quantize'
        runner.export(path, metadata)
        return runner
