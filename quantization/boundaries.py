"""Explicit Q/DQ boundary ablations; graph arithmetic and master weights are unchanged."""
import torch
from torch import nn, fx

PROFILES = ('baseline', 'opt1', 'opt2', 'opt12')


def boundary_policy(graph, profile='baseline'):
    if profile not in PROFILES:
        raise ValueError('Unknown boundary profile: '+profile)
    boundaries = {n.name for n in graph.graph.nodes
                  if n.op not in ('get_attr', 'output') and
                  not (n.op == 'call_method' and n.target == 'split')}
    aliases, removed = {}, {}
    for node in graph.graph.nodes:
        if profile in ('opt1', 'opt12'):
            if node.op == 'call_module' and isinstance(graph.get_submodule(node.target), nn.Conv2d):
                users = list(node.users)
                if len(users) == 1 and users[0].op == 'call_module' and isinstance(
                        graph.get_submodule(users[0].target), (nn.ReLU, nn.PReLU)):
                    removed[node.name] = 'Conv accumulator -> activation -> one A8 quantizer'
            if node.op == 'call_function' and node.target in (torch.conv2d, torch.conv_transpose2d):
                weight = node.args[1]
                if isinstance(weight, fx.Node) and weight.op == 'get_attr' and weight.target in (
                        'space_weight', 'input_space_weight', 'output_space_weight'):
                    # Only this model's validated 0/1 permutation kernels qualify.
                    value = getattr(graph, weight.target)
                    flat = value.flatten(1)
                    if not torch.all((value == 0) | (value == 1)) or not torch.all(flat.sum(1) == 1):
                        raise ValueError('Expected one-hot space/depth rearrangement')
                    aliases[node.name] = node.args[0].name
                    removed[node.name] = 'Exact rearrangement; inherit input encoding'
        if profile in ('opt2', 'opt12') and node.op == 'call_function':
            if node.target == torch.conv2d:
                weight = node.args[1]
                if isinstance(weight, fx.Node) and weight.op == 'get_attr' and weight.target in ('haar_weight', 'input_haar_weight'):
                    users = list(node.users)
                    if len(users) != 1 or users[0].target != 'split':
                        raise ValueError('Expected Haar analysis followed by LL/HF split')
                    removed[node.name] = 'Wide Haar analysis; quantize LL and HF separately after split'
            if node.target == torch.cat:
                users = list(node.users)
                if len(users) == 1 and users[0].op == 'call_function' and users[0].target == torch.conv_transpose2d:
                    weight = users[0].args[1]
                    if isinstance(weight, fx.Node) and weight.op == 'get_attr' and weight.target == 'haar_weight':
                        removed[node.name] = 'IWT consumes separately scaled LL/HF; wide sum then output A8'
    boundaries.difference_update(removed)
    return dict(profile=profile, boundaries=sorted(boundaries), aliases=aliases, removed=removed)


def scale_source(name, aliases):
    seen = set()
    while name in aliases:
        if name in seen:
            raise ValueError('Cyclic encoding aliases')
        seen.add(name)
        name = aliases[name]
    return name
