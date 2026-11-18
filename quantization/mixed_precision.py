"""Phase ablations on opt1; explicit A16 -> A8 casts at phase interfaces."""
from torch import nn
from quantization.boundaries import boundary_policy

MIXED_PROFILES = ('none', 'phase_exp1', 'phase_exp2', 'phase_exp3', 'phase_exp4', 'phase_exp5', 'phase_exp6', 'phase_exp3_pres2d')


def prepare_precision_graph(graph, mixed_profile):
    """Materialize the A8 boundary before S2D as an explicit parameter-free node."""
    nodes = {n.name: n for n in graph.graph.nodes}
    cast = nodes.get('phase2_input_a8')
    if mixed_profile == 'phase_exp3_pres2d':
        if cast is None:
            s2d = nodes['conv2d_3']
            if s2d.args[0].name != 'conv_transpose2d_2':
                raise ValueError('Expected Phase 1 reconstruction feeding S2D')
            graph.add_submodule('phase2_input_a8', nn.Identity())
            with graph.graph.inserting_after(s2d.args[0]):
                cast = graph.graph.call_module('phase2_input_a8', (s2d.args[0],))
            s2d.replace_input_with(s2d.args[0], cast)
    elif cast is not None:
        cast.replace_all_uses_with(cast.args[0])
        graph.graph.erase_node(cast)
        graph.delete_submodule('phase2_input_a8')
    graph.graph.lint()
    graph.recompile()
    return graph


def precision_policy(graph, boundary_profile, bits, mixed_profile='none'):
    policy = boundary_policy(graph, boundary_profile)
    if mixed_profile not in MIXED_PROFILES:
        raise ValueError('Unknown mixed precision profile')
    if mixed_profile == 'none':
        return policy, {name: bits for name in policy['boundaries']}
    if boundary_profile != 'opt1' or bits != 16:
        raise ValueError('Phase ablations require --boundary-profile opt1 --activation-bits 16')
    required = {'raw', 'conv2d_3', 'conv2d_4', 'conv_transpose2d_2', 'conv_transpose2d_3', 'stem', 'head', 'cat_4', 'fusion'}
    if not required.issubset({n.name for n in graph.graph.nodes}):
        raise ValueError('Expected LearningDWT phase graph')
    # The boundary casts are necessary precision changes, not redundant S2D rounding.
    casts = ([] if mixed_profile in ('phase_exp5', 'phase_exp6', 'phase_exp3_pres2d') else ['conv2d_3']) + (['conv2d_4'] if mixed_profile == 'phase_exp1' else [])
    for name in casts:
        policy['aliases'].pop(name)
        policy['removed'].pop(name)
        policy['boundaries'].append(name)
    policy['boundaries'].sort()
    mapping = {name: 16 for name in policy['boundaries']}
    phase2 = [name for name in ('conv2d_3', 'stem', 'head', 'cat_4', 'fusion') if name in mapping] + [f'blocks_{i}_activation' for i in range(5)]
    for name in phase2 + (['conv2d_4'] if mixed_profile == 'phase_exp1' else []):
        mapping[name] = 8
    if mixed_profile in ('phase_exp3', 'phase_exp4', 'phase_exp5', 'phase_exp6', 'phase_exp3_pres2d'):
        for name in ('head', 'cat_4', 'fusion'):
            mapping[name] = 16
    if mixed_profile in ('phase_exp4', 'phase_exp6'):
        mapping['head'] = 8
    if mixed_profile == 'phase_exp3_pres2d':
        if policy['aliases'].get('conv2d_3') != 'phase2_input_a8':
            raise ValueError('Missing explicit pre-S2D cast')
        mapping['phase2_input_a8'] = 8
    return policy, mapping
