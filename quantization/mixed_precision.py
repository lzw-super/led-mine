"""Phase ablations on opt1; explicit A16 -> A8 casts at phase interfaces."""
from quantization.boundaries import boundary_policy

MIXED_PROFILES = ('none', 'phase_exp1', 'phase_exp2', 'phase_exp3', 'phase_exp4', 'phase_exp5', 'phase_exp6')


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
    casts = ([] if mixed_profile in ('phase_exp5', 'phase_exp6') else ['conv2d_3']) + (['conv2d_4'] if mixed_profile == 'phase_exp1' else [])
    for name in casts:
        policy['aliases'].pop(name)
        policy['removed'].pop(name)
        policy['boundaries'].append(name)
    policy['boundaries'].sort()
    mapping = {name: 16 for name in policy['boundaries']}
    phase2 = [name for name in ('conv2d_3', 'stem', 'head', 'cat_4', 'fusion') if name in mapping] + [f'blocks_{i}_activation' for i in range(5)]
    for name in phase2 + (['conv2d_4'] if mixed_profile == 'phase_exp1' else []):
        mapping[name] = 8
    if mixed_profile in ('phase_exp3', 'phase_exp4', 'phase_exp5', 'phase_exp6'):
        for name in ('head', 'cat_4', 'fusion'):
            mapping[name] = 16
    if mixed_profile in ('phase_exp4', 'phase_exp6'):
        mapping['head'] = 8
    return policy, mapping
