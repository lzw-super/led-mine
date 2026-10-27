"""Export and verify the LED Haar L3 model for fixed-shape mobile inference."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import onnx
import onnxruntime as ort
import torch

from led.archs import build_network
from led.archs.sid_reference.mobile_haar import MobileHaarRepNCB
from scripts.export_led_onnx import load_options
from scripts.prepare_led_deploy_checkpoint import strip_thop_profile_buffers

DEFAULT_EXPERIMENT = ROOT / 'experiments/LED_Pretrain_CVPR20_ELD_LearningDWT_Haar_L3_RepNCB_W32'


def clean_graph(model):
    """Fold constant tensors and remove Identity aliases without approximation."""
    aliases, nodes = {}, []
    outputs = {out.name for out in model.graph.output}
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            while name in aliases:
                name = aliases[name]
            node.input[index] = name
        if node.op_type == 'Identity' and node.output[0] not in outputs:
            aliases[node.output[0]] = node.input[0]
        elif node.op_type == 'Constant' and len(node.attribute) == 1 and node.attribute[0].name == 'value':
            tensor = onnx.TensorProto()
            tensor.CopyFrom(node.attribute[0].t)
            tensor.name = node.output[0]
            model.graph.initializer.append(tensor)
        else:
            nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    used = {name for node in nodes for name in node.input} | outputs
    initializers = [tensor for tensor in model.graph.initializer if tensor.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(initializers)
    return onnx.shape_inference.infer_shapes(model)


def error_stats(expected, actual):
    difference = np.abs(expected - actual)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
    return {'max_abs_error': float(difference.max()), 'mean_abs_error': float(difference.mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_EXPERIMENT / 'models/net_g_latest.pth')
    parser.add_argument('--options', type=Path, default=ROOT / 'options/LED/pretrain/CVPR20_ELD_LearningDWT_Haar_L3_RepNCB_W32.yaml')
    parser.add_argument('--output', type=Path, default=DEFAULT_EXPERIMENT / 'onnx/net_g_latest_mobile_1x4x360x640.onnx')
    parser.add_argument('--height', type=int, default=360)
    parser.add_argument('--width', type=int, default=640)
    args = parser.parse_args()
    if min(args.height, args.width) < 8 or args.height % 8 or args.width % 8:
        parser.error('Packed RAW height and width must be positive multiples of 8')
    torch.set_num_threads(4)
    torch.manual_seed(2026)
    checkpoint_bytes = args.checkpoint.read_bytes()
    # Load the same bytes whose hash is recorded, even if training replaces latest.
    import io
    checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location='cpu')
    weights, removed = strip_thop_profile_buffers(checkpoint['params'])
    reference = build_network(load_options(args.options)['network_g']).cpu().eval()
    reference.load_state_dict(weights, strict=True)
    mobile = MobileHaarRepNCB(reference).eval()
    shape = (1, 4, args.height, args.width)
    sample = torch.rand(shape)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(mobile, sample, str(args.output), opset_version=17,
                          do_constant_folding=True, input_names=['raw'],
                          output_names=['denoised_raw'], dynamic_axes=None)
    model = clean_graph(onnx.load(str(args.output)))
    counts = dict(sorted(Counter(n.op_type for n in model.graph.node).items()))
    allowed = {'Conv', 'ConvTranspose', 'Split', 'Concat', 'Add', 'PRelu', 'Relu'}
    if set(counts) - allowed:
        raise RuntimeError(f'Unexpected operators: {set(counts) - allowed}')
    for tensor in (model.graph.input[0], model.graph.output[0]):
        assert tuple(dim.dim_value for dim in tensor.type.tensor_type.shape.dim) == shape
    metadata = {
        'layout': 'NCHW', 'raw_channel_order': 'R,G1,B,G2',
        'source_sha256': hashlib.sha256(checkpoint_bytes).hexdigest(),
        'source_param_key': 'params',
        'preprocessing': 'Same normalized packed RAW as LED training; no clipping or ISP in graph',
    }
    onnx.helper.set_model_props(model, metadata)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(args.output))
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(args.output), sess_options=session_options,
                                  providers=['CPUExecutionProvider'])
    report = {
        'checkpoint': str(args.checkpoint.resolve()), **metadata,
        'onnx': str(args.output.resolve()), 'shape': shape, 'dtype': 'float32', 'opset': 17,
        'removed_profile_buffers': removed, 'operators': counts,
        'onnx_bytes': args.output.stat().st_size,
        'validation_tolerance': {'rtol': 1e-4, 'atol': 1e-5},
        'versions': {'torch': torch.__version__, 'onnx': onnx.__version__, 'onnxruntime': ort.__version__},
        'checks': {},
    }
    impulse = torch.zeros(shape)
    for channel in range(4):
        impulse[0, channel, channel, channel] = (channel + 1) / 4
        impulse[0, channel, -1-channel, -1-channel] = -(channel + 1) / 4
    cases = {'uniform_0_1': sample, 'signed_low_light': torch.randn(shape) * .02 + .03,
             'boundary_impulses': impulse}
    for name, value in cases.items():
        with torch.no_grad():
            expected = reference(value).numpy()
            converted = mobile(value).numpy()
        actual = session.run(['denoised_raw'], {'raw': value.numpy()})[0]
        report['checks'][name] = {
            'original_vs_mobile': error_stats(expected, converted),
            'original_vs_onnx': error_stats(expected, actual),
            'mobile_vs_onnx': error_stats(converted, actual),
        }
        print(name, report['checks'][name], flush=True)
    report['onnx_sha256'] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Validated ONNX: {args.output}\nOperators: {counts}', flush=True)


if __name__ == '__main__':
    main()
