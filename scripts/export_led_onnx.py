"""Export a LED checkpoint as a fixed-shape ONNX model.

RepNR checkpoints are converted to their regular convolutional deployment form
before export.  This removes the training-only RepNR operators and selects the
generalized (``-1``) branch, which is the branch used for inference.
"""

import argparse
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
import sys
from typing import Any, Mapping, Tuple

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import onnx
import onnxruntime as ort
import torch

from led.archs import build_network
from led.archs.repnr_utils import build_repnr_arch_from_base
from led.utils.options import yaml_load
from scripts.prepare_led_deploy_checkpoint import (
    is_repnr_state_dict,
    select_state_dict,
    strip_thop_profile_buffers,
)

DEFAULT_CONFIG = REPOSITORY_ROOT / 'options/LED/pretrain/CVPR20_ELD_Setting.yaml'


def _merge_options(overrides: Mapping[str, Any], base: Mapping[str, Any]) -> OrderedDict:
    """Return ``base`` with values in ``overrides`` recursively applied."""
    merged = OrderedDict(base)
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge_options(value, merged[key])
        else:
            merged[key] = deepcopy(value)
    return merged


def load_options(options_path: Path) -> OrderedDict:
    """Load an option YAML and all of its ``base`` files relative to this repo."""
    options_path = options_path.resolve()
    options = yaml_load(str(options_path))
    if not isinstance(options, Mapping):
        raise TypeError(f'Expected a mapping in options file: {options_path}')

    options = OrderedDict(options)
    base_paths = options.pop('base', [])
    if isinstance(base_paths, str):
        base_paths = [base_paths]

    merged: OrderedDict = OrderedDict()
    for base_path in base_paths:
        base_path = Path(base_path)
        if not base_path.is_absolute():
            base_path = options_path.parent / base_path if str(base_path).startswith('.') else REPOSITORY_ROOT / base_path
        # Keep the repository's base-file precedence: earlier base files and
        # the current YAML override values from later base files.
        merged = _merge_options(merged, load_options(base_path))
    return _merge_options(options, merged)


def load_deployed_network(
        checkpoint_path: Path, options_path: Path, param_key: str) -> Tuple[torch.nn.Module, str, str, int]:
    """Load a checkpoint and fuse RepNR layers when needed."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f'Checkpoint root must be a mapping: {checkpoint_path}')

    source_param_key, state_dict = select_state_dict(checkpoint, param_key)
    state_dict, removed_thop_keys = strip_thop_profile_buffers(state_dict)
    options = load_options(options_path)
    if 'network_g' not in options:
        raise KeyError(f'network_g is missing from options: {options_path}')
    network = build_network(deepcopy(options['network_g']))

    if is_repnr_state_dict(state_dict):
        if 'repnr_opt' not in options:
            raise KeyError('The RepNR checkpoint needs repnr_opt in its options YAML.')
        repnr_network = build_repnr_arch_from_base(network, **deepcopy(options['repnr_opt']))
        repnr_network.load_state_dict(state_dict, strict=True)
        network = repnr_network.eval().deploy()
        conversion_mode = 'repnr_to_deploy'
    else:
        network.load_state_dict(state_dict, strict=True)
        conversion_mode = 'already_deployed'

    return network.cpu().eval(), source_param_key, conversion_mode, removed_thop_keys


def export_onnx(
        checkpoint_path: Path,
        output_path: Path,
        options_path: Path,
        input_shape: Tuple[int, int, int, int],
        param_key: str,
        opset: int,
) -> Mapping[str, Any]:
    """Export the fixed-shape model and verify its ONNX Runtime result."""
    network, source_key, conversion_mode, removed_thop_keys = load_deployed_network(
        checkpoint_path, options_path, param_key)
    torch.manual_seed(2026)
    sample = torch.rand(input_shape, dtype=torch.float32)

    with torch.no_grad():
        torch_output = network(sample).cpu().numpy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        network,
        sample,
        str(output_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=['raw'],
        output_names=['denoised_raw'],
        dynamic_axes=None,
    )

    onnx_model = onnx.load(str(output_path))
    # The crop after LED's padding is represented as ONNX Slice.  PyTorch 1.13
    # cannot infer its output dimensions in the model metadata, even though
    # this export has a fully static input.  Record the known output shape so
    # deployment tools also see a fixed-size output.
    for dimension, value in zip(onnx_model.graph.output[0].type.tensor_type.shape.dim, torch_output.shape):
        dimension.ClearField('dim_param')
        dimension.dim_value = value
    onnx.save(onnx_model, str(output_path))
    onnx.checker.check_model(onnx_model)
    onnx_input_shape = tuple(
        dim.dim_value for dim in onnx_model.graph.input[0].type.tensor_type.shape.dim)
    onnx_output_shape = tuple(
        dim.dim_value for dim in onnx_model.graph.output[0].type.tensor_type.shape.dim)
    if onnx_input_shape != input_shape:
        raise RuntimeError(f'ONNX input is not static {input_shape}: got {onnx_input_shape}')

    session = ort.InferenceSession(str(output_path), providers=['CPUExecutionProvider'])
    onnx_output = session.run(['denoised_raw'], {'raw': sample.numpy()})[0]
    max_abs_error = float(np.max(np.abs(torch_output - onnx_output)))
    if not np.allclose(torch_output, onnx_output, rtol=1e-4, atol=1e-5):
        raise RuntimeError(f'ONNX Runtime output differs from PyTorch (max abs error: {max_abs_error:.6e}).')

    return {
        'source_param_key': source_key,
        'conversion_mode': conversion_mode,
        'removed_thop_profile_keys': removed_thop_keys,
        'input_shape': onnx_input_shape,
        'output_shape': onnx_output_shape,
        'max_abs_error': max_abs_error,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Export a LED checkpoint to a fixed-shape ONNX model.')
    parser.add_argument('-p', '--checkpoint', required=True, type=Path, help='Input LED .pth checkpoint.')
    parser.add_argument('-o', '--output', type=Path, help='Output .onnx path (defaults beside the checkpoint).')
    parser.add_argument('-opt', '--options', type=Path, default=DEFAULT_CONFIG, help='LED option YAML used to build the network.')
    parser.add_argument('--param-key', choices=('auto', 'params', 'params_ema', 'params_deploy', 'params_ema_deploy'), default='auto')
    parser.add_argument('--input-shape', nargs=4, type=int, default=(1, 4, 256, 256), metavar=('N', 'C', 'H', 'W'))
    parser.add_argument('--opset', type=int, default=17, help='ONNX opset version.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_path = (args.output or checkpoint_path.with_suffix('.onnx')).expanduser().resolve()
    options_path = args.options.expanduser().resolve()
    input_shape = tuple(args.input_shape)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'Checkpoint does not exist: {checkpoint_path}')
    if not options_path.is_file():
        raise FileNotFoundError(f'Options file does not exist: {options_path}')
    if any(dimension <= 0 for dimension in input_shape):
        raise ValueError(f'Input dimensions must be positive: {input_shape}')

    result = export_onnx(
        checkpoint_path, output_path, options_path, input_shape, args.param_key, args.opset)
    print(f'ONNX model: {output_path}')
    print(f'Source parameter key: {result["source_param_key"]}')
    print(f'Conversion mode: {result["conversion_mode"]}')
    print(f'Fixed input shape: {result["input_shape"]}')
    print(f'Output shape: {result["output_shape"]}')
    print(f'PyTorch / ONNX Runtime max abs error: {result["max_abs_error"]:.8e}')


if __name__ == '__main__':
    main()
