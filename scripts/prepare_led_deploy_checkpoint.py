"""将 LED 的 RepNR 或部署态 checkpoint 统一为 ``params_deploy`` 格式。"""

import argparse
import os
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import torch

from led.archs import build_network
from led.archs.repnr_utils import build_repnr_arch_from_base
from led.utils.options import yaml_load


VALID_PARAM_KEYS = ('auto', 'params', 'params_ema', 'params_deploy', 'params_ema_deploy')
AUTO_PARAM_KEYS = ('params_ema_deploy', 'params_deploy', 'params_ema', 'params')


def strip_module_prefix(state_dict: Mapping[str, torch.Tensor]) -> OrderedDict:
    """移除 DataParallel/DDP 保存时可能出现的 ``module.`` 前缀。"""
    return OrderedDict(
        (key[7:] if key.startswith('module.') else key, value) for key, value in state_dict.items())


def select_state_dict(checkpoint: Mapping[str, object], param_key: str) -> Tuple[str, OrderedDict]:
    """按指定键或自动优先级从 checkpoint 中选取一个模型 state dict。"""
    if param_key not in VALID_PARAM_KEYS:
        raise ValueError(f'Unsupported param key: {param_key}. Choices: {", ".join(VALID_PARAM_KEYS)}')

    if param_key == 'auto':
        selected_key = next((key for key in AUTO_PARAM_KEYS if key in checkpoint), None)
        if selected_key is None:
            available = ', '.join(checkpoint.keys())
            raise KeyError(f'No supported parameter key found. Available keys: {available}')
    else:
        selected_key = param_key

    state_dict = checkpoint.get(selected_key)
    if not isinstance(state_dict, Mapping):
        raise KeyError(f'Checkpoint key {selected_key!r} does not contain a state dict.')
    return selected_key, strip_module_prefix(state_dict)


def is_repnr_state_dict(state_dict: Mapping[str, torch.Tensor]) -> bool:
    """RepNR 卷积层以 ``main_weight`` 代替普通 Conv2d 的 ``weight``。"""
    return any('.main_weight' in key for key in state_dict)


def build_base_network(options_path: Path):
    """根据网络 YAML 构建普通（部署态）基础网络。"""
    options = yaml_load(str(options_path))
    if not isinstance(options, Mapping) or 'network_g' not in options:
        raise KeyError(f'Network options must contain network_g: {options_path}')
    return build_network(deepcopy(options['network_g'])), options


def _deploy_repnr_state(
        state_dict: Mapping[str, torch.Tensor], options_path: Path) -> Tuple[OrderedDict, float]:
    """加载 RepNR 权重，将泛化分支（-1）融合到普通基础网络。"""
    base_network, options = build_base_network(options_path)
    if 'repnr_opt' not in options:
        raise KeyError(f'RepNR checkpoint requires repnr_opt in network options: {options_path}')

    repnr_network = build_repnr_arch_from_base(base_network, **deepcopy(options['repnr_opt']))
    repnr_network.load_state_dict(state_dict, strict=True)
    repnr_network.eval()

    # deploy() 内部同样使用 -1 分支；随机输入用于验证融合前后数值一致。
    generator = torch.Generator().manual_seed(2022)
    sample = torch.rand((1, 4, 64, 64), generator=generator)
    with torch.no_grad():
        repnr_output = repnr_network(sample, -1)
        deployed_network = repnr_network.deploy().eval()
        deployed_output = deployed_network(sample)
        max_abs_error = (repnr_output - deployed_output).abs().max().item()

    return OrderedDict(deployed_network.state_dict()), max_abs_error


def _load_base_state(
        state_dict: Mapping[str, torch.Tensor], options_path: Path) -> OrderedDict:
    """严格加载已部署/普通网络，保证 YAML 与 checkpoint 匹配。"""
    base_network, _ = build_base_network(options_path)
    base_network.load_state_dict(state_dict, strict=True)
    return OrderedDict(base_network.state_dict())


def prepare_deploy_checkpoint(input_path: Path,
                              output_path: Path,
                              options_path: Path,
                              param_key: str = 'auto') -> Dict[str, object]:
    """规范化 checkpoint，并返回源键、转换方式与 RepNR 融合误差。"""
    input_path = Path(input_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    options_path = Path(options_path).expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f'Input checkpoint does not exist: {input_path}')
    if not options_path.is_file():
        raise FileNotFoundError(f'Network options do not exist: {options_path}')
    if input_path == output_path:
        raise ValueError('Input and output checkpoint paths must be different.')

    checkpoint = torch.load(input_path, map_location='cpu')
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f'Checkpoint root must be a mapping: {input_path}')
    source_param_key, state_dict = select_state_dict(checkpoint, param_key)

    if is_repnr_state_dict(state_dict):
        deploy_state_dict, max_abs_error = _deploy_repnr_state(state_dict, options_path)
        conversion_mode = 'repnr_to_deploy'
    else:
        deploy_state_dict = _load_base_state(state_dict, options_path)
        max_abs_error = None
        conversion_mode = 'already_deployed' if source_param_key.endswith('_deploy') else 'base_to_deploy'

    payload = {
        'params_deploy': deploy_state_dict,
        'source_checkpoint': str(input_path),
        'source_param_key': source_param_key,
        'conversion_mode': conversion_mode,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f'.{output_path.name}.tmp')
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return {
        'output_path': str(output_path),
        'source_param_key': source_param_key,
        'conversion_mode': conversion_mode,
        'max_abs_error': max_abs_error,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description='Normalize a LED RepNR or deployed checkpoint to the params_deploy format.')
    parser.add_argument('-p', '--pretrained_network', required=True, help='Input RepNR or deployed checkpoint.')
    parser.add_argument('-o', '--output_checkpoint', required=True, help='Output checkpoint with params_deploy.')
    parser.add_argument(
        '-opt', '--network_options', default='options/base/network_g/repnr_unet.yaml',
        help='YAML containing network_g and, for RepNR input, repnr_opt.')
    parser.add_argument('--param_key', choices=VALID_PARAM_KEYS, default='auto', help='Input checkpoint key.')
    return parser.parse_args()


def main():
    args = parse_args()
    result = prepare_deploy_checkpoint(
        args.pretrained_network, args.output_checkpoint, args.network_options, args.param_key)
    print(f"Selected checkpoint key: {result['source_param_key']}")
    print(f"Conversion mode: {result['conversion_mode']}")
    if result['max_abs_error'] is not None:
        print(f"RepNR deploy max abs error: {result['max_abs_error']:.8e}")
    print(f"Deployment checkpoint: {result['output_path']}")


if __name__ == '__main__':
    main()
