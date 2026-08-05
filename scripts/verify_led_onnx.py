"""端到端验证导出的 LED ONNX 模型。

`scripts/export_led_onnx.py` 在导出时只用一组随机均匀分布的输入做 PyTorch ↔ ONNX Runtime
的数值一致性校验。本脚本补充更贴近真实推理场景的验证：取真实相机 RAW（ARW/DNG/...），
按 ``scripts/image_process.py`` 的同一套预处理（pack Bayer → 减黑电平 / 除白电平 →
数字增益 ratio → clip）打包成 4 通道网络输入，中心裁剪到 ONNX 的固定输入尺寸后，同时
喂给 PyTorch 原始网络与 ONNX Runtime，比对两者输出的最大 / 平均绝对误差。

若误差与导出时随机输入的误差（~1e-6）同量级，则认为 ONNX 图在真实数据分布下与
PyTorch 推理完全一致。可选地保存「输入 / PyTorch 输出 / ONNX 输出 / 放大差异」的
RGB 四联可视化，便于人工核查。
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch

# 复用已有的网络构建 / 部署态加载逻辑，保证与导出时的网络完全一致。
from scripts.export_led_onnx import load_deployed_network
# 复用 image_process 的 RAW 读取与 ISP 后处理，保证预处理 / 可视化与推理脚本一致。
from scripts.image_process import (
    DEFAULT_CAMERA_RAW_EXTENSIONS,
    find_camera_raw_paths,
    parse_extensions,
    postprocess,
    read_img,
)

# 经验阈值：导出时随机输入的 max abs error 通常在 1e-6 量级，这里放宽到 1e-4，
# 既容忍真实数据下的微小数值漂移，又能可靠捕捉 ONNX 图转换错误。
PARITY_THRESHOLD = 1e-4


def read_onnx_input_shape(onnx_path: Path) -> Tuple[int, int, int, int]:
    """从 ONNX 图读取首个输入的静态 shape (N, C, H, W)。"""
    model = onnx.load(str(onnx_path))
    shape = tuple(dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim)
    if len(shape) != 4 or any(value <= 0 for value in shape):
        raise ValueError(f'ONNX input must be a fully static 4-D shape, got: {shape}')
    return shape


def center_crop(tensor: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """对 (1, C, H, W) 张量做中心裁剪到 (target_h, target_w)。"""
    _, _, height, width = tensor.shape
    if height < target_h or width < target_w:
        raise ValueError(
            f'Packed RAW size ({height}x{width}) smaller than ONNX input ({target_h}x{target_w}). '
            'Pick a larger RAW file or re-export the ONNX with a smaller input shape.')
    top = (height - target_h) // 2
    left = (width - target_w) // 2
    return tensor[:, :, top:top + target_h, left:left + target_w].contiguous()


def parse_args() -> argparse.Namespace:
    extensions_help = ','.join(ext.lstrip('.') for ext in DEFAULT_CAMERA_RAW_EXTENSIONS)
    parser = argparse.ArgumentParser(description='Verify a LED ONNX model against the PyTorch network on real RAW data.')
    parser.add_argument('--onnx', required=True, type=Path, help='ONNX model exported by scripts/export_led_onnx.py.')
    parser.add_argument('-p', '--checkpoint', required=True, type=Path, help='Source LED .pth checkpoint used to build the PyTorch network.')
    parser.add_argument('-opt', '--options', default=Path('options/base/network_g/nafnet_w16_d3_s3.yaml'), type=Path, help='Network option YAML containing network_g.')
    parser.add_argument('--param-key', choices=('auto', 'params', 'params_ema', 'params_deploy', 'params_ema_deploy'), default='auto')
    parser.add_argument('--data_path', default=Path('test_data_mine/raw_type'), type=Path, help='A camera RAW file or a directory holding RAW files.')
    parser.add_argument('--extensions', default=extensions_help, help='Comma-separated camera RAW container extensions.')
    parser.add_argument('--ratio', type=float, default=70.0, help='Digital gain simulating low-light exposure (same meaning as in image_process.py).')
    parser.add_argument('--num_images', type=int, default=3, help='Maximum number of RAW files to verify.')
    parser.add_argument('--save_path', default=Path('tmp/onnx_verify'), type=Path, help='Directory for RGB verification visualizations.')
    parser.add_argument('--diff_amp', type=float, default=10.0, help='Amplification factor for the per-pixel RGB difference panel.')
    parser.add_argument('--no_vis', action='store_true', help='Skip saving RGB verification panels.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    onnx_path = args.onnx.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    options_path = args.options.expanduser().resolve()

    if not onnx_path.is_file():
        raise FileNotFoundError(f'ONNX model does not exist: {onnx_path}')
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'Checkpoint does not exist: {checkpoint_path}')
    if not options_path.is_file():
        raise FileNotFoundError(f'Network options do not exist: {options_path}')

    input_shape = read_onnx_input_shape(onnx_path)
    _, _, input_h, input_w = input_shape
    print(f'ONNX model:     {onnx_path}')
    print(f'ONNX input shape: {input_shape}')

    # PyTorch 网络构建在 CPU 上，与 ONNX Runtime(CPU) 同设备比对，排除跨设备数值疑虑。
    network, source_key, conversion_mode, _ = load_deployed_network(
        checkpoint_path, options_path, args.param_key)
    network.eval()
    session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    print(f'PyTorch network loaded (param_key={source_key}, mode={conversion_mode}).')
    print(f'ONNX Runtime session ready (provider=CPUExecutionProvider).')

    data_path = args.data_path.expanduser()
    try:
        raw_paths = find_camera_raw_paths(data_path, parse_extensions(args.extensions))
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(error)
    if not raw_paths:
        raise SystemExit(f'No camera RAW files found under: {data_path}')
    raw_paths = raw_paths[:args.num_images]
    print(f'Verifying {len(raw_paths)} RAW file(s) under: {data_path}')

    save_vis = not args.no_vis
    if save_vis:
        args.save_path.mkdir(parents=True, exist_ok=True)

    overall_max = 0.0
    for index, raw_path in enumerate(raw_paths, start=1):
        raw_path_str = str(raw_path)
        raw = None
        try:
            raw, raw_pattern, packed, black_level, white_level = read_img(raw_path_str)
            # 与 image_process.py 完全相同的归一化 + 数字增益路径。
            packed = (packed - black_level) / (white_level - black_level)
            packed = (packed * args.ratio).clip(max=torch.tensor(1.0))
            crop = center_crop(packed, input_h, input_w)

            with torch.no_grad():
                torch_output = network(crop).cpu()
            onnx_output = session.run(['denoised_raw'], {'raw': crop.numpy().astype(np.float32)})[0]
            onnx_output = torch.from_numpy(onnx_output)

            abs_diff = (torch_output - onnx_output).abs()
            max_abs = float(abs_diff.max())
            mean_abs = float(abs_diff.mean())
            overall_max = max(overall_max, max_abs)
            print(f'  [{index}/{len(raw_paths)}] {raw_path.name}: crop {tuple(crop.shape)} | '
                  f'max abs {max_abs:.3e} | mean abs {mean_abs:.3e}')

            if save_vis:
                input_rgb = postprocess(raw, raw_pattern, crop.cpu().clip(0, 1), black_level, white_level, 8)
                torch_rgb = postprocess(raw, raw_pattern, torch_output.clip(0, 1), black_level, white_level, 8)
                onnx_rgb = postprocess(raw, raw_pattern, onnx_output.clip(0, 1), black_level, white_level, 8)
                # PyTorch 与 ONNX 输出几乎相同，差异近乎 0；放大后用伪彩色突出显示残差分布。
                diff_rgb = cv2.absdiff(torch_rgb, onnx_rgb)
                diff_amp = np.clip(diff_rgb.astype(np.float32) * args.diff_amp, 0, 255).astype(np.uint8)
                diff_vis = cv2.applyColorMap(diff_amp, cv2.COLORMAP_JET)
                panel = cv2.hconcat([input_rgb, torch_rgb, onnx_rgb, diff_vis])
                out_file = args.save_path / f'{raw_path.stem}.onnx_verify.png'
                if not cv2.imwrite(str(out_file), panel):
                    raise OSError(f'Failed to write {out_file}')
        finally:
            if raw is not None:
                raw.close()

    print(f'Overall max abs error across {len(raw_paths)} image(s): {overall_max:.3e}')
    status = 'PASS' if overall_max < PARITY_THRESHOLD else 'FAIL'
    print(f'Parity test (ONNX Runtime vs PyTorch, threshold {PARITY_THRESHOLD:.0e}): {status}')
    if save_vis:
        print(f'Visualizations saved to: {args.save_path}')
    if status == 'FAIL':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
