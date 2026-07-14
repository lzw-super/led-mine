import os
import glob
import numpy as np
from tqdm import tqdm


def unpack_mipi_raw10(bin_path, H=720, W=1280):
    """
    将 MIPI RAW10 密集的 5 字节打包格式，完美解码为 10 位 uint16 矩阵 (H, W)
    """
    buf = np.fromfile(bin_path, dtype=np.uint8)
    expected = W * H * 5 // 4  # 720 * 1280 * 5 // 4 = 1152000 字节
    if buf.size != expected:
        raise ValueError(f"文件大小错配: 路径 {bin_path}, 读到 {buf.size} 字节, 预期 {expected} 字节")

    # 每 5 个字节一组进行解包
    p = buf.reshape(-1, 5).astype(np.uint16)
    b0, b1, b2, b3, b4 = p[:, 0], p[:, 1], p[:, 2], p[:, 3], p[:, 4]

    # 位移拼接还原真正的 10bit 物理像素
    px0 = (b0 << 2) | (b4 & 0x03)
    px1 = (b1 << 2) | ((b4 >> 2) & 0x03)
    px2 = (b2 << 2) | ((b4 >> 4) & 0x03)
    px3 = (b3 << 2) | ((b4 >> 6) & 0x03)

    raw = np.empty(W * H, dtype=np.uint16)
    raw[0::4] = px0
    raw[1::4] = px1
    raw[2::4] = px2
    raw[3::4] = px3

    return raw.reshape(H, W)


def pack_rggb_bayer_raw_scale(raw: np.ndarray):
    """ 保持 0~1023 的原始物理尺度，严格按照 RGGB 排列切片 """
    raw = raw.astype(np.float32)
    R = raw[0::2, 0::2]
    G1 = raw[0::2, 1::2]
    G2 = raw[1::2, 0::2]
    B = raw[1::2, 1::2]
    return np.stack((R, G1, B, G2), axis=0).astype(np.float32)


def process_bin_dataset(input_dir, save_dir, bl_val=64.0):
    for child in ['long', 'short']:
        in_child_dir = os.path.join(input_dir, child)
        out_child_dir = os.path.join(save_dir, child)
        os.makedirs(out_child_dir, exist_ok=True)

        bin_files = glob.glob(os.path.join(in_child_dir, "*.bin"))
        print(f"正在解码并处理 {child} 中的 {len(bin_files)} 个 MIPI RAW10 文件...")

        for file_path in tqdm(bin_files):
            try:
                # ?? 核心修正：锁定探测出来的真实物理尺寸 720x1280
                raw_img = unpack_mipi_raw10(file_path, H=720, W=1280)
            except ValueError as e:
                print(e)
                continue

            packed_img = pack_rggb_bayer_raw_scale(raw_img)

            # 10 位物理黑白电平参数（白电平 1023，黑电平推荐先用通用的 64）
            black_level = np.array([bl_val, bl_val, bl_val, bl_val], dtype=np.float32).reshape(4, 1, 1)
            white_level = np.array([1023.0, 1023.0, 1023.0, 1023.0], dtype=np.float32).reshape(4, 1, 1)

            wb = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
            cam2rgb = np.eye(3, dtype=np.float32)

            file_name = os.path.splitext(os.path.basename(file_path))[0]
            np.savez(
                os.path.join(out_child_dir, f"{file_name}.npz"),
                im=packed_img, wb=wb, ccm=cam2rgb,
                black_level=black_level, white_level=white_level
            )
    print("? 10位 MIPI RAW10 (720x1280) 微调数据集打包完美完成！")


if __name__ == '__main__':
    # 填入你微调用的真实 bin 文件夹路径和保存 npz 的路径
    process_bin_dataset(
        input_dir="/home/aa509/Desktop/bin",
        save_dir="/home/aa509/Desktop/npz",
        bl_val=64.0
    ) 