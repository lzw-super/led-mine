#!/usr/bin/env python3
import numpy as np
from PIL import Image

# ====== 修改这里 ======
in_path = "./tdm_in_raw10_0000.bin"
out_path = "./tdm_in_raw10_0000_rggb_raw10.png"
W, H = 1280, 720
# =====================

buf = np.fromfile(in_path, dtype=np.uint8)
expected = W * H * 5 // 4  # RAW10 packed
if buf.size != expected:
    raise ValueError(f"size mismatch: got {buf.size}, expected {expected}")

# MIPI RAW10 unpack: every 5 bytes -> 4 pixels (10-bit)
p = buf.reshape(-1, 5).astype(np.uint16)
b0, b1, b2, b3, b4 = p[:, 0], p[:, 1], p[:, 2], p[:, 3], p[:, 4]

px0 = (b0 << 2) | (b4 & 0x03)
px1 = (b1 << 2) | ((b4 >> 2) & 0x03)
px2 = (b2 << 2) | ((b4 >> 4) & 0x03)
px3 = (b3 << 2) | ((b4 >> 6) & 0x03)

raw = np.empty(W * H, dtype=np.uint16)
raw[0::4] = px0
raw[1::4] = px1
raw[2::4] = px2
raw[3::4] = px3
raw = raw.reshape(H, W)

# 10-bit -> 8-bit
raw8 = ((raw.astype(np.uint32) * 255 + 511) // 1023).astype(np.float32)

# RGGB masks
yy, xx = np.indices((H, W))
r_mask = ((yy % 2 == 0) & (xx % 2 == 0)).astype(np.float32)
g_mask = (((yy % 2 == 0) & (xx % 2 == 1)) | ((yy % 2 == 1) & (xx % 2 == 0))).astype(np.float32)
b_mask = ((yy % 2 == 1) & (xx % 2 == 1)).astype(np.float32)

# Simple bilinear-like interpolation (neighbor average)
def interp(channel_mask):
    v = raw8 * channel_mask
    w = channel_mask
    s = v.copy()
    c = w.copy()
    for dy, dx in [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
        s += np.roll(np.roll(v, dy, axis=0), dx, axis=1)
        c += np.roll(np.roll(w, dy, axis=0), dx, axis=1)
    return s / np.maximum(c, 1e-6)

R = interp(r_mask)
G = interp(g_mask)
B = interp(b_mask)

rgb = np.stack([R, G, B], axis=-1)
rgb = np.clip(rgb, 0, 255).astype(np.uint8)

Image.fromarray(rgb, "RGB").save(out_path)
print("saved:", out_path)
print("raw10 min/max:", int(raw.min()), int(raw.max()))