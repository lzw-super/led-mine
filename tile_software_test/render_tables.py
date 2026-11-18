"""Render measured inventory and analytical tile cost tables from analysis.json."""
import json
import math
from pathlib import Path
P = Path(__file__).resolve().parent
s = json.loads((P/'analysis.json').read_text())
lines = ['# 输入分辨率统计', '', '尺寸均按高×宽，扫描范围与方法见 README。', '', '| 范围 | 数量 | packed C×H×W | 对应 Bayer H×W |', '|---|---:|---|---|']
for group in ['Sony_train_long_patches','Sony_npy']:
    for shape,count in s[group]['shapes'].items():
        c,h,w=map(int,shape.split('x'))
        lines.append(f'| {group}（磁盘文件） | {count} | {c}×{h}×{w} | {2*h}×{2*w} |')
for name in ['train','valid_100','valid_250','valid_300']:
    for shape,count in s[name]['packed_hw'].items():
        h,w=map(int,shape.split('x'))
        lines.append(f'| {name}（配对条目） | {count} | 4×{h}×{w} | {2*h}×{2*w} |')
lines += ['', '当前实验实际使用 1848 个训练 patch；验证共 93 对（29+29+35），不将共用 GT 重复解释为独立场景。train 列表仅为补充统计，不是当前 RAWGT 的数据入口。全库 2928 包含 short/long，并非 2928 对。所有配对文件存在且输入与 GT shape 相同。', '', '训练 packed 面积为 262,144；整图 packed 面积为 3,030,272，是训练 patch 的约 11.56 倍。', '', '| 单张完整张量（不含批次） | 1 byte/元素 MiB | 2 bytes/元素 MiB | 4 bytes/元素 MiB |', '|---|---:|---:|---:|']
for title,elements in [('输入 4×1424×2128',4*1424*2128),('输入+输出',8*1424*2128),('Refiner 一张 W32 特征 32×712×1064',8*1424*2128),('两张 Refiner 特征',16*1424*2128)]:
    lines.append(f'| {title} | '+ ' | '.join(f'{elements*b/2**20:.3f}' for b in (1,2,4))+' |')
lines += ['', '以上为张量字节数，不是网络峰值内存；未包含权重、累加器及其他中间量。']
(P/'dataset_summary.md').write_text('\n'.join(lines)+'\n')
lines = ['# Tile 候选与静态成本', '', 'T 包含 halo；C=T−2h。单 tile batch=1。所有尺寸为 packed 像素。面积比为处理输入面积/原图面积，不等于实测耗时比。', '', '| T | h | C | 内部 tile 面积比 (T/C)² | 1424×2128 tile 数 | 截断边缘后实际面积比 | 输入 FP16 KiB | 单张 Refiner W32 特征 FP16 KiB |', '|---:|---:|---:|---:|---:|---:|---:|---:|']
for t in [256,384,512,768,1024]:
    for h in [56,64]:
        c=t-2*h
        def axis(n):
            return sum(min(n,start+c+h)-max(0,start-h) for start in range(0,n,c))
        ratio=axis(1424)*axis(2128)/(1424*2128)
        count=math.ceil(1424/c)*math.ceil(2128/c)
        lines.append(f'| {t} | {h} | {c} | {(t/c)**2:.3f} | {count} | {ratio:.3f} | {4*t*t*2/1024:.0f} | {8*t*t*2/1024:.0f} |')
lines += ['', '单张特征表按最大 T×T 计算，实际边缘 tile 可更小；两张特征 ping-pong 容量为最后一列的 2 倍。FP32 容量为表中 2 倍，INT8 为一半（不含更宽累加器）。', '', '推荐起点：512/64/384；最小精确 halo 候选：512/56/400；小内存候选：256/56/144 与 256/64/128。应结合片上缓存寿命分析，而非只看输入容量。', '', '对照：在选定 T 下测试 h=0/32/48（不足 halo）与 h=80（冗余）；它们用于确定误差和冗余成本，不改变 h=56 的结构下界。', '', '原图边界按 README 截断；这里没有裁掉任何输出像素，也没有额外补大整幅图。FPGA 如对所有末块都做完整 T×T 运算，工作量应改为 tile_count×T²/(H×W)，并另行证明有效边界语义。']
(P/'tile_candidates.md').write_text('\n'.join(lines)+'\n')
