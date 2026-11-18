"""Read NPZ array headers only; reproduce inventory and exact spatial support."""
import ast
import csv
import json
import struct
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent

def header(path):
    with zipfile.ZipFile(path) as z, z.open('im.npy') as f:
        assert f.read(6) == b'\x93NUMPY'
        major, minor = f.read(2)
        n = struct.unpack('<H' if major == 1 else '<I', f.read(2 if major == 1 else 4))[0]
        return ast.literal_eval(f.read(n).decode('utf-8' if major == 3 else 'latin1'))

def support(q):
    # Refiner: S2D2 -> seven 3x3 layers -> D2S2; LL3: five 3x3.
    lo, hi = 2*(q//2)-14, 2*(q//2)+15
    return 8*(lo//8)-40, 8*(hi//8)+47

def main():
    root = ROOT/'datasets/ICCV23-LED'
    rows, cache = [], {}
    for group in ['Sony_train_long_patches', 'Sony_npy']:
        for p in sorted((root/group).rglob('*.npz')):
            h = header(p)
            c, height, width = h['shape']
            assert c == 4
            cache[p.resolve()] = (height, width)
            rows.append(dict(path=str(p.relative_to(ROOT)), group=group, channels=c,
                             height=height, width=width, bayer_height=height*2,
                             bayer_width=width*2, dtype=h['descr']))
    with (OUT/'image_inventory.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    summary = {}
    for group in ['Sony_train_long_patches', 'Sony_npy']:
        subset = [r for r in rows if r['group'] == group]
        summary[group] = dict(files=len(subset), shapes=dict(Counter(f"{r['channels']}x{r['height']}x{r['width']}" for r in subset)))
    for name in ['train', 'valid_100', 'valid_250', 'valid_300']:
        pairs = [s.split()[:2] for s in (ROOT/f'datasets/txtfiles/SID/SonyA7S2/{name}.txt').read_text().splitlines() if s.strip()]
        shapes, missing, lqs, gts = Counter(), [], set(), set()
        for lq, gt in pairs:
            a, b = [(root/'Sony_npy'/(s+'.npz')).resolve() for s in (lq,gt)]
            lqs.add(str(a)); gts.add(str(b))
            if a not in cache or b not in cache:
                missing.append([str(a),str(b)]); continue
            assert cache[a] == cache[b]
            shapes['x'.join(map(str,cache[a]))] += 1
        summary[name] = dict(pairs=len(pairs), unique_lq=len(lqs), unique_gt=len(gts), packed_hw=dict(shapes), missing=missing)
    phases = [dict(phase=q, lo=support(q)[0], hi=support(q)[1], left=q-support(q)[0], right=support(q)[1]-q, length=support(q)[1]-support(q)[0]+1) for q in range(8)]
    summary['support_phases'] = phases
    # Exhaustive finite period check, all aligned cores from 8 through 1024.
    valid = []
    for halo in range(0, 81, 8):
        if all(all(support(q)[0] >= -halo and support(q)[1] < core+halo for q in range(core)) for core in range(8,1025,8)):
            valid.append(halo)
    assert min(valid) == 56
    summary['minimum_aligned_halo'] = min(valid)
    (OUT/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__ == '__main__':
    main()
