"""Run explicit QAT tile/halo pairs using the existing, unchanged benchmark."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_PAIRS = [(80,8), (80,10), (80,16), (88,10), (88,16), (88,20), (96,10)]


def parse_pair(value):
    try:
        tile, halo = map(int, value.split(':'))
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Use TILE:HALO, for example 80:10') from exc
    if tile <= 2*halo or halo < 0 or tile % 2 or halo % 2:
        raise argparse.ArgumentTypeError('Require even tile/halo and tile > 2*halo >= 0')
    return tile, halo


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        'Other options are forwarded to benchmark_qat_crop.py, including '
        '--artifact, --device, --batch-size, --serial-repeats, '
        '--max-per-ratio, --resume and --check-only.'))
    parser.add_argument('--pairs', nargs='+', type=parse_pair, default=DEFAULT_PAIRS,
                        metavar='TILE:HALO', help='Explicit pairs, not a Cartesian product.')
    parser.add_argument('--output', type=Path, default=ROOT/'tile_software_test/results_qat_extra7')
    args, forwarded = parser.parse_known_args()
    if len(set(args.pairs)) != len(args.pairs):
        parser.error('Duplicate pairs are not allowed')
    from tile_software_test import benchmark_qat_crop as benchmark
    # CONFIGS is included in the original benchmark's strict resume signature.
    # Do not edit the original runner: existing results retain their code hash.
    benchmark.CONFIGS = list(args.pairs)
    sys.argv = [sys.argv[0], '--output', str(args.output), *forwarded]
    benchmark.main()


if __name__ == '__main__':
    main()
