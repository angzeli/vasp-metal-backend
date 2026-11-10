#!/usr/bin/env python3
"""Attribute the busy time of a sampled VASP rank to libraries and kernels.

Reads the text written by macOS `sample PID SECONDS 1 -file OUT` and uses its
"Sort by top of stack" section: each line is a leaf symbol, its image and a
sample count. Idle waits (kevent, condition waits, workqueue parking) are
excluded, so shares are of busy time. This is a statistical profile of a short
window, not a census: it says where time goes, not how large each call is.

Standard library only; it only reads the files it is given.

  vmb_profile.py SAMPLE.txt [SAMPLE.txt ...] [--json NEW.json]
"""
import argparse
from collections import Counter
import json
import re
import sys

IDLE = {'kevent', 'kevent64', '__workq_kernreturn', '__psynch_cvwait', 'mach_msg2_trap',
        '__semwait_signal', '__select', 'poll'}
LINE = re.compile(r'^\s+(\S.*?)\s+\(in ([^)]+)\)\s+(\d+)\s*$', re.M)


def category(symbol, image):
    if 'openblas' in image:
        for kernel in ('gemv', 'gemm', 'trsm', 'trmm', 'herk', 'syrk', 'axpy', 'dot', 'ger'):
            if kernel in symbol:
                return f'BLAS {kernel}'
        return 'BLAS/LAPACK other'
    if 'scalapack' in image:
        return 'ScaLAPACK'
    if 'fftw' in image:
        return 'FFTW'
    if re.search(r'mpi|pmix|prrte|open-pal|libopen|libevent', image):
        return 'MPI/runtime'
    if image.startswith('vasp_std'):
        return 'VASP own code'
    if 'Metal' in image or 'AGX' in image or 'IOGPU' in image:
        return 'Metal/GPU driver'
    return 'system/other'


def profile(path):
    text = open(path, errors='replace').read()
    if 'Sort by top of stack' not in text:
        raise ValueError(f'{path}: no "Sort by top of stack" section; not `sample` output?')
    rows = [(s, i, int(n)) for s, i, n in LINE.findall(text.split('Sort by top of stack', 1)[1])]
    busy = [(s, i, n) for s, i, n in rows if s not in IDLE]
    total = sum(n for *_, n in busy)
    if not total:
        raise ValueError(f'{path}: no busy samples')
    cats, symbols = Counter(), Counter()
    for s, i, n in busy:
        cats[category(s, i)] += n
        symbols[f'{s[:60]} [{i}]'] += n
    return {'file': str(path), 'busy_samples': total, 'idle_samples': sum(n for *_, n in rows) - total,
            'categories': {k: n / total for k, n in cats.most_common()},
            'top_symbols': {k: n / total for k, n in symbols.most_common(12)}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('samples', nargs='+')
    parser.add_argument('--json', help='new file; never overwritten')
    args = parser.parse_args(argv)
    try:
        results = [profile(path) for path in args.samples]
    except (OSError, ValueError) as exc:
        print(f'profile failed: {exc}', file=sys.stderr)
        return 1
    for r in results:
        print(f"== {r['file']}: {r['busy_samples']} busy samples ({r['idle_samples']} idle excluded)")
        for name, share in r['categories'].items():
            print(f'   {100 * share:5.1f}%  {name}')
    if args.json:
        with open(args.json, 'x') as handle:
            json.dump(results, handle, indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
