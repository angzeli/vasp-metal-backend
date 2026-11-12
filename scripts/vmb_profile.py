#!/usr/bin/env python3
"""Attribute the busy time of a sampled VASP rank to libraries and kernels.

Reads the text written by macOS `sample PID SECONDS 1 -file OUT` and uses its
"Sort by top of stack" section: each line is a leaf symbol, its image and a
sample count. Idle waits (kevent, condition waits, workqueue parking) are
excluded, so shares are of busy time. This is a statistical profile of a short
window, not a census: it says where time goes, not how large each call is.

Standard library only; it only reads the files it is given.

  vmb_profile.py SAMPLE.txt [SAMPLE.txt ...] [--json NEW.json]
                [--callers REGEX [--frames N]]

--callers (v0.5) walks the "Call graph" section instead: each outermost frame
outside VASP's own image whose "SYMBOL (in IMAGE)" matches REGEX (for example
'gemv' for dgemv_ and its kernels, 'libfftw3' for FFTW) is attributed, with its
inclusive sample count, to the nearest N frames of VASP's own code above it.
Shares are of busy samples. The backend's own interposers (dgemv_, dfftw_* and
so on, linked into the VASP image) are skipped. A frame that VASP reaches through
a tail call is attributed to the caller of the routine that made it.
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
    with open(path, errors='replace') as handle:
        text = handle.read()
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


# The backend's interposers are linked into the VASP image; they are not VASP code.
INTERPOSED = re.compile(r'(dgemv_|dgemm_|zgemm_|dfftw_\w+|vmb_\w+)\Z')
TREE = re.compile(r'^([\s+!:|]*?)(\d+) (\S.*?)\s+\(in ([^)]+)\)')


def callers(path, pattern, frames=1):
    """Samples inside frames matching PATTERN, by the nearest VASP-code callers."""
    with open(path, errors='replace') as handle:
        text = handle.read()
    if 'Call graph:' not in text:
        raise ValueError(f'{path}: no "Call graph" section; not `sample` output?')
    graph = text.split('Call graph:', 1)[1].split('Total number in stack', 1)[0]
    graph = graph.split('Sort by top of stack', 1)[0]
    match = re.compile(pattern)
    stack, by_caller, matched = [], Counter(), 0
    for line in graph.splitlines():
        found = TREE.match(line)
        if not found:
            continue
        depth, count, symbol, image = len(found.group(1)), int(found.group(2)), found.group(3), found.group(4)
        while stack and stack[-1][0] >= depth:
            stack.pop()
        own = image.startswith('vasp_std') and INTERPOSED.match(symbol) is None
        hit = not own and match.search(f'{symbol} (in {image})') is not None
        inside = any(frame[3] for frame in stack)
        stack.append((depth, symbol, own, hit or inside))
        if hit and not inside:
            chain = [f[1] for f in reversed(stack[:-1]) if f[2]][:frames]
            by_caller[' < '.join(chain) if chain else '(no VASP frame)'] += count
            matched += count
    busy = profile(path)['busy_samples']
    return {'file': str(path), 'pattern': pattern, 'busy_samples': busy, 'matched_samples': matched,
            'matched_share': matched / busy,
            'callers': {k: n / busy for k, n in by_caller.most_common(20)}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('samples', nargs='+')
    parser.add_argument('--json', help='new file; never overwritten')
    parser.add_argument('--callers', metavar='REGEX', help='attribute matching frames to VASP callers')
    parser.add_argument('--frames', type=int, default=1, help='VASP frames per caller (default 1)')
    args = parser.parse_args(argv)
    if args.callers:
        try:
            results = [callers(path, args.callers, args.frames) for path in args.samples]
        except (OSError, ValueError, re.error) as exc:
            print(f'profile failed: {exc}', file=sys.stderr)
            return 1
        for r in results:
            print(f"== {r['file']}: {100 * r['matched_share']:.1f}% of {r['busy_samples']} busy samples "
                  f"inside /{r['pattern']}/")
            for name, share in r['callers'].items():
                print(f'   {100 * share:5.1f}%  {name}')
        if args.json:
            with open(args.json, 'x') as handle:
                json.dump(results, handle, indent=1)
        return 0
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
