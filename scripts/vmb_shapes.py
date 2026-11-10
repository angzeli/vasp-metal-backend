#!/usr/bin/env python3
"""Shape census of one managed run from its per-rank ZGEMM telemetry.

Merges OUTPUT/vmb-reports/rank-RRRRR.telemetry.tsv (VMB_TELEMETRY=rank) into one
table, reconciles it with the attempt evidence, and states how much ZGEMM time
falls inside candidate GPU regions. The census bounds any possible GPU gain:
time outside a region cannot be accelerated by a rule that selects that region.
It never decides whether the GPU is faster; that needs a benchmark.

Standard library only. Reads files; writes only the explicitly named outputs.

  census   RUN_DIR [--region mn=16384,mnk=1.5e7 ...] [--output NEW.json]
           [--replay NEW.tsv --label LABEL [--role select|holdout] [--top N]]
"""
import argparse
import json
import math
from pathlib import Path
import re
import sys

COLUMNS = ['m', 'n', 'k', 'transa', 'transb', 'lda', 'ldb', 'ldc', 'beta', 'status', 'count',
           'total_s', 'min_s', 'max_s']
KEY = ('m', 'n', 'k', 'transa', 'transb', 'lda', 'ldb', 'ldc', 'beta')
NAME = re.compile(r'rank-([0-9]{5})\.telemetry\.tsv\Z')
DEFAULT_REGION = {'mn': 16384, 'mnk': 1.5e7}  # unshipped v0.2 candidate (RESULTS-v0.2.md)


class CensusError(ValueError):
    pass


def parse_region(text):
    """'mn=16384,mnk=1.5e7' -> {'mn': 16384.0, 'mnk': 1.5e7}; m, n, k, mn, mnk allowed."""
    region = {}
    for part in text.split(','):
        key, sep, value = part.partition('=')
        if not sep or key not in ('m', 'n', 'k', 'mn', 'mnk'):
            raise CensusError(f'bad region term {part!r}')
        region[key] = float(value)
    return region


def in_region(shape, region):
    m, n, k = shape['m'], shape['n'], shape['k']
    values = {'m': m, 'n': n, 'k': k, 'mn': m * n, 'mnk': m * n * k}
    return all(values[key] >= bound for key, bound in region.items())


def read_telemetry(path):
    rows, dropped = [], None
    with open(path) as handle:
        for number, line in enumerate(handle, 1):
            line = line.rstrip('\n')
            if line.startswith('#'):
                found = re.search(r'dropped_shapes=(\d+)', line)
                if found:
                    dropped = int(found.group(1))
                continue
            fields = line.split('\t')
            if fields == COLUMNS:
                continue
            if len(fields) != len(COLUMNS):
                raise CensusError(f'{path.name}:{number}: expected {len(COLUMNS)} columns')
            row = dict(zip(COLUMNS, fields))
            try:
                for key in ('m', 'n', 'k', 'lda', 'ldb', 'ldc', 'status', 'count'):
                    row[key] = int(row[key])
                for key in ('total_s', 'min_s', 'max_s'):
                    row[key] = float(row[key])
            except ValueError:
                raise CensusError(f'{path.name}:{number}: malformed number') from None
            if (row['count'] < 1 or min(row['m'], row['n'], row['k']) < 0
                    or not all(math.isfinite(row[k]) and row[k] >= 0 for k in ('total_s', 'min_s', 'max_s'))
                    or row['transa'] not in 'NTCntc' or row['transb'] not in 'NTCntc'
                    or row['beta'] not in ('zero', 'one', 'other')):
                raise CensusError(f'{path.name}:{number}: invalid values')
            rows.append(row)
    if dropped is None:
        raise CensusError(f'{path.name}: missing telemetry header')
    return rows, dropped


def outcar_loop_seconds(path):
    """Sum of LOOP+ real times (one per ionic step), or None if unavailable."""
    try:
        text = path.read_text(errors='replace')
    except OSError:
        return None
    times = re.findall(r'LOOP\+:\s+cpu time\s+[-\d.]+:\s+real time\s+([-\d.]+)', text)
    return sum(float(t) for t in times) if times else None


def census(run_dir, regions):
    run_dir = Path(run_dir)
    reports = run_dir / 'vmb-reports'
    files = sorted(p for p in reports.iterdir() if NAME.fullmatch(p.name))
    if not files:
        raise CensusError('no rank telemetry files; was VMB_TELEMETRY=rank set?')
    shapes, ranks, dropped = {}, [], 0
    for path in files:
        rows, lost = read_telemetry(path)
        ranks.append(int(NAME.fullmatch(path.name).group(1)))
        dropped += lost
        for row in rows:
            key = tuple(row[k] for k in KEY)
            entry = shapes.setdefault(key, {**{k: row[k] for k in KEY}, 'calls': 0, 'total_s': 0.0,
                                            'ranks': set(), 'statuses': {}})
            entry['calls'] += row['count']
            entry['total_s'] += row['total_s']
            entry['ranks'].add(ranks[-1])
            entry['statuses'][row['status']] = entry['statuses'].get(row['status'], 0) + row['count']
    calls = sum(s['calls'] for s in shapes.values())
    zgemm_s = sum(s['total_s'] for s in shapes.values())
    result = {'schema': 'vmb.shape-census.v1', 'run_dir': str(run_dir), 'ranks': ranks,
              'distinct_shapes': len(shapes), 'calls': calls, 'zgemm_rank_seconds': zgemm_s,
              'dropped_shapes': dropped, 'reconciliation': None, 'wall': None, 'regions': [],
              'decades': []}
    evidence_path = run_dir / 'vmb-evidence.json'
    if evidence_path.exists():
        evidence = json.loads(evidence_path.read_text())
        expected = (evidence.get('totals') or {}).get('calls')
        result['reconciliation'] = {'evidence_status': evidence.get('evidence'), 'evidence_calls': expected,
                                    'telemetry_calls': calls,
                                    'agrees': expected == calls and dropped == 0
                                    and evidence.get('evidence') == 'COMPLETE'}
    meta = {}
    if (run_dir / 'RUN_METADATA.txt').exists():
        for line in (run_dir / 'RUN_METADATA.txt').read_text(errors='replace').splitlines():
            key, sep, value = line.partition(': ')
            if sep:
                meta[key] = value
    elapsed = float(meta['elapsed_s']) if 'elapsed_s' in meta else None
    loops = outcar_loop_seconds(run_dir / 'OUTCAR')
    world = len(ranks)
    if elapsed:
        result['wall'] = {'elapsed_s': elapsed, 'outcar_loop_real_s': loops, 'world_size': world,
                          'zgemm_share_of_rank_time': zgemm_s / (world * elapsed)}
    decades = {}
    for s in shapes.values():
        work = s['m'] * s['n'] * s['k']
        decade = 0 if work < 10 else int(math.floor(math.log10(work)))
        bucket = decades.setdefault(decade, {'decade': decade, 'calls': 0, 'total_s': 0.0})
        bucket['calls'] += s['calls']
        bucket['total_s'] += s['total_s']
    result['decades'] = [decades[d] for d in sorted(decades)]
    for region in regions:
        selected = [s for s in shapes.values() if in_region(s, region)]
        region_s = sum(s['total_s'] for s in selected)
        entry = {'region': region, 'shapes': len(selected), 'calls': sum(s['calls'] for s in selected),
                 'rank_seconds': region_s, 'share_of_zgemm_time': region_s / zgemm_s if zgemm_s else 0.0,
                 'upper_bound_share_of_rank_time': None}
        if elapsed:
            # No rule selecting this region can save more than this fraction of rank time.
            entry['upper_bound_share_of_rank_time'] = region_s / (world * elapsed)
        result['regions'].append(entry)
    ordered = sorted(shapes.values(), key=lambda s: -s['total_s'])
    result['top_shapes'] = [{**{k: s[k] for k in KEY}, 'calls': s['calls'], 'total_s': s['total_s'],
                             'mnk': s['m'] * s['n'] * s['k'], 'ranks': len(s['ranks']),
                             'statuses': {str(k): v for k, v in sorted(s['statuses'].items())}}
                            for s in ordered[:40]]
    result['_shapes'] = ordered
    return result


def replay_rows(shapes, label, role, top, region=None):
    """Rows for bench/shape_replay in the bench/shapes-v0.2.tsv column format."""
    rows = []
    for index, s in enumerate(shapes):
        if region is not None and not in_region(s, region):
            continue
        ta, tb = s['transa'].upper(), s['transb'].upper()
        rows_a = s['m'] if ta == 'N' else s['k']
        rows_b = s['k'] if tb == 'N' else s['n']
        pads = (s['lda'] - rows_a, s['ldb'] - rows_b, s['ldc'] - s['m'])
        if min(pads) < 0 or min(s['m'], s['n'], s['k']) < 1:
            continue
        rows.append([f'{label}-{len(rows) + 1:02d}', label, f'observed-{label}'[:39], role, ta, tb,
                     s['m'], s['n'], s['k'], *pads, s['beta']])
        if len(rows) >= top:
            break
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    c = sub.add_parser('census')
    c.add_argument('run_dir', type=Path)
    c.add_argument('--region', action='append', type=parse_region,
                   help='m/n/k/mn/mnk lower bounds, e.g. mn=16384,mnk=1.5e7 (default: the v0.2 candidate)')
    c.add_argument('--output', type=Path, help='new JSON file; never overwritten')
    c.add_argument('--replay', type=Path, help='new shape file of the most time-consuming shapes')
    c.add_argument('--label', default='observed')
    c.add_argument('--role', choices=('select', 'holdout'), default='select')
    c.add_argument('--top', type=int, default=12)
    args = parser.parse_args(argv)
    try:
        result = census(args.run_dir, args.region or [DEFAULT_REGION])
    except (CensusError, OSError, ValueError) as exc:
        print(f'census failed: {exc}', file=sys.stderr)
        return 1
    shapes = result.pop('_shapes')
    if args.output:
        with open(args.output, 'x') as handle:
            json.dump(result, handle, indent=1, sort_keys=True)
    if args.replay:
        if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,20}', args.label):
            parser.error('--label must be 1-21 lowercase letters, digits or hyphens')
        rows = replay_rows(shapes, args.label, args.role, args.top)
        with open(args.replay, 'x') as handle:
            handle.write(f'# Observed shapes from {args.run_dir.name} ({len(result["ranks"])} ranks), most '
                         'time-consuming first; written by scripts/vmb_shapes.py.\n')
            handle.write('id\tfamily\tprovenance\trole\ttransa\ttransb\tm\tn\tk\tpad_a\tpad_b\tpad_c\tbeta\n')
            for row in rows:
                handle.write('\t'.join(str(v) for v in row) + '\n')
    wall = result['wall'] or {}
    print(f"ranks={len(result['ranks'])} shapes={result['distinct_shapes']} calls={result['calls']} "
          f"zgemm_rank_s={result['zgemm_rank_seconds']:.3f} "
          f"zgemm_share={wall.get('zgemm_share_of_rank_time', float('nan')):.4f} "
          f"reconciled={(result['reconciliation'] or {}).get('agrees')}")
    for entry in result['regions']:
        bound = entry['upper_bound_share_of_rank_time']
        print(f"region {entry['region']}: shapes={entry['shapes']} calls={entry['calls']} "
              f"share_of_zgemm={entry['share_of_zgemm_time']:.4f} "
              f"upper_bound_share_of_rank_time={'n/a' if bound is None else f'{bound:.4f}'}")
    return 0 if (result['reconciliation'] or {}).get('agrees', True) else 3


if __name__ == '__main__':
    sys.exit(main())
