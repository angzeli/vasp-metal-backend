#!/usr/bin/env python3
"""Strict aggregation of per-rank backend reports into one attempt verdict.

Contract: docs/CONTRACT-v0.3.md, section B. Standard library only; it reads the
report directory and never executes anything. Every rank report is validated on
its own before any counter is summed. The verdicts are kept separate: evidence
completeness, CPU-only or GPU-consumed execution, shadow-check state and GPU
qualification. None of them says anything about speed or scientific validity.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import stat
import sys

SCHEMA = 'vmb.attempt-evidence.v1'
REPORT_SCHEMA = 'vmb.rank-report.v1'
CLAIM_SCHEMA = 'vmb.rank-claim.v1'
PROFILE_SCHEMA = 'vmb.rank-profile.v1'
PROFILE_LEVELS = ('profile', 'reuse')
COUNTERS = ('calls', 'gpu_submitted', 'gpu_completed', 'consumed', 'cpu_calls', 'checked')
STATUSES = ('ok', 'disabled', 'trivial', 'badarg', 'nonfinite', 'too_small', 'too_large',
            'no_device', 'gpu_error', 'gpu_nonfinite', 'check_failed', 'range', 'auto_cpu')
MODES = ('off', 'auto', 'force')
ATTEMPT = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z')
NAME = re.compile(r'rank-([0-9]{5})\.(claim\.json|report\.json|telemetry\.tsv|profile\.json)\Z')
MAX_REPORT_BYTES = 64 * 1024
MAX_PROFILE_BYTES = 64 * 1024 * 1024  # v0.5 profile files hold shape tables
MAX_FILES = 4 * 4096 + 16
# Fields every rank must share; mixing any of them means mixed attempts/binaries.
SHARED = ('backend', 'mode', 'check', 'world_size', 'attempt_id', 'hostname', 'executable',
          'executable_stat')


class EvidenceError(ValueError):
    pass


def _require(condition, message):
    if not condition:
        raise EvidenceError(message)


def _count(value, label):
    _require(type(value) is int and value >= 0, f'{label} must be a non-negative integer')
    return value


def _reject_constant(token):
    raise EvidenceError(f'non-finite JSON constant {token}')


def _read_json(path, limit=MAX_REPORT_BYTES):
    info = path.lstat()
    _require(stat.S_ISREG(info.st_mode), f'{path.name} is not a regular file')
    _require(info.st_size <= limit, f'{path.name} exceeds {limit} bytes')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as handle:
        raw = handle.read(limit + 1)
    try:
        return json.loads(raw.decode('utf-8'), parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f'{path.name} is truncated or malformed: {exc}') from None


def check_rank_report(report, rank, attempt_id, world_size):
    """Validate one report against its file name and the expected attempt."""
    _require(isinstance(report, dict) and report.get('schema') == REPORT_SCHEMA
             and report.get('complete') is True, 'not a complete vmb.rank-report.v1')
    _require(report.get('rank') == rank, 'report rank differs from its file name')
    _require(report.get('world_size') == world_size, 'report world size differs from the attempt')
    _require(report.get('attempt_id') == attempt_id, 'report belongs to another attempt')
    backend = report.get('backend')
    _require(isinstance(backend, dict) and backend.get('name') == 'vasp-metal-backend'
             and isinstance(backend.get('version'), str) and isinstance(backend.get('source_commit'), str)
             and isinstance(backend.get('source_digest'), str)
             and isinstance(backend.get('source_dirty'), bool), 'malformed backend identity')
    _require(report.get('mode') in MODES and isinstance(report.get('check'), bool),
             'malformed mode or check setting')
    process = report.get('process')
    _require(isinstance(process, dict) and type(process.get('pid')) is int and process['pid'] > 0
             and isinstance(process.get('hostname'), str) and process['hostname']
             and isinstance(process.get('executable'), str) and process['executable'].startswith('/'),
             'malformed process identity')
    exe_stat = process.get('executable_stat')
    _require(isinstance(exe_stat, list) and len(exe_stat) == 4
             and all(type(v) is int and v >= 0 for v in exe_stat), 'malformed executable identity')
    times = report.get('times')
    _require(isinstance(times, dict) and all(type(times.get(k)) in (int, float)
             and math.isfinite(times[k]) and times[k] > 0 for k in ('start_unix', 'end_unix'))
             and times['start_unix'] <= times['end_unix'], 'malformed or non-finite times')
    counters, statuses = report.get('counters'), report.get('statuses')
    _require(isinstance(counters, dict) and set(counters) == set(COUNTERS), 'malformed counters')
    _require(isinstance(statuses, dict) and set(statuses) == set(STATUSES), 'malformed statuses')
    c = {k: _count(counters[k], 'counter ' + k) for k in COUNTERS}
    s = {k: _count(statuses[k], 'status ' + k) for k in STATUSES}
    _require(c['calls'] == c['consumed'] + c['cpu_calls'], 'calls != consumed + cpu_calls')
    _require(c['gpu_submitted'] >= c['gpu_completed'] >= c['consumed'],
             'submitted >= completed >= consumed violated')
    _require(sum(s.values()) == c['calls'], 'outcome totals != calls')
    _require(s['ok'] == c['consumed'], 'status.ok != consumed')
    _require(c['gpu_submitted'] - c['gpu_completed'] == s['gpu_error'],
             'submitted - completed != gpu_error')
    _require(c['gpu_completed'] - c['consumed'] == s['gpu_nonfinite'] + s['check_failed'],
             'completed - consumed != gpu_nonfinite + check_failed')
    if report['check']:
        _require(c['checked'] == c['consumed'] + s['check_failed'],
                 'checked != consumed + check_failed')
    else:
        _require(c['checked'] == 0 and s['check_failed'] == 0, 'checks recorded with the check off')
    if report['mode'] == 'off':
        _require(c['gpu_submitted'] == 0 and c['calls'] == s['disabled'], 'off mode used the GPU')
    state, ratio = report.get('max_check_ratio_state'), report.get('max_check_ratio')
    if c['checked'] == 0:
        _require(state == 'none' and ratio is None, 'check ratio recorded without checks')
    elif state == 'finite':
        _require(type(ratio) in (int, float) and math.isfinite(ratio) and ratio >= 0,
                 'malformed finite check ratio')
    else:
        _require(state == 'nonfinite' and ratio is None, 'malformed check ratio state')
    timing, decades = report.get('timing'), report.get('work_decades')
    if timing is not None or decades is not None:  # optional since v0.4
        _require(isinstance(timing, dict) and set(timing) == {'try_s', 'cpu_s'}
                 and all(type(timing[k]) in (int, float) and math.isfinite(timing[k]) and timing[k] >= 0
                         for k in timing), 'malformed timing')
        _require(isinstance(decades, list) and len(decades) == 16
                 and all(type(v) is int and v >= 0 for v in decades) and sum(decades) == c['calls'],
                 'malformed work_decades or total differs from calls')
    return c, s


def check_profile(report, profile, rank, attempt_id, world_size):
    """Optional v0.5 profile file: declared in the report, bound to the same process.

    Only its identity is checked here; the profile never changes a verdict."""
    record = report.get('profile')
    if record is None:
        _require(profile is None, 'profile file without a report declaration')
        return
    _require(isinstance(record, dict) and set(record) == {'file', 'level'}
             and record['level'] in PROFILE_LEVELS, 'malformed profile declaration')
    if record['file'] is None:
        _require(profile is None, 'profile file present but declared unpublished')
        raise EvidenceError('declared profile was not published')
    _require(record['file'] == f'rank-{rank:05d}.profile.json' and profile is not None,
             'declared profile file is missing')
    _require(isinstance(profile, dict) and profile.get('schema') == PROFILE_SCHEMA
             and profile.get('complete') is True, 'not a complete vmb.rank-profile.v1')
    _require(profile.get('attempt_id') == attempt_id and profile.get('rank') == rank
             and profile.get('world_size') == world_size
             and profile.get('pid') == report['process']['pid'],
             'profile belongs to another attempt, rank or process')
    _require(profile.get('level') == record['level'], 'profile level differs from the report')
    _require(isinstance(profile.get('blas'), dict) and isinstance(profile.get('fftw'), dict),
             'profile lacks its blas or fftw section')


CACHE_COUNTS = ('hits', 'misses', 'bypassed', 'live_handles')


def check_plan_cache(report):
    """Optional v0.5 FFTW plan cache declaration; absent means the cache was off."""
    record = report.get('fftw_plan_cache')
    if record is None:
        return False
    _require(isinstance(record, dict) and set(record) == {'enabled', *CACHE_COUNTS, 'entries'}
             and isinstance(record['enabled'], bool)
             and all(type(record[k]) is int and record[k] >= 0 for k in (*CACHE_COUNTS, 'entries')),
             'malformed fftw_plan_cache declaration')
    if not record['enabled']:
        _require(all(record[k] == 0 for k in (*CACHE_COUNTS, 'entries')),
                 'fftw_plan_cache used while declared off')
    _require(record['entries'] <= record['misses'], 'fftw_plan_cache entries exceed misses')
    return record['enabled']


DBLAS_IMAGES = {'openblas': 'libopenblas', 'accelerate': 'libBLAS'}


def check_dblas(report):
    """Optional v0.5 DGEMV/DGEMM library declaration; absent means OpenBLAS."""
    record = report.get('dblas_library')
    if record is None:
        return 'openblas'
    _require(isinstance(record, dict) and set(record) == {'library', 'image'}
             and record['library'] in DBLAS_IMAGES
             and (record['image'] is None or isinstance(record['image'], str)),
             'malformed dblas_library declaration')
    _require(record['image'] is None or record['image'].startswith(DBLAS_IMAGES[record['library']]),
             'dblas_library image does not match the declared library')
    return record['library']


def _shared(report):
    process = report['process']
    return {'backend': json.dumps(report['backend'], sort_keys=True), 'mode': report['mode'],
            'check': report['check'], 'world_size': report['world_size'],
            'attempt_id': report['attempt_id'], 'hostname': process['hostname'],
            'executable': process['executable'], 'executable_stat': tuple(process['executable_stat'])}


def aggregate(directory, attempt_id, world_size, *, mode=None, check=None, executable=None,
              not_before=None):
    """Return the attempt evidence dictionary. Never raises for evidence problems."""
    result = {'schema': SCHEMA, 'attempt_id': attempt_id, 'world_size': world_size,
              'report_dir': str(directory), 'evidence': 'INVALID', 'problems': [],
              'ranks_reported': [], 'missing_ranks': [], 'claimed_without_report': [],
              'totals': None, 'statuses': None, 'max_check_ratio': None,
              'ranks_with_consumed': [], 'execution': None, 'shadow_check': None,
              'gpu_qualified': False, 'backend': None, 'mode': None, 'check': None,
              'executable': None, 'hostname': None, 'timing': None, 'work_decades': None,
              'profile': None, 'fftw_plan_cache': None, 'dblas_library': None,
              'limitations': ['Counts calls made by the program through zgemm_ only.',
                              'Says nothing about speed, scientific convergence or agreement '
                              'with a reference.']}
    problems = result['problems']
    try:
        _require(type(world_size) is int and 1 <= world_size <= 4096, 'world size must be 1..4096')
        _require(isinstance(attempt_id, str) and ATTEMPT.fullmatch(attempt_id) is not None,
                 'malformed attempt ID')
        directory = Path(directory)
        info = directory.lstat()
        _require(stat.S_ISDIR(info.st_mode), 'report directory is not a real directory')
        _require(not info.st_mode & (stat.S_IWGRP | stat.S_IWOTH),
                 'report directory is group or world writable')
        names = sorted(os.listdir(directory))
        _require(len(names) <= MAX_FILES, 'too many files in the report directory')
    except (OSError, EvidenceError) as exc:
        problems.append(str(exc))
        return result

    claims, reports, telemetry, profiles = {}, {}, set(), {}
    for name in names:
        match = NAME.fullmatch(name)
        if match is None:
            problems.append(f'unexpected file {name!r} (temporary, conflicting or stray)')
            continue
        rank, kind = int(match.group(1)), match.group(2)
        if rank >= world_size:
            problems.append(f'{name}: rank outside world size {world_size}')
            continue
        path = directory / name
        if kind == 'telemetry.tsv':
            if not stat.S_ISREG(path.lstat().st_mode):
                problems.append(f'{name} is not a regular file')
            telemetry.add(rank)
            continue
        try:
            document = _read_json(path, MAX_PROFILE_BYTES if kind == 'profile.json'
                                  else MAX_REPORT_BYTES)
        except (OSError, EvidenceError) as exc:
            problems.append(str(exc))
            continue
        {'claim.json': claims, 'report.json': reports, 'profile.json': profiles}[kind][rank] = document

    for rank, claim in sorted(claims.items()):
        if not (isinstance(claim, dict) and claim.get('schema') == CLAIM_SCHEMA
                and claim.get('rank') == rank and claim.get('attempt_id') == attempt_id
                and claim.get('world_size') == world_size and type(claim.get('pid')) is int):
            problems.append(f'rank {rank}: malformed or foreign claim')

    counters, statuses, ratios, consumed_ranks, shared, cache_enabled, dblas = \
        {}, {}, [], [], {}, {}, {}
    for rank, report in sorted(reports.items()):
        try:
            c, s = check_rank_report(report, rank, attempt_id, world_size)
            claim = claims.get(rank)
            _require(isinstance(claim, dict) and claim.get('pid') == report['process']['pid'],
                     'report has no matching claim from the same process')
            if not_before is not None:
                _require(report['times']['start_unix'] >= not_before, 'report predates the attempt')
            telemetry_record = report.get('telemetry')
            if telemetry_record is not None:
                _require(isinstance(telemetry_record, dict)
                         and telemetry_record.get('file') == f'rank-{rank:05d}.telemetry.tsv'
                         and rank in telemetry, 'declared telemetry file is missing')
            else:
                _require(rank not in telemetry, 'telemetry file without a report declaration')
            check_profile(report, profiles.get(rank), rank, attempt_id, world_size)
            cache_enabled[rank] = check_plan_cache(report)
            dblas[rank] = check_dblas(report)
        except EvidenceError as exc:
            problems.append(f'rank {rank}: {exc}')
            continue
        counters[rank], statuses[rank] = c, s
        for key, value in _shared(report).items():
            shared.setdefault(key, set()).add(value)
        if report['max_check_ratio_state'] == 'nonfinite':
            ratios.append(math.inf)
        elif report['max_check_ratio'] is not None:
            ratios.append(float(report['max_check_ratio']))
        if c['consumed']:
            consumed_ranks.append(rank)

    for key in SHARED:
        if len(shared.get(key, ())) > 1:
            problems.append(f'ranks disagree on {key} (mixed attempt, binary or settings)')
    first = reports.get(min(counters)) if counters else None
    if first is not None:
        result.update(backend=first['backend'], mode=first['mode'], check=first['check'],
                      executable=first['process']['executable'], hostname=first['process']['hostname'])
        if mode is not None and first['mode'] != mode:
            problems.append(f'reports record mode {first["mode"]}, expected {mode}')
        if check is not None and first['check'] != check:
            problems.append(f'reports record check {first["check"]}, expected {check}')
        if executable is not None and first['process']['executable'] != str(executable):
            problems.append('reports were written by a different executable')

    result['ranks_reported'] = sorted(counters)
    result['missing_ranks'] = [r for r in range(world_size) if r not in reports]
    result['claimed_without_report'] = sorted(set(claims) - set(reports))
    totals = {k: sum(c[k] for c in counters.values()) for k in COUNTERS}
    result['totals'] = totals
    result['statuses'] = {k: sum(s[k] for s in statuses.values()) for k in STATUSES}
    worst = max(ratios) if ratios else None
    result['max_check_ratio'] = worst if worst is None or math.isfinite(worst) else 'nonfinite'
    result['ranks_with_consumed'] = consumed_ranks
    declared = {r: reports[r]['profile']['level'] for r in counters if reports[r].get('profile')}
    if declared:
        if len(set(declared.values())) > 1 or len(declared) != len(counters):
            problems.append('ranks disagree on the profile level')
        result['profile'] = {'level': sorted(set(declared.values()))[0],
                             'ranks': sorted(r for r in declared if r in profiles)}
    cached = [r for r in counters if cache_enabled.get(r)]
    if cached:
        if len(cached) != len(counters):
            problems.append('ranks disagree on the FFTW plan cache')
        result['fftw_plan_cache'] = {
            'enabled': True, 'ranks': cached,
            **{k: sum(reports[r]['fftw_plan_cache'][k] for r in cached)
               for k in ('hits', 'misses', 'bypassed')}}
    libraries = {dblas[r] for r in counters if r in dblas}
    if len(libraries) > 1:
        problems.append('ranks disagree on the DGEMV/DGEMM library')
    elif libraries:
        result['dblas_library'] = libraries.pop()
    timed = [reports[r] for r in counters if reports[r].get('timing') is not None]
    if timed:
        result['timing'] = {'try_s': sum(r['timing']['try_s'] for r in timed),
                            'cpu_s': sum(r['timing']['cpu_s'] for r in timed),
                            'ranks_with_timing': len(timed)}
        result['work_decades'] = [sum(r['work_decades'][i] for r in timed) for i in range(16)]
    if problems:
        result['evidence'] = 'INVALID'
    elif result['missing_ranks']:
        result['evidence'] = 'INCOMPLETE'
    else:
        result['evidence'] = 'COMPLETE'
    result['execution'] = 'GPU_CONSUMED' if totals['consumed'] > 0 else 'CPU_ONLY'
    if result['check'] is False:
        shadow = 'NOT_ENABLED'
    elif result['statuses']['check_failed'] or (worst is not None and not worst <= 1.0):
        shadow = 'FAILED'
    elif totals['checked'] == 0:
        shadow = 'NO_CHECKS'
    else:
        shadow = 'PASSED'
    result['shadow_check'] = shadow
    result['gpu_qualified'] = (result['evidence'] == 'COMPLETE' and totals['consumed'] > 0
                               and shadow in ('PASSED', 'NOT_ENABLED'))
    return result


def summary(result):
    """Compact one-line form used by the runner metadata and job details."""
    totals = result.get('totals') or {}
    keys = ('evidence', 'execution', 'shadow_check', 'gpu_qualified', 'mode', 'check',
            'world_size', 'max_check_ratio')
    data = {key: result.get(key) for key in keys}
    data.update(ranks_reported=len(result.get('ranks_reported') or ()),
                missing_ranks=result.get('missing_ranks'),
                ranks_with_consumed=len(result.get('ranks_with_consumed') or ()),
                calls=totals.get('calls'), consumed=totals.get('consumed'),
                cpu_calls=totals.get('cpu_calls'), gpu_submitted=totals.get('gpu_submitted'),
                range_fallbacks=(result.get('statuses') or {}).get('range'),
                problems=len(result.get('problems') or ()))
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    agg = sub.add_parser('aggregate')
    agg.add_argument('--dir', required=True, type=Path)
    agg.add_argument('--attempt-id', required=True)
    agg.add_argument('--world-size', required=True, type=int)
    agg.add_argument('--mode', choices=MODES)
    agg.add_argument('--check', choices=('0', '1'))
    agg.add_argument('--executable')
    agg.add_argument('--not-before', type=float)
    agg.add_argument('--output', type=Path, help='new file; never overwritten')
    args = parser.parse_args(argv)
    result = aggregate(args.dir, args.attempt_id, args.world_size, mode=args.mode,
                       check=None if args.check is None else args.check == '1',
                       executable=args.executable, not_before=args.not_before)
    text = json.dumps(result, indent=1, sort_keys=True) + '\n'
    if args.output:
        with open(args.output, 'x') as handle:
            handle.write(text)
    else:
        sys.stdout.write(text)
    print(json.dumps(summary(result), sort_keys=True, separators=(',', ':')), file=sys.stderr)
    return {'COMPLETE': 0, 'INCOMPLETE': 3}.get(result['evidence'], 1)


if __name__ == '__main__':
    sys.exit(main())
