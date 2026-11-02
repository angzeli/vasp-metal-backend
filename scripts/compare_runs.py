#!/usr/bin/env python3
"""Compare a backend-off and a backend-on VASP run and classify the result.

Reads only run metadata, energies, iteration counts, convergence messages,
forces and the backend's own report. Staged inputs are compared by digest; no
input content is printed. Classes:

  VASP-verified        both runs valid, the on-run consumed GPU results, the
                       shadow check (if enabled) never failed, and
                       |dTOTEN| / N_atoms <= 1e-5 eV/atom
  CPU-only consistent  both runs valid, the on-run was in auto mode and served
                       every call on the CPU (a valid run, not GPU integration)
  FAILED               any check below failed

Exit status: 0 VASP-verified, 3 CPU-only consistent, 1 FAILED.
"""
import hashlib
import math
import re
import sys
from pathlib import Path

THRESHOLD = 1e-5  # eV/atom
INPUTS = ('INCAR', 'POSCAR', 'KPOINTS', 'POTCAR')
STEP = re.compile(r'^\s*(DAV|RMM|CG|SDA):')


def read_kv(path):
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            key, sep, value = line.partition('=')
            if sep:
                out[key.strip()] = value.strip()
    return out


def parse(run):
    run = Path(run)
    r = {'dir': run, 'problems': []}
    meta = read_kv(run / 'run-meta.txt')
    r['exit'] = meta.get('exit')
    r['reason'] = meta.get('reason')
    outcar = (run / 'OUTCAR').read_text(errors='replace') if (run / 'OUTCAR').exists() else ''
    lines = (run / 'POSCAR').read_text().splitlines() if (run / 'POSCAR').exists() else []
    try:
        r['natoms'] = sum(int(x) for x in lines[6].split())
    except (IndexError, ValueError):
        r['natoms'] = None
    toten = re.findall(r'free  energy   TOTEN\s*=\s*(\S+)\s*eV', outcar)
    r['toten'] = float(toten[-1]) if toten else None
    nelm = re.search(r'NELM\s*=\s*(\d+)', outcar)
    r['nelm'] = int(nelm.group(1)) if nelm else None
    oszicar = (run / 'OSZICAR').read_text().splitlines() if (run / 'OSZICAR').exists() else []
    r['iterations'] = sum(1 for line in oszicar if STEP.match(line))
    r['ediff_reached'] = 'aborting loop because EDIFF is reached' in outcar
    r['normal_end'] = 'General timing and accounting' in outcar
    r['converged'] = (r['ediff_reached'] and r['nelm'] is not None
                      and 0 < r['iterations'] < r['nelm'])
    r['forces'] = []
    block = outcar.rsplit('TOTAL-FORCE (eV/Angst)', 1)
    if len(block) == 2 and r['natoms']:
        for line in block[1].splitlines()[2:2 + r['natoms']]:
            fields = line.split()
            if len(fields) != 6:  # e.g. the closing rule of a truncated block
                break
            try:
                r['forces'].append([float(x) for x in fields[3:6]])
            except ValueError:
                r['forces'].append([math.nan] * 3)
    r['digests'] = {name: hashlib.sha256((run / name).read_bytes()).hexdigest()
                    if (run / name).exists() else None for name in INPUTS}
    r['report'] = read_kv(run / 'vmb-report.txt')
    return r


def counters(rep):
    """Integer counters from a v0.1 or v0.2 report; missing statuses are 0."""
    get = lambda key: int(rep.get(key, '0'))
    statuses = {k[len('vmb.status.'):]: int(v) for k, v in rep.items()
                if k.startswith('vmb.status.')}
    mode = rep.get('vmb.mode') or ('force' if rep.get('vmb.enabled') == '1' else 'off')
    return {'calls': get('vmb.calls'), 'submitted': get('vmb.gpu_submitted'),
            'completed': get('vmb.gpu_completed'), 'consumed': get('vmb.consumed'),
            'cpu': get('vmb.cpu_calls'), 'checked': get('vmb.checked'),
            'check': rep.get('vmb.check') == '1', 'mode': mode, 'statuses': statuses,
            'ratio': float(rep.get('vmb.max_check_ratio', 'nan'))}


def check_run(name, r):
    p = []
    if r['exit'] != '0' or r['reason'] != 'completed':
        p.append(f'{name}: abnormal exit (exit={r["exit"]}, reason={r["reason"]})')
    if not r['normal_end']:
        p.append(f'{name}: no normal-termination timing block')
    if not r['converged']:
        p.append(f'{name}: SCF not converged ({r["iterations"]} iterations, NELM={r["nelm"]})')
    if r['toten'] is None or not math.isfinite(r['toten']):
        p.append(f'{name}: missing or non-finite TOTEN')
    if r['natoms'] is None or len(r['forces']) != r['natoms']:
        p.append(f'{name}: incomplete forces ({len(r["forces"])} of {r["natoms"]} atoms)')
    elif not all(math.isfinite(x) for f in r['forces'] for x in f):
        p.append(f'{name}: non-finite force component')
    if not r['report']:
        p.append(f'{name}: backend report missing')
        return p, None
    c = counters(r['report'])
    s = c['statuses']
    if c['calls'] != c['consumed'] + c['cpu']:
        p.append(f'{name}: calls != consumed + cpu_calls')
    if not c['submitted'] >= c['completed'] >= c['consumed'] >= 0:
        p.append(f'{name}: submitted >= completed >= consumed violated')
    if sum(s.values()) != c['calls']:
        p.append(f'{name}: per-outcome counts do not sum to calls')
    if s.get('ok', 0) != c['consumed']:
        p.append(f'{name}: status.ok != consumed')
    if c['submitted'] - c['completed'] != s.get('gpu_error', 0):
        p.append(f'{name}: submitted - completed != gpu_error')
    if c['completed'] - c['consumed'] != s.get('gpu_nonfinite', 0) + s.get('check_failed', 0):
        p.append(f'{name}: completed - consumed != gpu_nonfinite + check_failed')
    if c['check'] and c['checked'] != c['consumed'] + s.get('check_failed', 0):
        p.append(f'{name}: checked != consumed + check_failed')
    if s.get('check_failed', 0):
        p.append(f'{name}: {s["check_failed"]} shadow-check failures')
    if c['check'] and c['consumed'] and not c['ratio'] <= 1.0:
        p.append(f'{name}: max shadow-check ratio {c["ratio"]} > 1')
    return p, c


def compare(off_dir, on_dir):
    off, on = parse(off_dir), parse(on_dir)
    problems = []
    p_off, c_off = check_run('off', off)
    p_on, c_on = check_run('on', on)
    problems += p_off + p_on
    mismatched = [n for n in INPUTS if off['digests'][n] is None or off['digests'][n] != on['digests'][n]]
    if mismatched:
        problems.append('inputs differ or missing: ' + ', '.join(mismatched))
    if c_off and (c_off['mode'] != 'off' or c_off['consumed'] or c_off['submitted']):
        problems.append('off-run used the GPU or was not in off mode')
    delta = fdiff = math.nan
    if off['toten'] is not None and on['toten'] is not None and off['natoms']:
        delta = abs(on['toten'] - off['toten']) / off['natoms']
        if not delta <= THRESHOLD:
            problems.append(f'|dTOTEN|/N_atoms = {delta:.3e} exceeds {THRESHOLD:.0e}')
    if off['forces'] and len(off['forces']) == len(on['forces']):
        fdiff = max(abs(a - b) for fa, fb in zip(off['forces'], on['forces']) for a, b in zip(fa, fb))
    active = bool(c_on and c_on['consumed'] > 0)
    if problems:
        verdict = 'FAILED'
    elif active:
        verdict = 'VASP-verified'
    elif c_on and c_on['mode'] == 'auto':
        verdict = 'CPU-only consistent'
    else:
        verdict = 'FAILED'
        problems.append('on-run consumed no GPU result and was not an auto-mode run')
    return {'off': off, 'on': on, 'c_off': c_off, 'c_on': c_on, 'delta': delta,
            'fdiff': fdiff, 'verdict': verdict, 'problems': problems}


def main(off_dir, on_dir):
    res = compare(off_dir, on_dir)
    for name in ('off', 'on'):
        r, c = res[name], res['c_' + name]
        print(f'{name}: exit={r["exit"]} TOTEN={r["toten"]} eV iterations={r["iterations"]} '
              f'converged={r["converged"]} normal_end={r["normal_end"]} forces={len(r["forces"])}')
        if c:
            s = c['statuses']
            print(f'{name}: mode={c["mode"]} calls={c["calls"]} submitted={c["submitted"]} '
                  f'completed={c["completed"]} consumed={c["consumed"]} cpu_calls={c["cpu"]} '
                  f'checked={c["checked"]} check_failed={s.get("check_failed", 0)} '
                  f'range={s.get("range", 0)} auto_cpu={s.get("auto_cpu", 0)} '
                  f'max_check_ratio={c["ratio"]:.3e}')
    print(f'inputs matched: {not any("inputs differ" in p for p in res["problems"])}')
    print(f'|dTOTEN|/N_atoms = {res["delta"]:.3e} eV/atom (threshold {THRESHOLD:.0e})')
    print(f'max |dF| component = {res["fdiff"]:.3e} eV/A')
    for p in res['problems']:
        print('PROBLEM:', p)
    print('VERDICT:', res['verdict'])
    return {'VASP-verified': 0, 'CPU-only consistent': 3}.get(res['verdict'], 1)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit('usage: compare_runs.py OFF_DIR ON_DIR')
    sys.exit(main(sys.argv[1], sys.argv[2]))
