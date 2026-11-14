#!/usr/bin/env python3
"""Compare a backend-off and a backend-on VASP run and classify the result.

Reads only run metadata, energies, iteration counts, convergence messages,
forces, stress and the backend's own reports. Staged inputs are compared by
digest; no input content is printed. A run directory is either a v0.1/v0.2
single-rank run (run-meta.txt, vmb-report.txt) or a v0.3 run made by the VASP
port's managed runner (RUN_METADATA.txt, vmb-reports/), whose per-rank reports
are aggregated by scripts/vmb_evidence.py, the one acceptance model. Classes:

  VASP-verified        both runs valid, the on-run consumed GPU results, the
                       shadow check (if enabled) never failed, multi-rank
                       evidence is complete for every rank, and
                       |dTOTEN| / N_atoms <= 1e-5 eV/atom
  CPU-only consistent  both runs valid, the on-run was in auto mode and served
                       every call on the CPU (a valid run, not GPU integration)
  CPU A/B consistent   both runs valid and in off mode, neither used the GPU,
                       within the same energy threshold: an A/B of a CPU-side
                       setting such as VMB_DBLAS_LIBRARY (v0.5)
  FAILED               any check below failed

--fixed-work: single points capped at exactly NELM electronic steps
(NELMIN = NELM), compared at equal work. Such a run is valid without EDIFF
convergence when it made exactly NELM steps, and both runs must make the same
number. Not accepted for ionic relaxations.

--no-backend off|on: that run is declared to carry no backend, e.g. VASP linked
directly against a BLAS (v0.6, #9). It must have no backend report at all, and
counts as an off-mode run, so the pair can be CPU A/B consistent.

Exit status: 0 VASP-verified, 3 CPU-only consistent, 4 CPU A/B consistent, 1 FAILED.
"""
import argparse
import hashlib
import importlib.util
import math
import re
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location('vmb_evidence', Path(__file__).with_name('vmb_evidence.py'))
vmb_evidence = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vmb_evidence)

THRESHOLD = 1e-5  # eV/atom
# Ionic relaxations (v0.4), fixed before any comparison was run.
RELAX_FORCE = 1e-3     # eV/Angstrom, max component difference of final forces
RELAX_STRESS = 0.1     # kB, max component difference of final stress
RELAX_LATTICE = 1e-4   # Angstrom, max component difference of final lattice vectors
IONIC = re.compile(r'^\s*(\d+)\s+F=\s*(\S+)')
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


def read_metadata(path):
    """RUN_METADATA.txt of the port runner: 'key: value' lines, last value wins."""
    out = {}
    if path.exists():
        for line in path.read_text(errors='replace').splitlines():
            key, sep, value = line.partition(': ')
            if sep:
                out[key] = value
    return out


def parse(run):
    run = Path(run)
    r = {'dir': run, 'problems': []}
    if (run / 'RUN_METADATA.txt').exists():
        meta = read_metadata(run / 'RUN_METADATA.txt')
        r['exit'], r['reason'] = meta.get('launcher_status'), meta.get('reason')
        r['metadata'] = meta
    else:
        meta = read_kv(run / 'run-meta.txt')
        r['exit'], r['reason'] = meta.get('exit'), meta.get('reason')
        r['metadata'] = {}
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
    nbands = re.findall(r'NBANDS\s*=\s*(\d+)', outcar)
    r['nbands'] = int(nbands[-1]) if nbands else None
    stress = re.findall(r'^\s*in kB((?:\s+\S+){6})\s*$', outcar, re.M)
    try:
        r['stress'] = [float(x) for x in stress[-1].split()] if stress else None
    except ValueError:
        r['stress'] = [math.nan] * 6
    oszicar = (run / 'OSZICAR').read_text().splitlines() if (run / 'OSZICAR').exists() else []
    r['iterations'] = sum(1 for line in oszicar if STEP.match(line))
    r['ediff_reached'] = 'aborting loop because EDIFF is reached' in outcar
    r['normal_end'] = 'General timing and accounting' in outcar
    r['converged'] = (r['ediff_reached'] and r['nelm'] is not None
                      and 0 < r['iterations'] < r['nelm'])
    # Ionic relaxation: one F= line per ionic step; electronic steps counted per step.
    nsw = re.search(r'NSW\s*=\s*(\d+)', outcar)
    ionic, scf, current = [], [], 0
    for line in oszicar:
        if STEP.match(line):
            current += 1
        found = IONIC.match(line)
        if found:
            try:
                ionic.append(float(found.group(2)))
            except ValueError:
                ionic.append(math.nan)
            scf.append(current)
            current = 0
    r['relaxation'] = bool(nsw and int(nsw.group(1)) > 0)
    r['ionic_energies'], r['scf_per_ionic'] = ionic, scf
    if r['relaxation']:
        r['ionic_converged'] = 'reached required accuracy' in outcar
        r['converged'] = (bool(ionic) and r['nelm'] is not None and all(0 < n < r['nelm'] for n in scf)
                          and outcar.count('aborting loop because EDIFF is reached') == len(ionic))
    lattice = re.findall(r'direct lattice vectors[^\n]*\n((?:[^\n]*\n){3})', outcar)
    try:
        r['lattice'] = [[float(x) for x in row.split()[:3]] for row in lattice[-1].splitlines()] if lattice else None
    except ValueError:
        r['lattice'] = None
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
    r['evidence'] = None
    if (run / 'vmb-reports').is_dir():
        meta = r['metadata']
        attempt = meta.get('backend_attempt_id', '').partition(';')[0]
        ranks = re.search(r'ranks=(\d+)', meta.get('mpi', ''))
        world = int(ranks.group(1)) if ranks else 0
        r['evidence'] = vmb_evidence.aggregate(
            run / 'vmb-reports', attempt, world, mode=meta.get('backend_mode') or None,
            check={'True': True, 'False': False}.get(meta.get('backend_check')))
    return r


def counters_multi(ev):
    """Counters dictionary of an aggregated multi-rank attempt."""
    totals, statuses = ev['totals'] or {}, ev['statuses'] or {}
    ratio = ev['max_check_ratio']
    return {'calls': totals.get('calls', 0), 'submitted': totals.get('gpu_submitted', 0),
            'completed': totals.get('gpu_completed', 0), 'consumed': totals.get('consumed', 0),
            'cpu': totals.get('cpu_calls', 0), 'checked': totals.get('checked', 0),
            'check': ev['check'] is True, 'mode': ev['mode'], 'statuses': statuses,
            'ratio': math.inf if ratio == 'nonfinite' else 0.0 if ratio is None else float(ratio),
            'evidence': ev['evidence'], 'ranks': len(ev['ranks_reported']),
            'ranks_with_consumed': len(ev['ranks_with_consumed']), 'shadow': ev['shadow_check'],
            'gpu_qualified': ev['gpu_qualified']}


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


NO_BACKEND = {'mode': 'none', 'consumed': 0, 'submitted': 0}


def check_run(name, r, fixed_work=False, no_backend=False):
    p = []
    if r['exit'] != '0' or r['reason'] != 'completed':
        p.append(f'{name}: abnormal exit (exit={r["exit"]}, reason={r["reason"]})')
    if not r['normal_end']:
        p.append(f'{name}: no normal-termination timing block')
    if fixed_work and r.get('relaxation'):
        p.append(f'{name}: --fixed-work applies to single points only, not ionic relaxations')
    elif fixed_work:
        if r['nelm'] is None or r['iterations'] != r['nelm']:
            p.append(f'{name}: fixed-work run made {r["iterations"]} electronic steps, not NELM={r["nelm"]}')
    elif not r['converged']:
        if r.get('relaxation'):
            p.append(f'{name}: an ionic step ended without electronic convergence '
                     f'(electronic steps per ionic step {r["scf_per_ionic"]}, NELM={r["nelm"]})')
        else:
            p.append(f'{name}: SCF not converged ({r["iterations"]} iterations, NELM={r["nelm"]})')
    if r.get('relaxation') and not r.get('ionic_converged'):
        p.append(f'{name}: ionic relaxation did not reach the required accuracy')
    if r['toten'] is None or not math.isfinite(r['toten']):
        p.append(f'{name}: missing or non-finite TOTEN')
    if r['natoms'] is None or len(r['forces']) != r['natoms']:
        p.append(f'{name}: incomplete forces ({len(r["forces"])} of {r["natoms"]} atoms)')
    elif not all(math.isfinite(x) for f in r['forces'] for x in f):
        p.append(f'{name}: non-finite force component')
    if r['stress'] is not None and not all(math.isfinite(x) for x in r['stress']):
        p.append(f'{name}: non-finite stress component')
    if no_backend:
        if r['evidence'] is not None and r['evidence']['ranks_reported']:
            p.append(f"{name}: declared without a backend, but {len(r['evidence']['ranks_reported'])} "
                     'ranks reported')
        elif r['report']:
            p.append(f'{name}: declared without a backend, but has a backend report')
        return p, NO_BACKEND
    if r['evidence'] is not None:
        ev = r['evidence']
        c = counters_multi(ev)
        if ev['evidence'] != 'COMPLETE':
            p.append(f"{name}: backend evidence {ev['evidence']}: missing ranks {ev['missing_ranks']}; "
                     + '; '.join(ev['problems'][:5]))
        if ev['shadow_check'] == 'FAILED':
            p.append(f'{name}: shadow check failed')
        return p, c  # per-rank invariants were checked before aggregation
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


def compare(off_dir, on_dir, fixed_work=False, no_backend=None):
    off, on = parse(off_dir), parse(on_dir)
    problems = []
    p_off, c_off = check_run('off', off, fixed_work, no_backend == 'off')
    p_on, c_on = check_run('on', on, fixed_work, no_backend == 'on')
    problems += p_off + p_on
    if fixed_work and off['iterations'] != on['iterations']:
        problems.append(f"electronic step counts differ ({off['iterations']} vs {on['iterations']})")
    mismatched = [n for n in INPUTS if off['digests'][n] is None or off['digests'][n] != on['digests'][n]]
    if mismatched:
        problems.append('inputs differ or missing: ' + ', '.join(mismatched))
    if c_off and (c_off['mode'] not in ('off', 'none') or c_off['consumed'] or c_off['submitted']):
        problems.append('off-run used the GPU or was not in off mode')
    delta = fdiff = math.nan
    if off['toten'] is not None and on['toten'] is not None and off['natoms']:
        delta = abs(on['toten'] - off['toten']) / off['natoms']
        if not delta <= THRESHOLD:
            problems.append(f'|dTOTEN|/N_atoms = {delta:.3e} exceeds {THRESHOLD:.0e}')
    if off['forces'] and len(off['forces']) == len(on['forces']):
        fdiff = max(abs(a - b) for fa, fb in zip(off['forces'], on['forces']) for a, b in zip(fa, fb))
    sdiff = math.nan
    if off['stress'] and on['stress']:
        sdiff = max(abs(a - b) for a, b in zip(off['stress'], on['stress']))
    if off.get('nbands') != on.get('nbands'):
        problems.append(f"effective NBANDS differ ({off.get('nbands')} vs {on.get('nbands')})")
    ldiff = math.nan
    relaxation = off.get('relaxation') or on.get('relaxation')
    if relaxation:
        if len(off['ionic_energies']) != len(on['ionic_energies']):
            problems.append(f"ionic step counts differ ({len(off['ionic_energies'])} vs {len(on['ionic_energies'])})")
        if not fdiff <= RELAX_FORCE:
            problems.append(f'max |dF| = {fdiff:.3e} eV/A exceeds {RELAX_FORCE:.0e}')
        if not sdiff <= RELAX_STRESS:
            problems.append(f'max |dstress| = {sdiff:.3e} kB exceeds {RELAX_STRESS}')
        if off.get('lattice') and on.get('lattice'):
            ldiff = max(abs(a - b) for ra, rb in zip(off['lattice'], on['lattice']) for a, b in zip(ra, rb))
        if not ldiff <= RELAX_LATTICE:
            problems.append(f'max |dlattice| = {ldiff:.3e} A exceeds {RELAX_LATTICE:.0e} (or lattice missing)')
    active = bool(c_on and c_on['consumed'] > 0 and c_on.get('gpu_qualified', True))
    if problems:
        verdict = 'FAILED'
    elif active:
        verdict = 'VASP-verified'
    elif c_on and c_on['mode'] == 'auto':
        verdict = 'CPU-only consistent'
    elif c_on and c_on['mode'] in ('off', 'none') and not c_on['consumed'] and not c_on['submitted']:
        verdict = 'CPU A/B consistent'
    else:
        verdict = 'FAILED'
        problems.append('on-run consumed no GPU result and was not an auto-mode run')
    return {'off': off, 'on': on, 'c_off': c_off, 'c_on': c_on, 'delta': delta,
            'fdiff': fdiff, 'sdiff': sdiff, 'ldiff': ldiff, 'relaxation': bool(relaxation),
            'fixed_work': fixed_work, 'verdict': verdict, 'problems': problems}


def main(off_dir, on_dir, fixed_work=False, no_backend=None):
    res = compare(off_dir, on_dir, fixed_work, no_backend)
    for name in ('off', 'on'):
        r, c = res[name], res['c_' + name]
        print(f'{name}: exit={r["exit"]} TOTEN={r["toten"]} eV iterations={r["iterations"]} '
              f'converged={r["converged"]} normal_end={r["normal_end"]} forces={len(r["forces"])} '
              f'NBANDS={r["nbands"]}')
        if c is NO_BACKEND:
            print(f'{name}: no backend (declared)')
        elif c:
            s = c['statuses']
            print(f'{name}: mode={c["mode"]} calls={c["calls"]} submitted={c["submitted"]} '
                  f'completed={c["completed"]} consumed={c["consumed"]} cpu_calls={c["cpu"]} '
                  f'checked={c["checked"]} check_failed={s.get("check_failed", 0)} '
                  f'range={s.get("range", 0)} auto_cpu={s.get("auto_cpu", 0)} '
                  f'max_check_ratio={c["ratio"]:.3e}')
            if 'evidence' in c:
                print(f'{name}: evidence={c["evidence"]} ranks={c["ranks"]} '
                      f'ranks_with_consumed={c["ranks_with_consumed"]} shadow={c["shadow"]} '
                      f'gpu_qualified={c["gpu_qualified"]}')
    print(f'inputs matched: {not any("inputs differ" in p for p in res["problems"])}')
    print(f'|dTOTEN|/N_atoms = {res["delta"]:.3e} eV/atom (threshold {THRESHOLD:.0e})')
    print(f'max |dF| component = {res["fdiff"]:.3e} eV/A')
    print(f'max |dstress| component = {res["sdiff"]:.3e} kB')
    if res['relaxation']:
        for name in ('off', 'on'):
            r = res[name]
            print(f"{name}: ionic steps={len(r['ionic_energies'])} electronic per step={r['scf_per_ionic']} "
                  f"ionic converged={r.get('ionic_converged')}")
        print(f'max |dlattice| component = {res["ldiff"]:.3e} A')
    for p in res['problems']:
        print('PROBLEM:', p)
    if res['fixed_work']:
        print(f"fixed work: {res['off']['iterations']} electronic steps (NELM={res['off']['nelm']})")
    print('VERDICT:', res['verdict'])
    return {'VASP-verified': 0, 'CPU-only consistent': 3, 'CPU A/B consistent': 4}.get(res['verdict'], 1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--fixed-work', action='store_true',
                    help='single points capped at exactly NELM steps, compared at equal work')
    ap.add_argument('--no-backend', choices=('off', 'on'),
                    help='that run carries no backend (VASP linked directly against a BLAS)')
    ap.add_argument('off_dir')
    ap.add_argument('on_dir')
    a = ap.parse_args()
    sys.exit(main(a.off_dir, a.on_dir, a.fixed_work, a.no_backend))
