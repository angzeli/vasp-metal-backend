#!/usr/bin/env python3
"""Compare backend-off and backend-on Si2 runs and apply the v0.1 verdict rule.

Reads only energies, iteration counts, convergence messages and forces from the
private outputs, plus the backend's own report. Prints a short summary.
VASP-verified requires: both runs converged, the on-run consumed at least one
GPU result, and |delta TOTEN| / N_atoms <= 1e-5 eV/atom.
"""
import re
import sys
from pathlib import Path

THRESHOLD = 1e-5  # eV/atom


def parse(run):
    outcar = (run / 'OUTCAR').read_text(errors='replace')
    toten = [float(x) for x in re.findall(r'free  energy   TOTEN\s*=\s*(\S+)\s*eV', outcar)]
    lines = (run / 'POSCAR').read_text().splitlines()
    natoms = sum(int(x) for x in lines[6].split())
    nelm = int(re.search(r'NELM\s*=\s*(\d+)', outcar).group(1))
    oszicar = (run / 'OSZICAR').read_text().splitlines()
    steps = [line for line in oszicar if re.match(r'^\s*(DAV|RMM|CG|SDA):', line)]
    forces = []
    block = outcar.rsplit('TOTAL-FORCE (eV/Angst)', 1)
    if len(block) == 2:
        for line in block[1].splitlines()[2:2 + natoms]:
            forces.append([float(x) for x in line.split()[3:6]])
    report = {}
    path = run / 'vmb-report.txt'
    if path.exists():
        for line in path.read_text().splitlines():
            key, _, value = line.partition('=')
            report[key] = value
    return {
        'toten': toten[-1] if toten else None,
        'natoms': natoms,
        'iterations': len(steps),
        'nelm': nelm,
        'converged': 'aborting loop because EDIFF is reached' in outcar and len(steps) < nelm,
        'normal': 'General timing and accounting' in outcar,
        'forces': forces,
        'report': report,
    }


def main(off_dir, on_dir):
    off, on = parse(Path(off_dir)), parse(Path(on_dir))
    for name, r in (('off', off), ('on', on)):
        rep = r['report']
        print(f'{name}: TOTEN={r["toten"]:.8f} eV  iterations={r["iterations"]}  '
              f'converged={r["converged"]}  normal_end={r["normal"]}')
        print(f'{name}: vmb calls={rep.get("vmb.calls")} consumed={rep.get("vmb.consumed")} '
              f'gpu_completed={rep.get("vmb.gpu_completed")} cpu_calls={rep.get("vmb.cpu_calls")} '
              f'checked={rep.get("vmb.checked")} max_check_ratio={rep.get("vmb.max_check_ratio")} '
              f'device={rep.get("vmb.device")}')
    delta = abs(on['toten'] - off['toten']) / off['natoms']
    fdiff = max((abs(a - b) for fa, fb in zip(off['forces'], on['forces'])
                 for a, b in zip(fa, fb)), default=float('nan'))
    consumed = int(on['report'].get('vmb.consumed', '0'))
    print(f'|delta TOTEN|/N_atoms = {delta:.3e} eV/atom (threshold {THRESHOLD:.0e})')
    print(f'max |delta F| component = {fdiff:.3e} eV/A over {len(off["forces"])} atoms')
    verified = off['converged'] and on['converged'] and consumed > 0 and delta <= THRESHOLD
    print('VERDICT:', 'VASP-verified' if verified else 'NOT verified')
    return 0 if verified else 1


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:3]))
