#!/usr/bin/env python3
"""Stage a private Si2 case and make one bounded single-rank VASP launch.

Safety contract:
- at most two launches in total, recorded in a ledger before each launch;
- memory must be at normal pressure (or WARN with an explicit
  --allow-warn-pressure override, never critical) with enough free memory;
- 1 MPI rank, one library thread, NCORE = KPAR = 1, synthetic-topology OpenMPI;
- the launch gets its own session and process group; on timeout or interrupt
  only that group is signalled. There are no retries.
Inputs are copied byte for byte; the potential file is never parsed or printed.
"""
import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

MPIRUN = '/opt/homebrew/Cellar/open-mpi/5.0.9/bin/mpirun'
PATH = ('/opt/homebrew/Cellar/open-mpi/5.0.9/bin:/opt/homebrew/Cellar/gcc/16.1.0/bin:'
        '/usr/bin:/bin:/usr/sbin:/sbin')
INPUTS = ('INCAR', 'POSCAR', 'KPOINTS', 'POTCAR')
MAX_LAUNCHES = 2
MIN_FREE_PERCENT = 20
TIMEOUT = 60.0


def sysctl(name):
    return subprocess.run(['/usr/sbin/sysctl', '-n', name], capture_output=True, text=True,
                          check=True, timeout=10).stdout.strip()


def memory_gate(allow_warn):
    level = int(sysctl('kern.memorystatus_vm_pressure_level'))  # 1 normal, 2 warn, 4 critical
    free = int(sysctl('kern.memorystatus_level'))                # percent available
    allowed = (1, 2) if allow_warn else (1,)
    return level, free, level in allowed and free >= MIN_FREE_PERCENT


def stage(fixture, out):
    out.mkdir()  # parent must exist; refuses an existing directory
    for name in INPUTS:
        shutil.copyfile(fixture / name, out / name)
    incar = (out / 'INCAR').read_text()
    tags = {line.split('=')[0].strip().upper() for line in incar.splitlines() if '=' in line}
    if tags & {'NCORE', 'KPAR', 'NPAR'}:
        raise SystemExit('fixture INCAR already sets a parallel tag; refusing to edit it')
    (out / 'INCAR').write_text(incar.rstrip('\n') + '\nNCORE = 1\nKPAR = 1\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--binary', required=True, type=Path)
    parser.add_argument('--fixture', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='new directory')
    parser.add_argument('--ledger', required=True, type=Path)
    parser.add_argument('--backend', choices=('off', 'auto', 'force', 'on'), required=True,
                        help='VMB_MODE for the launch; on is the v0.1 name for force')
    parser.add_argument('--check', action='store_true', help='VMB_CHECK=1 shadow check')
    parser.add_argument('--telemetry', action='store_true',
                        help='write aggregated shape telemetry (VMB_TELEMETRY)')
    parser.add_argument('--allow-warn-pressure', action='store_true',
                        help='explicit override: also launch at WARN (level 2) pressure')
    args = parser.parse_args()

    launches = args.ledger.read_text().splitlines() if args.ledger.exists() else []
    if len(launches) >= MAX_LAUNCHES:
        raise SystemExit(f'launch budget exhausted ({len(launches)} recorded); not launching')
    level, free, ok = memory_gate(args.allow_warn_pressure)
    print(f'memory gate: pressure_level={level} free={free}% -> {"PASS" if ok else "FAIL"}')
    if not ok:
        raise SystemExit('insufficient memory headroom; not launching')

    stage(args.fixture, args.output)
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('DYLD_', 'OMPI_MCA_', 'PRTE_MCA_', 'PMIX_MCA_', 'VMB_'))}
    cores = sysctl('hw.physicalcpu')
    env.update(PATH=PATH, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               VECLIB_MAXIMUM_THREADS='1', HWLOC_SYNTHETIC=f'Package:1 Core:{cores} PU:1',
               VMB_MODE='force' if args.backend == 'on' else args.backend,
               VMB_CHECK='1' if args.check else '0',
               VMB_REPORT=str(args.output.resolve() / 'vmb-report.txt'))
    forwarded = ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
                 'VMB_MODE', 'VMB_CHECK', 'VMB_REPORT']
    if args.telemetry:
        env['VMB_TELEMETRY'] = str(args.output.resolve() / 'vmb-telemetry.tsv')
        forwarded.append('VMB_TELEMETRY')
    command = [MPIRUN, '-np', '1', '--bind-to', 'none', '--map-by', 'slot']
    for name in forwarded:
        command += ['-x', name]
    command.append(str(args.binary.resolve()))

    with args.ledger.open('a') as ledger:
        ledger.write(f'{time.strftime("%Y-%m-%dT%H:%M:%S")} backend={args.backend} '
                     f'output={args.output}\n')
    started = time.monotonic()
    status, reason = None, 'completed'
    with (args.output / 'stdout.log').open('w') as log:
        process = subprocess.Popen(command, cwd=args.output, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            status = process.wait(timeout=TIMEOUT)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as stop:
            reason = 'timeout' if isinstance(stop, subprocess.TimeoutExpired) else 'interrupted'
            for sig, grace in ((signal.SIGTERM, 10), (signal.SIGKILL, 10)):
                try:
                    os.killpg(process.pid, sig)  # our own new process group only
                except ProcessLookupError:
                    break
                try:
                    process.wait(timeout=grace)
                    break
                except subprocess.TimeoutExpired:
                    continue
            status = 124
    wall = time.monotonic() - started
    meta = (f'backend={args.backend}\ncheck={int(args.check)}\nexit={status}\nreason={reason}\n'
            f'wall_seconds={wall:.3f}\nmemory_pressure_level={level}\nmemory_free_percent={free}\n'
            f'allow_warn_pressure={int(args.allow_warn_pressure)}\n'
            f'command={" ".join(command)}\n')
    (args.output / 'run-meta.txt').write_text(meta)
    print(meta, end='')
    return 0 if status == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
