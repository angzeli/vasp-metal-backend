#!/usr/bin/env python3
"""Run the shape-replay benchmark once, serially, behind the memory gate.

The benchmark process gets one OpenMP/OpenBLAS/vecLib thread, forced Metal mode
with the shadow check off (correctness is checked outside the timed regions),
its own process group and a hard wall-time limit. Host state is recorded before
and after; any sign of a busy host marks the timing exploratory, and an
exploratory run must not be used to ship an auto-mode rule.
"""
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MIN_FREE_PERCENT = 20
BUSY_LOAD = 2.0          # 1-minute load average above which the host is busy
BUSY_PROCESS_CPU = 20.0  # any other process above this %CPU marks the host busy


def sysctl(name):
    return subprocess.run(['/usr/sbin/sysctl', '-n', name], capture_output=True, text=True,
                          check=True, timeout=10).stdout.strip()


def host_state():
    level = int(sysctl('kern.memorystatus_vm_pressure_level'))  # 1 normal, 2 warn, 4 critical
    free = int(sysctl('kern.memorystatus_level'))
    load = float(sysctl('vm.loadavg').strip('{} ').split()[0])
    ps = subprocess.run(['/bin/ps', '-Ao', 'pcpu=,pid='], capture_output=True, text=True,
                        check=True, timeout=10).stdout.split('\n')
    busy = [line for line in ps if line.strip() and float(line.split()[0]) > BUSY_PROCESS_CPU
            and int(line.split()[1]) != os.getpid()]
    return {'pressure_level': level, 'free_percent': free, 'load1': load, 'busy_processes': len(busy)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--output', required=True, type=Path, help='new directory')
    parser.add_argument('--shapes', type=Path, default=ROOT / 'bench/shapes-v0.2.tsv')
    parser.add_argument('--binary', type=Path, default=ROOT / 'build/shape_replay')
    parser.add_argument('--reps', type=int, default=7)
    parser.add_argument('--shape-seconds', type=float, default=20.0)
    parser.add_argument('--total-seconds', type=float, default=300.0)
    parser.add_argument('--max-bytes', type=float, default=128 * 1024 * 1024)
    parser.add_argument('--allow-warn-pressure', action='store_true',
                        help='explicit override: also run at WARN (level 2) pressure')
    args = parser.parse_args()

    before = host_state()
    allowed = (1, 2) if args.allow_warn_pressure else (1,)
    ok = before['pressure_level'] in allowed and before['free_percent'] >= MIN_FREE_PERCENT
    print(f'memory gate: pressure_level={before["pressure_level"]} '
          f'free={before["free_percent"]}% -> {"PASS" if ok else "FAIL"}')
    if not ok:
        raise SystemExit('insufficient memory headroom; not running')

    args.output.mkdir()  # parent must exist; refuses an existing directory
    env = {k: v for k, v in os.environ.items() if not k.startswith(('DYLD_', 'VMB_'))}
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1',
               VMB_MODE='force', VMB_CHECK='0')
    command = [str(args.binary), str(args.shapes), str(args.output / 'summary.tsv'),
               str(args.output / 'raw.tsv'), '--reps', str(args.reps),
               '--shape-seconds', str(args.shape_seconds),
               '--total-seconds', str(args.total_seconds), '--max-bytes', str(args.max_bytes)]
    limit = args.total_seconds + 60.0
    started = time.monotonic()
    reason = 'completed'
    with (args.output / 'stderr.log').open('w') as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            status = process.wait(timeout=limit)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as stop:
            reason = 'timeout' if isinstance(stop, subprocess.TimeoutExpired) else 'interrupted'
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(process.pid, sig)  # our own new process group only
                    process.wait(timeout=10)
                    break
                except ProcessLookupError:
                    break
                except subprocess.TimeoutExpired:
                    continue
            status = 124
    wall = time.monotonic() - started
    after = host_state()
    busy = any(s['pressure_level'] != 1 or s['load1'] > BUSY_LOAD or s['busy_processes'] > 0
               for s in (before, after))
    lines = [f'exit={status}', f'reason={reason}', f'wall_seconds={wall:.3f}',
             f'timing_class={"exploratory" if busy else "quiet"}',
             f'allow_warn_pressure={int(args.allow_warn_pressure)}']
    lines += [f'before.{k}={v}' for k, v in before.items()]
    lines += [f'after.{k}={v}' for k, v in after.items()]
    lines.append('command=' + ' '.join(command))
    (args.output / 'host-state.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    return 0 if status == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
