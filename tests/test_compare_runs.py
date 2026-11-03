#!/usr/bin/env python3
"""Tests of scripts/compare_runs.py on synthetic run directories.

Every fixture file is invented placeholder text that only mimics the handful of
output lines the parser reads; no scientific input or output is used.
"""
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('compare_runs', ROOT / 'scripts/compare_runs.py')
compare_runs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare_runs)

GOOD_REPORT = {
    'vmb.mode': 'force', 'vmb.check': '1', 'vmb.calls': '10', 'vmb.gpu_submitted': '10',
    'vmb.gpu_completed': '10', 'vmb.consumed': '10', 'vmb.cpu_calls': '0', 'vmb.checked': '10',
    'vmb.status.ok': '10', 'vmb.max_check_ratio': '4.0e-02',
}
OFF_REPORT = {
    'vmb.mode': 'off', 'vmb.check': '0', 'vmb.calls': '10', 'vmb.gpu_submitted': '0',
    'vmb.gpu_completed': '0', 'vmb.consumed': '0', 'vmb.cpu_calls': '10', 'vmb.checked': '0',
    'vmb.status.disabled': '10', 'vmb.max_check_ratio': '0',
}


def write_run(path, *, toten=-1.25, iterations=5, nelm=40, ediff=True, normal=True,
              forces=((0.0, 0.0, 0.01), (0.0, 0.0, -0.01)), natoms=2, exit_code='0',
              report=None, poscar_extra=''):
    path.mkdir()
    (path / 'run-meta.txt').write_text(f'exit={exit_code}\nreason=completed\n')
    (path / 'INCAR').write_text('PLACEHOLDER = 1\n')
    (path / 'KPOINTS').write_text('placeholder kpoints\n')
    (path / 'POTCAR').write_text('synthetic placeholder, not a potential\n')
    (path / 'POSCAR').write_text('synthetic\n1.0\n1 0 0\n0 1 0\n0 0 1\nX\n'
                                 f'{natoms}\nDirect\n' + poscar_extra)
    (path / 'OSZICAR').write_text(''.join(f'DAV: {i + 1} {toten} 0 0 0 0\n'
                                          for i in range(iterations)) + f'1 F= {toten}\n')
    lines = [f'   NELM   =     {nelm}']
    if ediff:
        lines.append('---- aborting loop because EDIFF is reached ----')
    lines.append(f'  free  energy   TOTEN  =       {toten:.8f} eV')
    lines.append(' POSITION                                       TOTAL-FORCE (eV/Angst)')
    lines.append(' ' + '-' * 80)
    for f in forces:
        lines.append('      0.0 0.0 0.0     ' + '  '.join(str(x) for x in f))
    lines.append(' ' + '-' * 80)
    if normal:
        lines.append(' General timing and accounting informations for this job:')
    (path / 'OUTCAR').write_text('\n'.join(lines) + '\n')
    rep = dict(report if report is not None else GOOD_REPORT)
    (path / 'vmb-report.txt').write_text(''.join(f'{k}={v}\n' for k, v in rep.items()))


class CompareRunsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_pair(self, off=None, on=None):
        write_run(self.root / 'off', **{'report': OFF_REPORT, **(off or {})})
        write_run(self.root / 'on', **(on or {}))
        return compare_runs.compare(self.root / 'off', self.root / 'on')

    def assertFails(self, res, fragment):
        self.assertEqual(res['verdict'], 'FAILED')
        self.assertTrue(any(fragment in p for p in res['problems']), res['problems'])

    def test_active_gpu_verified(self):
        res = self.run_pair()
        self.assertEqual(res['verdict'], 'VASP-verified', res['problems'])
        self.assertEqual(res['delta'], 0.0)

    def test_auto_all_cpu_is_valid_but_not_integration(self):
        auto = {**OFF_REPORT, 'vmb.mode': 'auto', 'vmb.status.disabled': '0',
                'vmb.status.auto_cpu': '10'}
        res = self.run_pair(on={'report': auto})
        self.assertEqual(res['verdict'], 'CPU-only consistent', res['problems'])

    def test_force_mode_without_gpu_results_fails(self):
        res = self.run_pair(on={'report': {**OFF_REPORT, 'vmb.mode': 'force'}})
        self.assertFails(res, 'consumed no GPU result')

    def test_abnormal_exit(self):
        self.assertFails(self.run_pair(on={'exit_code': '124'}), 'abnormal exit')

    def test_missing_normal_termination(self):
        self.assertFails(self.run_pair(on={'normal': False}), 'normal-termination')

    def test_not_converged(self):
        self.assertFails(self.run_pair(on={'iterations': 40, 'nelm': 40}), 'not converged')
        self.assertFails(self.run_pair_fresh(on={'ediff': False}), 'not converged')

    def run_pair_fresh(self, **kw):
        self.tearDown()
        self.setUp()
        return self.run_pair(**kw)

    def test_nonfinite_force(self):
        self.assertFails(self.run_pair(on={'forces': ((0, 0, math.nan), (0, 0, 0))}), 'non-finite force')

    def test_incomplete_forces(self):
        self.assertFails(self.run_pair(on={'forces': ((0, 0, 0),)}), 'incomplete forces')

    def test_mismatched_inputs(self):
        self.assertFails(self.run_pair(on={'poscar_extra': '0 0 0\n'}), 'inputs differ')

    def test_inconsistent_counters(self):
        bad = {**GOOD_REPORT, 'vmb.cpu_calls': '1'}
        self.assertFails(self.run_pair(on={'report': bad}), 'calls != consumed + cpu_calls')
        bad = {**GOOD_REPORT, 'vmb.gpu_completed': '9'}
        self.assertFails(self.run_pair_fresh(on={'report': bad}), 'submitted - completed')

    def test_shadow_check_failures(self):
        bad = {**GOOD_REPORT, 'vmb.consumed': '9', 'vmb.cpu_calls': '1', 'vmb.status.ok': '9',
               'vmb.status.check_failed': '1'}
        self.assertFails(self.run_pair(on={'report': bad}), 'shadow-check failures')
        bad = {**GOOD_REPORT, 'vmb.max_check_ratio': 'nan'}
        self.assertFails(self.run_pair_fresh(on={'report': bad}), 'shadow-check ratio')

    def test_energy_difference(self):
        self.assertFails(self.run_pair(on={'toten': -1.25 + 3e-5}), 'exceeds')

    def test_off_run_must_not_use_gpu(self):
        self.assertFails(self.run_pair(off={'report': GOOD_REPORT}), 'off-run used the GPU')

    def test_missing_report(self):
        res = self.run_pair()
        (self.root / 'on' / 'vmb-report.txt').unlink()
        res = compare_runs.compare(self.root / 'off', self.root / 'on')
        self.assertFails(res, 'report missing')

    def test_v01_report_format(self):
        v01 = {k: v for k, v in GOOD_REPORT.items() if k != 'vmb.mode'}
        v01['vmb.enabled'] = '1'
        off01 = {k: v for k, v in OFF_REPORT.items() if k != 'vmb.mode'}
        off01['vmb.enabled'] = '0'
        res = self.run_pair(off={'report': off01}, on={'report': v01})
        self.assertEqual(res['verdict'], 'VASP-verified', res['problems'])


STATUSES = ('ok', 'disabled', 'trivial', 'badarg', 'nonfinite', 'too_small', 'too_large',
            'no_device', 'gpu_error', 'gpu_nonfinite', 'check_failed', 'range', 'auto_cpu')


def write_multi(path, mode, world=4, consumed_ranks=(0, 2), skip=(), attempt='att1', meta_attempt=None):
    """Add a v0.3 managed-runner record and synthetic rank reports to a run."""
    (path / 'run-meta.txt').unlink()
    (path / 'vmb-report.txt').unlink()
    check = mode == 'force'
    (path / 'RUN_METADATA.txt').write_text(
        f'state: STAGING\nmpi: /x/mpirun; mpirun (Open MPI) 5.0.11; mode=native (from command line); '
        f'ranks={world}\nbackend_mode: {mode}\nbackend_check: {check}\n'
        f'backend_attempt_id: {meta_attempt or attempt}; origin=cmw\nreason: completed\nlauncher_status: 0\n'
        'state: FINISHED\n')
    reports = path / 'vmb-reports'
    reports.mkdir(mode=0o700)
    for rank in range(world):
        pid = 100 + rank
        (reports / f'rank-{rank:05d}.claim.json').write_text(json.dumps(
            {'schema': 'vmb.rank-claim.v1', 'attempt_id': attempt, 'rank': rank,
             'world_size': world, 'pid': pid, 'start_unix': 10.0}))
        if rank in skip:
            continue
        used = 1 if mode == 'force' and rank in consumed_ranks else 0
        statuses = dict.fromkeys(STATUSES, 0)
        statuses['ok'] = used
        statuses['disabled' if mode == 'off' else 'range'] = 2 - used
        report = {'schema': 'vmb.rank-report.v1', 'complete': True,
                  'backend': {'name': 'vasp-metal-backend', 'version': '0.3.0', 'source_commit': 'c',
                              'source_digest': 'd', 'source_dirty': False},
                  'attempt_id': attempt, 'rank': rank, 'world_size': world,
                  'process': {'pid': pid, 'hostname': 'h', 'executable': '/x/vasp_std_vmb',
                              'executable_stat': [1, 2, 3, 4]},
                  'times': {'start_unix': 10.0, 'end_unix': 11.0}, 'mode': mode, 'check': check,
                  'counters': {'calls': 2, 'gpu_submitted': used, 'gpu_completed': used,
                               'consumed': used, 'cpu_calls': 2 - used,
                               'checked': used if check else 0},
                  'statuses': statuses,
                  'max_check_ratio': 0.01 if used and check else None,
                  'max_check_ratio_state': 'finite' if used and check else 'none', 'telemetry': None}
        (reports / f'rank-{rank:05d}.report.json').write_text(json.dumps(report))


class MultiRankCompareTest(CompareRunsTest):
    def run_multi(self, on_kw=None, outcar_extra=None):
        write_run(self.root / 'off')
        write_run(self.root / 'on')
        write_multi(self.root / 'off', 'off')
        write_multi(self.root / 'on', 'force', **(on_kw or {}))
        if outcar_extra:
            outcar = self.root / 'on' / 'OUTCAR'
            outcar.write_text(outcar.read_text() + outcar_extra)
        return compare_runs.compare(self.root / 'off', self.root / 'on')

    def test_complete_multi_rank_verified(self):
        res = self.run_multi()
        self.assertEqual(res['verdict'], 'VASP-verified', res['problems'])
        self.assertEqual((res['c_on']['consumed'], res['c_on']['ranks']), (2, 4))
        self.assertEqual(res['c_off']['mode'], 'off')

    def test_missing_rank_fails(self):
        self.assertFails(self.run_multi({'skip': (3,)}), 'backend evidence INCOMPLETE')

    def test_mixed_attempt_fails(self):
        self.assertFails(self.run_multi({'meta_attempt': 'other'}), 'backend evidence INVALID')

    def test_cpu_only_force_run_is_not_verified(self):
        self.assertFails(self.run_multi({'consumed_ranks': ()}), 'consumed no GPU result')

    def test_nbands_mismatch_fails(self):
        write_run(self.root / 'off')
        write_run(self.root / 'on')
        for name, bands in (('off', 8), ('on', 16)):
            outcar = self.root / name / 'OUTCAR'
            outcar.write_text(f'   NBANDS=     {bands}\n' + outcar.read_text())
        res = compare_runs.compare(self.root / 'off', self.root / 'on')
        self.assertFails(res, 'effective NBANDS differ')

    # The inherited single-rank cases run again here; keep this class focused.
    test_active_gpu_verified = test_auto_all_cpu_is_valid_but_not_integration = None
    test_force_mode_without_gpu_results_fails = test_abnormal_exit = None
    test_missing_normal_termination = test_not_converged = test_nonfinite_force = None
    test_incomplete_forces = test_mismatched_inputs = test_inconsistent_counters = None
    test_shadow_check_failures = test_energy_difference = test_off_run_must_not_use_gpu = None
    test_missing_report = test_v01_report_format = None


if __name__ == '__main__':
    unittest.main()
