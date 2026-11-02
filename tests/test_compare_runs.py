#!/usr/bin/env python3
"""Tests of scripts/compare_runs.py on synthetic run directories.

Every fixture file is invented placeholder text that only mimics the handful of
output lines the parser reads; no scientific input or output is used.
"""
import importlib.util
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


if __name__ == '__main__':
    unittest.main()
