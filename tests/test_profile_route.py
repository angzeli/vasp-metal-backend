#!/usr/bin/env python3
"""v0.5 profile route: rank-RRRRR.profile.json and its strict aggregation.

Runs build/test_blas_fftw (bit-identical forwarding of dgemv_, dgemm_ and the
dfftw_* Fortran interface) under Open MPI-style per-rank environments with
profiling on, then checks the profile contents against the calls it made and
the aggregator's handling of declared, missing and foreign profiles.
Requires `make all` first.
"""
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent
PROBE = ROOT / 'build/test_blas_fftw'
spec = importlib.util.spec_from_file_location('vmb_evidence', ROOT / 'scripts/vmb_evidence.py')
evidence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evidence)
CONFIG_ERROR = 78


def env_for(report_dir, rank, world, **extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith(('VMB_', 'OMPI_', 'PMIX_'))}
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', VMB_MODE='off', VMB_CHECK='0',
               VMB_REPORT_DIR=str(report_dir), VMB_ATTEMPT_ID='P1', VMB_TELEMETRY='rank',
               VMB_EXPECT_WORLD_SIZE=str(world), OMPI_COMM_WORLD_RANK=str(rank),
               OMPI_COMM_WORLD_SIZE=str(world), OMPI_COMM_WORLD_LOCAL_RANK=str(rank),
               OMPI_COMM_WORLD_LOCAL_SIZE=str(world), PMIX_RANK=str(rank))
    env.update({k: v for k, v in extra.items() if v is not None})
    for k in [k for k, v in extra.items() if v is None]:
        env.pop(k, None)
    return env


class ProfileRouteTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name).resolve() / 'reports'
        self.dir.mkdir(mode=0o700)

    def tearDown(self):
        self.tmp.cleanup()

    def run_ranks(self, world=2, **extra):
        procs = [subprocess.Popen([str(PROBE)], env=env_for(self.dir, r, world, **extra),
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                 for r in range(world)]
        outs = [p.communicate(timeout=120) + (p.returncode,) for p in procs]
        for out, err, code in outs:
            self.assertEqual(code, 0, out + err)
            self.assertIn('PASS: 0 failures', out)
        calls = dict(re.findall(r'(\w+)=(\d+)', re.search(r'^calls .*$', outs[0][0], re.M).group()))
        return {k: int(v) for k, v in calls.items()}

    def aggregate(self, world=2):
        return evidence.aggregate(self.dir, 'P1', world)

    def profile(self, rank=0):
        return json.loads((self.dir / f'rank-{rank:05d}.profile.json').read_text())

    def test_profile_counts_match_the_calls(self):
        calls = self.run_ranks(VMB_PROFILE='1')
        result = self.aggregate()
        self.assertEqual(result['evidence'], 'COMPLETE', result['problems'])
        self.assertEqual(result['profile'], {'level': 'profile', 'ranks': [0, 1]})
        for rank in (0, 1):
            prof = self.profile(rank)
            self.assertEqual(prof['level'], 'profile')
            blas, fftw = prof['blas'], prof['fftw']
            self.assertEqual(blas['totals']['dgemv']['calls'], calls['dgemv'])
            self.assertEqual(blas['totals']['dgemm']['calls'], calls['dgemm'])
            self.assertEqual(sum(s['calls'] for s in blas['shapes']),
                             calls['dgemv'] + calls['dgemm'])
            self.assertEqual(blas['dropped_keys'], 0)
            self.assertNotIn('reuse', blas)
            self.assertEqual(fftw['totals']['plans'], calls['fftw_plans'])
            self.assertEqual(fftw['totals']['executes'], calls['fftw_executes'])
            self.assertEqual(sum(s['destroyed'] for s in fftw['shapes']), calls['fftw_destroys'])
            self.assertEqual(fftw['unknown_executes'], 0)
            repeated = [s for s in fftw['shapes'] if s['kind'] == 'dft_3d']
            self.assertEqual(len(repeated), 1)
            self.assertEqual((repeated[0]['plans'], repeated[0]['executes'],
                              repeated[0]['new_array_executes']), (3, 6, 3))
            self.assertEqual(repeated[0]['n'], [6, 5, 8])
            for s in blas['shapes'] + fftw['shapes']:
                self.assertEqual(s['site']['image'], 'test_blas_fftw')
                self.assertIsInstance(s['site']['offset'], int)
            self.assertTrue(blas['cpu_library'].startswith('libopenblas'))
            self.assertTrue(fftw['library'].startswith('libfftw3.'))
        text = (self.dir / 'rank-00000.profile.json').read_text()
        self.assertNotIn(str(self.dir), text)
        self.assertNotIn('/', text)  # basenames only, no paths

    def test_reuse_probe(self):
        self.run_ranks(world=1, VMB_PROFILE='reuse')
        result = self.aggregate(world=1)
        self.assertEqual(result['evidence'], 'COMPLETE', result['problems'])
        reuse = self.profile()['blas']['reuse']
        # All DGEMV calls come from the one call site in gemv_case: runs of 3, 1, 2
        # and 1 calls with unchanged matrix contents (the third and fourth cases
        # differ in dimensions from their predecessors).
        (site,) = reuse['sites']
        self.assertEqual(site['calls'], 7)
        self.assertEqual(site['same_content'], 3)
        self.assertEqual(site['runs'], [2, 1, 1, 0, 0, 0, 0, 0])
        self.assertEqual(site['site']['image'], 'test_blas_fftw')

    def test_default_build_has_no_profile(self):
        self.run_ranks(VMB_PROFILE=None)
        result = self.aggregate()
        self.assertEqual(result['evidence'], 'COMPLETE', result['problems'])
        self.assertIsNone(result['profile'])
        self.assertFalse(list(self.dir.glob('*.profile.json')))

    def test_profile_requires_rank_telemetry(self):
        env = env_for(self.dir, 0, 1, VMB_PROFILE='1', VMB_TELEMETRY=None)
        code = subprocess.run([str(PROBE)], env=env, capture_output=True).returncode
        self.assertEqual(code, CONFIG_ERROR)
        env = {k: v for k, v in os.environ.items() if not k.startswith(('VMB_', 'OMPI_', 'PMIX_'))}
        env['VMB_PROFILE'] = '1'
        code = subprocess.run([str(PROBE)], env=env, capture_output=True).returncode
        self.assertEqual(code, CONFIG_ERROR)
        env = env_for(self.dir, 0, 1, VMB_PROFILE='yes')
        code = subprocess.run([str(PROBE)], env=env, capture_output=True).returncode
        self.assertEqual(code, CONFIG_ERROR)

    def test_tampering_is_rejected(self):
        self.run_ranks(VMB_PROFILE='1')
        path = self.dir / 'rank-00001.profile.json'
        original = path.read_text()

        def verdict():
            result = self.aggregate()
            return result['evidence'], ' '.join(result['problems'])

        path.unlink()
        self.assertEqual(verdict()[0], 'INVALID')
        self.assertIn('declared profile file is missing', verdict()[1])
        for change, message in (
                (lambda d: d.update(pid=d['pid'] + 1), 'another attempt, rank or process'),
                (lambda d: d.update(attempt_id='P2'), 'another attempt, rank or process'),
                (lambda d: d.update(level='reuse'), 'level differs'),
                (lambda d: d.update(schema='vmb.rank-profile.v0'), 'not a complete'),
                (lambda d: d.pop('fftw'), 'lacks its blas or fftw')):
            doc = json.loads(original)
            change(doc)
            path.write_text(json.dumps(doc))
            state, problems = verdict()
            self.assertEqual(state, 'INVALID')
            self.assertIn(message, problems)
        # A profile the report does not declare is stray evidence.
        path.write_text(original)
        report = self.dir / 'rank-00001.report.json'
        doc = json.loads(report.read_text())
        doc.pop('profile')
        report.write_text(json.dumps(doc))
        state, problems = verdict()
        self.assertEqual(state, 'INVALID')
        self.assertIn('without a report declaration', problems)


if __name__ == '__main__':
    unittest.main()
