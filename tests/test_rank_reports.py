#!/usr/bin/env python3
"""Per-rank report route and strict aggregation (docs/CONTRACT-v0.3.md, A and B).

Runs build/rank_probe with Open MPI-style environments (no MPI library), so
16 ranks report concurrently on any machine. Synthetic tampering covers the
aggregation invariants. Requires `make all` first.
"""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parent.parent
PROBE = ROOT / 'build/rank_probe'
spec = importlib.util.spec_from_file_location('vmb_evidence', ROOT / 'scripts/vmb_evidence.py')
evidence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evidence)
CONFIG_ERROR = 78


def base_env(**extra):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('VMB_', 'OMPI_', 'PMIX_'))}
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    env.update({k: str(v) for k, v in extra.items() if v is not None})
    return env


class RankReportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name).resolve() / 'reports'
        self.dir.mkdir(mode=0o700)

    def tearDown(self):
        self.tmp.cleanup()

    def launch(self, world, calls='0,5,9', mode='force', check='1', how=None, attempt='A1',
               ranks=None, wait=True, **extra):
        procs = []
        for rank in (range(world) if ranks is None else ranks):
            env = base_env(VMB_MODE=mode, VMB_CHECK=check, VMB_REPORT_DIR=self.dir,
                           VMB_ATTEMPT_ID=attempt, VMB_EXPECT_WORLD_SIZE=world,
                           VMB_TELEMETRY='rank', OMPI_COMM_WORLD_RANK=rank,
                           OMPI_COMM_WORLD_SIZE=world, OMPI_COMM_WORLD_LOCAL_RANK=rank,
                           OMPI_COMM_WORLD_LOCAL_SIZE=world, PMIX_RANK=rank, **extra)
            argv = [str(PROBE), calls] + ([how(rank)] if how and how(rank) else [])
            procs.append(subprocess.Popen(argv, env=env, stdout=subprocess.DEVNULL,
                                          stderr=subprocess.PIPE))
        if wait:
            return [(p.wait(timeout=120), p.stderr.read().decode()) for p in procs]
        return procs

    def single(self, env, calls='none'):
        p = subprocess.run([str(PROBE), calls], env=env, capture_output=True, text=True, timeout=60)
        return p.returncode, p.stderr

    def agg(self, world=16, **kw):
        return evidence.aggregate(self.dir, kw.pop('attempt', 'A1'), world, **kw)

    # 16 concurrent ranks, three of which call ZGEMM; the rest are zero-call ranks.
    def test_sixteen_concurrent_ranks_complete(self):
        started = time.time()
        results = self.launch(16)
        self.assertEqual([code for code, _ in results], [0] * 16, results)
        res = self.agg(mode='force', check=True, executable=str(PROBE.resolve()),
                       not_before=started - 1)
        self.assertEqual(res['evidence'], 'COMPLETE', res['problems'])
        self.assertEqual(res['ranks_reported'], list(range(16)))
        self.assertEqual(res['ranks_with_consumed'], [0, 5, 9])
        self.assertEqual(res['totals']['calls'], 3)
        self.assertEqual(res['totals']['consumed'], 3)
        self.assertEqual(res['execution'], 'GPU_CONSUMED')
        self.assertEqual(res['shadow_check'], 'PASSED')
        self.assertTrue(res['gpu_qualified'])
        zero = json.loads((self.dir / 'rank-00007.report.json').read_text())
        self.assertEqual(zero['counters']['calls'], 0)
        self.assertFalse(zero['device']['initialized'])  # reporting never initialises Metal
        self.assertIsNone(zero['cpu_library'])
        self.assertEqual(sorted(p.name for p in self.dir.iterdir() if p.name.startswith('.')), [])

    def test_off_mode_is_cpu_only_not_qualified(self):
        results = self.launch(16, mode='off', check='0', calls='all')
        self.assertEqual({c for c, _ in results}, {0})
        res = self.agg(mode='off', check=False)
        self.assertEqual(res['evidence'], 'COMPLETE', res['problems'])
        self.assertEqual((res['execution'], res['shadow_check']), ('CPU_ONLY', 'NOT_ENABLED'))
        self.assertEqual(res['totals']['cpu_calls'], 16)
        self.assertFalse(res['gpu_qualified'])

    def test_rank_leaving_without_report_is_missing(self):
        results = self.launch(16, how=lambda r: 'noreport' if r == 4 else None)
        self.assertEqual({c for c, _ in results}, {0})
        res = self.agg()
        self.assertEqual(res['evidence'], 'INCOMPLETE', res['problems'])
        self.assertEqual((res['missing_ranks'], res['claimed_without_report']), ([4], [4]))
        self.assertFalse(res['gpu_qualified'])

    def test_killed_rank_is_missing_not_zero_call(self):
        procs = self.launch(4, calls='none', how=lambda r: 'sleep' if r == 2 else None, wait=False)
        for i, p in enumerate(procs):
            if i == 2:
                deadline = time.time() + 30
                while not (self.dir / 'rank-00002.claim.json').exists() and time.time() < deadline:
                    time.sleep(.05)
                p.send_signal(signal.SIGKILL)
            p.wait(timeout=60)
        res = self.agg(world=4)
        self.assertEqual((res['evidence'], res['missing_ranks']), ('INCOMPLETE', [2]))

    def test_duplicate_rank_is_refused_and_invalid(self):
        first = self.launch(2, ranks=[0, 1])
        second = self.launch(2, ranks=[1])
        self.assertEqual([c for c, _ in first], [0, 0])
        self.assertEqual(second[0][0], CONFIG_ERROR)
        self.assertIn('already claimed', second[0][1])
        self.assertEqual(self.agg(world=2)['evidence'], 'INVALID')

    def test_malformed_and_contradictory_rank_information(self):
        good = dict(VMB_REPORT_DIR=self.dir, VMB_ATTEMPT_ID='A1', VMB_MODE='off')
        cases = [dict(OMPI_COMM_WORLD_RANK='01', OMPI_COMM_WORLD_SIZE='4'),
                 dict(OMPI_COMM_WORLD_RANK='-1', OMPI_COMM_WORLD_SIZE='4'),
                 dict(OMPI_COMM_WORLD_RANK='x', OMPI_COMM_WORLD_SIZE='4'),
                 dict(OMPI_COMM_WORLD_RANK='4', OMPI_COMM_WORLD_SIZE='4'),
                 dict(OMPI_COMM_WORLD_RANK='0', OMPI_COMM_WORLD_SIZE='0'),
                 dict(OMPI_COMM_WORLD_RANK='1'),
                 dict(OMPI_COMM_WORLD_RANK='1', OMPI_COMM_WORLD_SIZE='4', PMIX_RANK='2'),
                 dict(PMIX_RANK='0'),
                 dict(OMPI_COMM_WORLD_RANK='1', OMPI_COMM_WORLD_SIZE='4', VMB_EXPECT_WORLD_SIZE='16'),
                 dict(OMPI_COMM_WORLD_RANK='1', OMPI_COMM_WORLD_SIZE='4',
                      OMPI_COMM_WORLD_LOCAL_RANK='5', OMPI_COMM_WORLD_LOCAL_SIZE='4'),
                 dict(VMB_ATTEMPT_ID=None), dict(VMB_ATTEMPT_ID='../x'), dict(VMB_ATTEMPT_ID='a' * 65),
                 dict(VMB_REPORT='/tmp/x'), dict(VMB_TELEMETRY='/tmp/t.tsv')]
        for case in cases:
            env = base_env(**{**good, **case})
            if case.get('VMB_ATTEMPT_ID', 1) is None:
                env.pop('VMB_ATTEMPT_ID', None)
            code, err = self.single(env)
            self.assertEqual(code, CONFIG_ERROR, (case, err))
            self.assertIn('report configuration error', err)
        self.assertEqual(list(self.dir.iterdir()), [])

    def test_unsafe_directories(self):
        link = self.dir.parent / 'link'
        link.symlink_to(self.dir)
        loose = self.dir.parent / 'loose'
        loose.mkdir(mode=0o700)
        loose.chmod(0o770)
        for target in (link, loose, Path('relative/reports'), self.dir / 'absent',
                       self.dir.parent / 'reports/../reports'):
            code, err = self.single(base_env(VMB_REPORT_DIR=target, VMB_ATTEMPT_ID='A1'))
            self.assertEqual(code, CONFIG_ERROR, (target, err))
        self.assertEqual(list(self.dir.iterdir()), [])

    def test_legacy_literal_paths(self):
        report = self.dir.parent / 'legacy.txt'
        code, err = self.single(base_env(VMB_MODE='off', VMB_REPORT=report), calls='all')
        self.assertEqual(code, 0, err)
        self.assertIn('vmb.version=0.5', report.read_text())  # single-process route unchanged
        code, err = self.single(base_env(VMB_REPORT=report, OMPI_COMM_WORLD_RANK='0',
                                         OMPI_COMM_WORLD_SIZE='2'))
        self.assertEqual(code, CONFIG_ERROR, err)
        code, err = self.single(base_env(OMPI_COMM_WORLD_RANK='bad', VMB_MODE='force'), calls='all')
        self.assertEqual(code, 0, err)  # rank variables are not examined without reporting

    # Aggregation tampering on a valid 4-rank directory.
    def tampered(self, edit, rank=1, name='report'):
        self.launch(4, calls='0,1', ranks=None)
        path = self.dir / f'rank-{rank:05d}.{name}.json'
        data = json.loads(path.read_text())
        edit(data)
        path.chmod(0o600)
        path.write_text(json.dumps(data))
        return self.agg(world=4)

    def test_counter_invariants_rejected(self):
        def set_counter(key, value):
            return lambda d: d['counters'].__setitem__(key, value)
        edits = [set_counter('calls', 2), set_counter('cpu_calls', 1), set_counter('gpu_completed', 0),
                 set_counter('gpu_submitted', 0), set_counter('checked', 0), set_counter('consumed', -1),
                 set_counter('calls', 1.0),
                 lambda d: d['statuses'].__setitem__('ok', 0),
                 lambda d: d['statuses'].__setitem__('range', 1),
                 lambda d: d.__setitem__('max_check_ratio_state', 'none'),
                 lambda d: d.__setitem__('max_check_ratio', -1.0),
                 lambda d: d.__setitem__('complete', False),
                 lambda d: d.__setitem__('rank', 2),
                 lambda d: d.__setitem__('world_size', 16),
                 lambda d: d.__setitem__('attempt_id', 'OTHER'),
                 lambda d: d.__setitem__('mode', 'auto'),
                 lambda d: d['process'].__setitem__('executable_stat', [1, 2, 3, 4]),
                 lambda d: d['backend'].__setitem__('source_digest', 'x'),
                 lambda d: d['times'].__setitem__('start_unix', d['times']['end_unix'] + 1),
                 lambda d: d['process'].__setitem__('pid', d['process']['pid'] + 1)]
        for edit in edits:
            with self.subTest(edit=edit):
                self.tearDown()
                self.setUp()
                res = self.tampered(edit)
                self.assertEqual(res['evidence'], 'INVALID', res)
                self.assertFalse(res['gpu_qualified'])

    def test_shadow_failure_and_nonfinite_ratio(self):
        res = self.tampered(lambda d: d.__setitem__('max_check_ratio', 2.5))
        self.assertEqual((res['evidence'], res['shadow_check']), ('COMPLETE', 'FAILED'))
        self.assertFalse(res['gpu_qualified'])
        self.tearDown()
        self.setUp()
        res = self.tampered(lambda d: d.update(max_check_ratio=None, max_check_ratio_state='nonfinite'))
        self.assertEqual((res['shadow_check'], res['max_check_ratio']), ('FAILED', 'nonfinite'))

    def test_maximum_ratio_is_not_summed(self):
        self.launch(4, calls='all')
        ratios = [json.loads((self.dir / f'rank-{r:05d}.report.json').read_text())['max_check_ratio']
                  for r in range(4)]
        self.assertEqual(self.agg(world=4)['max_check_ratio'], max(ratios))

    def test_truncated_nonfinite_stray_and_stale(self):
        self.launch(4)
        good = self.dir / 'rank-00002.report.json'
        text = good.read_text()
        good.chmod(0o600)
        good.write_text(text[: len(text) // 2])
        self.assertIn('truncated', ' '.join(self.agg(world=4)['problems']))
        good.write_text(text.replace('"end_unix": ', '"end_unix": NaN, "x": '))
        self.assertEqual(self.agg(world=4)['evidence'], 'INVALID')
        good.write_text(text)
        self.assertEqual(self.agg(world=4)['evidence'], 'COMPLETE')
        (self.dir / '.rank-00002.report.json.99.tmp').write_text('{')  # interrupted publication
        self.assertEqual(self.agg(world=4)['evidence'], 'INVALID')
        (self.dir / '.rank-00002.report.json.99.tmp').unlink()
        self.assertEqual(self.agg(world=4, not_before=time.time() + 10)['evidence'], 'INVALID')
        self.assertEqual(self.agg(world=4, attempt='A2')['evidence'], 'INVALID')
        self.assertEqual(self.agg(world=5)['evidence'], 'INVALID')
        self.assertEqual(self.agg(world=4, mode='off')['evidence'], 'INVALID')

    def test_timing_fields_are_summed_and_optional(self):
        self.launch(4, calls='0,1')
        reports = [json.loads((self.dir / f'rank-{r:05d}.report.json').read_text()) for r in range(4)]
        self.assertTrue(all(sum(r['work_decades']) == r['counters']['calls'] for r in reports))
        res = self.agg(world=4)
        self.assertEqual(res['evidence'], 'COMPLETE', res['problems'])
        self.assertEqual(res['timing']['ranks_with_timing'], 4)
        self.assertEqual(sum(res['work_decades']), 2)
        self.assertAlmostEqual(res['timing']['cpu_s'], sum(r['timing']['cpu_s'] for r in reports))
        self.assertNotIn('timing', evidence.summary(res))  # compact summary unchanged

    def test_v03_report_without_timing_still_verifies(self):
        res = self.tampered(lambda d: (d.pop('timing'), d.pop('work_decades')))
        self.assertEqual(res['evidence'], 'COMPLETE', res['problems'])

    def test_malformed_timing_rejected(self):
        for edit in (lambda d: d['timing'].__setitem__('cpu_s', -1.0),
                     lambda d: d['timing'].__setitem__('try_s', 'fast'),
                     lambda d: d.__setitem__('work_decades', [0] * 15),
                     lambda d: d['work_decades'].__setitem__(0, d['work_decades'][0] + 1),
                     lambda d: d.pop('work_decades')):
            with self.subTest(edit=edit):
                self.tearDown()
                self.setUp()
                self.assertEqual(self.tampered(edit)['evidence'], 'INVALID')

    def test_mixed_attempts_in_one_directory(self):
        self.launch(4, ranks=[0, 1])
        self.launch(4, ranks=[2, 3], attempt='A2')
        res = self.agg(world=4)
        self.assertEqual(res['evidence'], 'INVALID')
        self.assertTrue(any('another attempt' in p or 'foreign' in p for p in res['problems']))


if __name__ == '__main__':
    unittest.main()
