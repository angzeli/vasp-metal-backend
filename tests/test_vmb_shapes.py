#!/usr/bin/env python3
"""Shape census (scripts/vmb_shapes.py): merging, regions, bounds and replay export.

Real per-rank telemetry comes from build/rank_probe with Open MPI-style rank
variables (no MPI library); the region arithmetic uses small synthetic tables.
Requires `make all` first.
"""
import importlib.util
import io
import json
import os
from contextlib import redirect_stdout
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f'scripts/{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shapes = load('vmb_shapes')
evidence = load('vmb_evidence')
HEADER = ('# vmb shape telemetry v1; times are seconds per zgemm_ call (host entry to return); '
          'dropped_shapes={dropped}\n'
          'm\tn\tk\ttransa\ttransb\tlda\tldb\tldc\tbeta\tstatus\tcount\ttotal_s\tmin_s\tmax_s\n')


class CensusTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run = Path(self.tmp.name).resolve() / 'run'
        (self.run / 'vmb-reports').mkdir(parents=True, mode=0o700)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, rank, rows, dropped=0):
        lines = ''.join('\t'.join(str(v) for v in row) + '\n' for row in rows)
        (self.run / f'vmb-reports/rank-{rank:05d}.telemetry.tsv').write_text(HEADER.format(dropped=dropped) + lines)

    def test_real_rank_telemetry_reconciles_with_evidence(self):
        world = 4
        for rank in range(world):
            env = {k: v for k, v in os.environ.items() if not k.startswith(('VMB_', 'OMPI_', 'PMIX_'))}
            env.update(VMB_MODE='off', VMB_REPORT_DIR=str(self.run / 'vmb-reports'), VMB_ATTEMPT_ID='census',
                       VMB_TELEMETRY='rank', OMPI_COMM_WORLD_RANK=str(rank), OMPI_COMM_WORLD_SIZE=str(world),
                       OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            subprocess.run([str(ROOT / 'build/rank_probe'), 'all'], env=env, check=True,
                           stdout=subprocess.DEVNULL, timeout=60)
        agg = evidence.aggregate(self.run / 'vmb-reports', 'census', world)
        (self.run / 'vmb-evidence.json').write_text(json.dumps(agg))
        (self.run / 'RUN_METADATA.txt').write_text('state: FINISHED\nelapsed_s: 2.0\n')
        result = shapes.census(self.run, [shapes.DEFAULT_REGION, {'mnk': 1e4}])
        self.assertEqual((result['calls'], result['distinct_shapes'], result['ranks']), (4, 1, [0, 1, 2, 3]))
        self.assertTrue(result['reconciliation']['agrees'])
        self.assertEqual(result['regions'][0]['calls'], 0)  # 32^3 is below the v0.2 candidate
        self.assertEqual(result['regions'][1]['calls'], 4)  # but inside mnk >= 1e4
        self.assertAlmostEqual(result['regions'][1]['share_of_zgemm_time'], 1.0)
        self.assertEqual(result['decades'], [{'decade': 4, 'calls': 4,
                                              'total_s': result['zgemm_rank_seconds']}])

    def test_regions_bounds_and_replay_export(self):
        big = [512, 512, 512, 'N', 'N', 512, 512, 512, 'zero', 1, 2, 0.6, 0.3, 0.3]
        small = [8, 8, 130, 'C', 'N', 130, 135, 8, 'one', 1, 100, 0.4, 0.004, 0.004]
        self.write(0, [big, small])
        self.write(1, [big, small])
        (self.run / 'RUN_METADATA.txt').write_text('elapsed_s: 10.0\n')
        (self.run / 'OUTCAR').write_text('     LOOP+:  cpu time      4.0: real time      4.5\n'
                                         '     LOOP+:  cpu time      5.0: real time      5.25\n')
        result = shapes.census(self.run, [shapes.DEFAULT_REGION])
        region = result['regions'][0]
        self.assertEqual((region['shapes'], region['calls']), (1, 4))
        self.assertAlmostEqual(region['share_of_zgemm_time'], 1.2 / 2.0)
        self.assertAlmostEqual(region['upper_bound_share_of_rank_time'], 1.2 / (2 * 10.0))
        self.assertAlmostEqual(result['wall']['outcar_loop_real_s'], 9.75)
        self.assertIsNone(result['reconciliation'])  # no evidence file: not claimed reconciled
        rows = shapes.replay_rows(result['_shapes'], 'tio2-r8', 'select', 5)
        self.assertEqual(rows[0][:10], ['tio2-r8-01', 'tio2-r8', 'observed-tio2-r8', 'select', 'N', 'N',
                                        512, 512, 512, 0])
        self.assertEqual(rows[1][4:13], ['C', 'N', 8, 8, 130, 0, 5, 0, 'one'])  # rows_a = k, rows_b = k

    def test_cli_writes_new_files_only(self):
        self.write(0, [[64, 64, 64, 'N', 'N', 64, 64, 64, 'zero', 0, 3, 0.03, 0.01, 0.01]])
        out, replay = self.run / 'census.json', self.run / 'replay.tsv'
        with redirect_stdout(io.StringIO()):
            self.assertEqual(shapes.main(['census', str(self.run), '--output', str(out), '--replay', str(replay),
                                          '--label', 'probe']), 0)
        self.assertEqual(json.loads(out.read_text())['calls'], 3)
        self.assertIn('probe-01\tprobe\tobserved-probe\tselect', replay.read_text())
        with redirect_stdout(io.StringIO()), self.assertRaises(FileExistsError):
            shapes.main(['census', str(self.run), '--output', str(out)])

    def test_malformed_telemetry_rejected(self):
        bad_rows = ([[1, 2, 3, 'N', 'N', 1, 3, 1, 'zero', 0, 0, 0.1, 0.1, 0.1]],       # zero count
                    [[1, 2, 3, 'X', 'N', 1, 3, 1, 'zero', 0, 1, 0.1, 0.1, 0.1]],       # bad trans
                    [[1, 2, 3, 'N', 'N', 1, 3, 1, 'zero', 0, 1, float('nan'), 0, 0]],  # non-finite
                    [[1, 2, 3, 'N', 'N', 1, 3, 1, 'zero', 0, 1]])                      # truncated
        for rows in bad_rows:
            with self.subTest(rows=rows):
                self.write(0, rows)
                with self.assertRaises(shapes.CensusError):
                    shapes.census(self.run, [shapes.DEFAULT_REGION])
        (self.run / 'vmb-reports/rank-00000.telemetry.tsv').write_text('no header\n')
        with self.assertRaises(shapes.CensusError):
            shapes.census(self.run, [shapes.DEFAULT_REGION])

    def test_region_parsing(self):
        self.assertEqual(shapes.parse_region('mn=16384,mnk=1.5e7'), {'mn': 16384.0, 'mnk': 1.5e7})
        for text in ('mn>=1', 'q=1', 'mn=abc'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                shapes.parse_region(text)


if __name__ == '__main__':
    unittest.main()
