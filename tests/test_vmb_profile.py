#!/usr/bin/env python3
"""scripts/vmb_profile.py on a synthetic `sample` report (no process is sampled)."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('vmb_profile', ROOT / 'scripts/vmb_profile.py')
vmb_profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vmb_profile)

REPORT = """Analysis of sampling vasp_std_vmb (pid 1) every 1 millisecond
Call graph:
    100 Thread_1   DispatchQueue_1: com.apple.main-thread  (serial)

Total number in stack (recursive counted multiple times):

Sort by top of stack, same collapsed (when >= 5):
        kevent  (in libsystem_kernel.dylib)        100
        .Lgemv_n_kernel_F320  (in libopenblasp-r0.3.34.dylib)        40
        .Lzgemm_kernel_L4_M4_22  (in libopenblasp-r0.3.34.dylib)        10
        n1_64  (in libfftw3.3.dylib)        20
        mca_btl_sm_component_progress  (in libopen-pal.80.dylib)        15
        __nonl_MOD_proj1_  (in vasp_std_vmb)        10
        _platform_memmove  (in libsystem_platform.dylib)        5
"""


class ProfileTest(unittest.TestCase):
    def test_busy_shares_exclude_idle_waits(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'rank.sample.txt'
            path.write_text(REPORT)
            result = vmb_profile.profile(path)
        self.assertEqual((result['busy_samples'], result['idle_samples']), (100, 100))
        self.assertEqual(result['categories'], {'BLAS gemv': 0.4, 'FFTW': 0.2, 'MPI/runtime': 0.15,
                                                'BLAS gemm': 0.1, 'VASP own code': 0.1, 'system/other': 0.05})

    def test_rejects_other_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'other.txt'
            path.write_text('not a sample report\n')
            with self.assertRaises(ValueError):
                vmb_profile.profile(path)


if __name__ == '__main__':
    unittest.main()
