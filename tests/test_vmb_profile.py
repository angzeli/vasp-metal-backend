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


# Call graph of 100 busy samples: dgemv_ under two VASP callers, FFTW under one,
# a v0.5 interposer frame in the VASP image (skipped as a caller), and a VASP
# routine whose name contains the pattern (must not match).
TREE = """Analysis of sampling vasp_std_vmb (pid 1) every 1 millisecond
Call graph:
    200 Thread_1   DispatchQueue_1: com.apple.main-thread  (serial)
    + 100 start  (in dyld) + 1  [0x1]
    + ! 100 main  (in vasp_std_vmb) + 1  [0x2]
    + !   60 vnlacc_  (in vasp_std_vmb) + 2  [0x3]
    + !   : 30 vnlac0__  (in vasp_std_vmb) + 3  [0x4]
    + !   : | 30 dgemv_  (in libopenblasp-r0.3.34.dylib) + 4  [0x5]
    + !   : |   25 .Lgemv_n_kernel_F320  (in libopenblasp-r0.3.34.dylib) + 5  [0x6]
    + !   : 10 proj1_  (in vasp_std_vmb) + 6  [0x7]
    + !   : | 10 dgemv_  (in vasp_std_vmb) + 1  [0x7a]
    + !   : |   10 dgemv_  (in libopenblasp-r0.3.34.dylib) + 7  [0x8]
    + !   : 20 fft3d_  (in vasp_std_vmb) + 8  [0x9]
    + !   :   20 n1_64  (in libfftw3.3.dylib) + 9  [0xa]
    + !   40 my_gemv_helper_  (in vasp_std_vmb) + 10  [0xb]
    100 kevent  (in libsystem_kernel.dylib) + 1  [0xc]

Total number in stack (recursive counted multiple times):

Sort by top of stack, same collapsed (when >= 5):
        kevent  (in libsystem_kernel.dylib)        100
        .Lgemv_n_kernel_F320  (in libopenblasp-r0.3.34.dylib)        25
        dgemv_  (in libopenblasp-r0.3.34.dylib)        15
        n1_64  (in libfftw3.3.dylib)        20
        my_gemv_helper_  (in vasp_std_vmb)        40
"""


class ProfileTest(unittest.TestCase):
    def test_callers_attribute_library_frames_to_vasp_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'tree.sample.txt'
            path.write_text(TREE)
            gemv = vmb_profile.callers(path, 'gemv')
            two = vmb_profile.callers(path, 'gemv', frames=2)
            fft = vmb_profile.callers(path, 'libfftw3')
        self.assertEqual((gemv['busy_samples'], gemv['matched_samples']), (100, 40))
        self.assertEqual(gemv['callers'], {'vnlac0__': 0.3, 'proj1_': 0.1})
        self.assertEqual(two['callers'], {'vnlac0__ < vnlacc_': 0.3, 'proj1_ < vnlacc_': 0.1})
        self.assertEqual(fft['callers'], {'fft3d_': 0.2})


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
