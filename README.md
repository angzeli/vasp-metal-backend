# vasp-metal-backend

> [!IMPORTANT]
> Independent, unofficial and experimental. This project is not affiliated with
> or endorsed by the VASP Software GmbH. It contains no VASP source code, VASP
> executables, proprietary patches or PAW/POTCAR datasets. Bring your own
> separately licensed VASP and potentials.

An opt-in Apple Metal backend for one numerical operation used by VASP 6.6.1
`vasp_std` on Apple Silicon: complex matrix multiplication through the public
reference-BLAS interface `ZGEMM`. The backend is attached at link time; the
licensed VASP objects are relinked privately and never leave the local machine.

## Status (v0.1)

- **Standalone backend: verified.** Metal kernel, C ABI and Fortran ABI pass
  against independent CPU oracles on an Apple M3 Max (results below).
- **Private relink: verified.** Existing VASP 6.6.1 std objects relink cleanly
  with the backend; `zgemm_` now resolves inside the executable.
- **VASP integration: UNVERIFIED.** No VASP run has been made yet, because the
  memory-pressure safety gate refused to launch. The blocker and the exact next
  commands are in [docs/RESULTS.md](docs/RESULTS.md).
- Default off. No performance claim is made.

## How it works

- `libvmb.a` defines `zgemm_` (LP64 integers, gfortran hidden string lengths).
  Linked ahead of OpenBLAS, it receives every ZGEMM call made by VASP's own
  objects. Calls made inside ScaLAPACK or OpenBLAS stay inside those libraries.
- With `VMB_ENABLE=1`, a supported call is packed into Metal shared buffers and
  computed by `src/zgemm_df64.metal` in float-float arithmetic (about 48-bit
  significands; Apple GPUs have no FP64). The kernel is compiled at runtime in
  IEEE-safe math mode. See [docs/PRECISION.md](docs/PRECISION.md).
- Any other call, and any GPU failure, goes to the original OpenBLAS `zgemm_`,
  resolved from the loaded OpenBLAS image itself (Metal pulls in Accelerate, so
  plain `RTLD_NEXT` would pick the wrong BLAS). C is written only after the whole
  GPU result has been validated, so there are no partial writes.
- Counters distinguish calls, GPU submissions, completed command buffers and
  results actually written into the caller's C (`consumed`).

## Build and test

Requires Apple Silicon, the Command Line Tools (clang, Metal framework), Homebrew
GCC 16.1 gfortran and OpenBLAS 0.3.34 at the paths in the `Makefile`.

```sh
make -j1 all
make -j1 test
```

## Use with a private VASP build

```sh
scripts/relink-vasp.sh /path/to/vasp/build/std private/relink-NEW
VMB_ENABLE=1 VMB_REPORT=vmb-report.txt mpirun -np 1 private/relink-NEW/vasp_std_vmb
```

`scripts/run_si2.py` stages a private Si2 case and performs one bounded launch
(1 rank, one library thread, NCORE = KPAR = 1, 60 s timeout, memory-pressure
gate, ledger capped at two launches, own process group, no retries).
`scripts/compare_runs.py OFF_DIR ON_DIR` applies the verdict rule.

| Variable | Default | Meaning |
| --- | --- | --- |
| `VMB_ENABLE` | off | `1` enables the GPU path |
| `VMB_CHECK` | off | `1` compares every GPU result with OpenBLAS before use |
| `VMB_REPORT` | none | file path (or `stderr`) for the exit report; single rank |
| `VMB_MIN_WORK` | 0 | minimum m·n·k sent to the GPU |
| `VMB_MAX_WORK` | 2^34 | maximum m·n·k sent to the GPU |
| `VMB_MAX_BYTES` | 2^30 | maximum packed buffer size |
| `VMB_TEST_NO_DEVICE`, `VMB_TEST_FAULT` | off | test-only fault injection |

## Standalone results (Apple M3 Max)

`build/test_zgemm` (100 cases) and `build/test_fortran_abi` (11 calls) pass.
Tolerance per element: `1e-18·max S + 8(k+2)·2^-48·S_ij` (fixed before testing).

| Group | Worst err/tol | Worst max\|err\|/S |
| --- | --- | --- |
| 81-case sweep: all N/T/C pairs (mixed case), awkward sizes, padded ld, 4 alpha/beta pairs | 1.09e-1 | 9.3e-15 |
| Values spanning 1e±30, 1e-150, 1e+150 | 8.3e-3 | 1.0e-14 |
| Cancellation (result about 2^-30 of S) | 3.1e-4 | 5.9e-16 |
| Fortran ABI, quad-precision oracle | 2.7e-2 | — |

Fallbacks (non-finite input, quick returns, invalid lda, below threshold, no
device, injected GPU error, injected non-finite GPU output, disabled) all route
to OpenBLAS exactly once and are bit-identical to calling OpenBLAS directly.
Repeated GPU calls are bitwise deterministic. A 32×32×32 call takes about
0.3 ms wall time including packing and synchronization.

## Limitations

- One operation (ZGEMM); one rank tested; `VMB_REPORT` is not rank-aware.
- Float-float arithmetic is not IEEE FP64; see the tolerance policy.
- Paths are pinned to the validated Homebrew toolchain.

## License

The original work in this repository is released under the MIT License (see
`LICENSE`). It grants no rights to VASP software, source, executables or
PAW/POTCAR datasets.
