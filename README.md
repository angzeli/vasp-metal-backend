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

## Status

| | v0.1 (`d3be811`) | v0.2 (final safety fix `afb272e`) | v0.3 (unreleased) |
| --- | --- | --- | --- |
| Standalone correctness | verified | **passed** at `afb272e`: 138 C cases, 12 Fortran ABI calls, 15 run-comparison cases | numerical gates unchanged and passing; per-rank report and aggregation tests added |
| Private relink | verified | verified (link only, `0344fdb` build) | link-only relink on the Mac mini with the port's toolchain convention |
| VASP integration | **verified** on Si2: 802/802 GPU results consumed, same TOTEN | **verified on the earlier `0344fdb` build** (Si2: 802/802 consumed, 0 shadow failures, 0 range fallbacks, same TOTEN); **`afb272e` not VASP-tested** | 16-rank managed qualification: see [docs/RESULTS-v0.3.md](docs/RESULTS-v0.3.md) |
| VASP performance acceleration | not established | **not established**; synthetic benchmarks exploratory only (busy host) | **not established** |

v0.3 adds managed 16-rank execution evidence. Each MPI rank writes its own
versioned report, `scripts/vmb_evidence.py` aggregates them strictly, and the
VASP port's runner selects the relinked binary only through a reviewed, pinned
manifest. The contract is in [docs/CONTRACT-v0.3.md](docs/CONTRACT-v0.3.md),
and the changes are listed in [CHANGELOG.md](CHANGELOG.md).

- **Correctness is not speedup.** The verified Si2 integrations show that GPU
  results can feed a converging SCF without changing the energy. At those sizes
  the GPU path is slower: VASP's elapsed time was 0.296 s with the backend off
  and 1.468 s on (v0.2 run, shadow check and telemetry enabled).
- **v0.1 defect fixed in v0.2.** v0.1 could silently consume wrong results for
  operands whose scales differ by about 2^90 or more within a matrix. The
  verified Si2 run did not contain such operands. v0.2 adds an
  exponent-range safety envelope with a counted fallback; see
  [docs/PRECISION.md](docs/PRECISION.md).
- **Overflow fix after the Si2 run.** PR #1 review found that the v0.2
  envelope did not bound the unscaled products and sums the CPU path forms.
  With a tiny alpha, a call whose FP64 evaluation overflows could reach the
  GPU. The fix (`afb272e`) rejects such calls as `range`. It passed the
  standalone regressions only; the Si2 result above is from the build before
  it (`0344fdb`). It is expected not to change Si2's routing, but no new VASP
  run has confirmed that.
- **Bound scope.** The overflow bound covers the intermediates of a
  conventional ZGEMM evaluation, not 3M/Gauss multiplication or other
  reassociations; see [docs/PRECISION.md](docs/PRECISION.md).
- **Exploratory benchmarks.** On a busy host, forced Metal beat single-thread
  OpenBLAS by 1.2 to 3.4× for synthetic shapes with m·n·k ≥ about 1.5e7. It was
  10 to 100× slower for Si2-sized shapes, because each call carries about
  0.25 ms of submit/sync latency. No auto rule is shipped from this data.
- **Real shapes are small.** Telemetry from the v0.2 Si2 run shows that every
  ZGEMM has m·n·k ≤ 1.5e5, at least 100× below the region where the GPU won.
  `auto` therefore keeps all of them on the CPU.
- Details: [docs/RESULTS-v0.1.md](docs/RESULTS-v0.1.md),
  [docs/RESULTS-v0.2.md](docs/RESULTS-v0.2.md). Default mode is off.

## How it works

- `libvmb.a` defines `zgemm_` (LP64 integers, gfortran hidden string lengths).
  Linked ahead of OpenBLAS, it receives every ZGEMM call made by VASP's own
  objects. Calls made inside ScaLAPACK or OpenBLAS stay inside those libraries.
- `VMB_MODE` selects `off` (default), `auto` or `force`.
  - `auto` uses the GPU only in shape regions with a reproducible measured win.
    None is established yet, so `auto` currently serves everything on the CPU.
  - `force` sends every eligible call to the GPU, for validation.
- An eligible call is packed into Metal shared buffers and computed by
  `src/zgemm_df64.metal` in float-float arithmetic (about 48-bit significands;
  Apple GPUs have no FP64). The kernel is compiled at runtime in IEEE-safe math
  mode.
- Before any GPU work, an exponent-range envelope must hold; otherwise the call
  is a counted `range` fallback. Any other declined call, and any GPU failure,
  goes to the original OpenBLAS `zgemm_`. That routine is resolved from the
  loaded OpenBLAS image itself: Metal pulls in Accelerate, so plain `RTLD_NEXT`
  would pick the wrong BLAS.
- C is written only after the whole GPU result has been validated, so there are
  no partial writes.
- Counters distinguish calls, GPU submissions, completed command buffers and
  results actually written into the caller's C (`consumed`).

## Build and test

Requires Apple Silicon, the Command Line Tools (clang, Metal framework), Homebrew
GCC 16.1 gfortran and OpenBLAS 0.3.34 at the paths in the `Makefile`.

```sh
make -j1 all                    # or: make -j1 PORT_ROOT=/path/to/vasp-port all
make -j1 test                   # numerical, Fortran ABI and per-rank report tests
python3 -B tests/test_compare_runs.py
```

## Benchmark

```sh
python3 scripts/run_bench.py --output private/bench-NEW
```

The benchmark replays the synthetic shapes in `bench/shapes-v0.2.tsv`,
alternating direct single-thread OpenBLAS with forced Metal through the
Fortran-ABI entry point. Correctness is checked outside the timed regions, and
the run reports per-phase medians and quartiles.
- Caps: memory gate, one thread, a wall-time limit, and repetition and
  working-set limits.
- Timing on a busy host is labeled exploratory.

## Use with a private VASP build

```sh
scripts/relink-vasp.sh --port-root /path/to/vasp-port /path/to/vasp/build/std private/relink-NEW
VMB_MODE=force VMB_REPORT=vmb-report.txt mpirun -np 1 private/relink-NEW/vasp_std_vmb
```

For managed MPI runs, the VASP port's runner selects the relinked binary through
a reviewed manifest, written by `scripts/make-binary-manifest.py`, and sets the
per-rank variables itself (contract section C).

`scripts/run_si2.py --backend off|auto|force` stages a private Si2 case and
performs one bounded launch: 1 rank, one library thread, NCORE = KPAR = 1, 60 s
timeout, memory-pressure gate, ledger capped at two launches, own process
group, no retries.
- `--allow-warn-pressure` is an explicit override for WARN-level pressure only.
- `scripts/compare_runs.py OFF_DIR ON_DIR` classifies a pair as VASP-verified,
  CPU-only consistent or FAILED.

| Variable | Default | Meaning |
| --- | --- | --- |
| `VMB_MODE` | `off` | `off`, `auto` (measured regions only; none yet) or `force` (validation) |
| `VMB_ENABLE` | off | v0.1 switch; `1` means `force` when `VMB_MODE` is unset |
| `VMB_CHECK` | off | `1` compares every GPU result with OpenBLAS before use (fails closed) |
| `VMB_REPORT` | none | file path (or `stderr`) for the exit report; single process only (refused when the world size is above 1) |
| `VMB_TELEMETRY` | none | file path for aggregated shape telemetry (no values or addresses); `rank` with `VMB_REPORT_DIR` |
| `VMB_REPORT_DIR` | none | v0.3 per-rank route: private directory receiving one claim and one report per rank |
| `VMB_ATTEMPT_ID` | none | required with `VMB_REPORT_DIR`; binds every report to one attempt |
| `VMB_EXPECT_WORLD_SIZE` | none | optional; must equal the launcher's world size |
| `VMB_MIN_WORK` | 0 | minimum m·n·k sent to the GPU |
| `VMB_MAX_WORK` | 2^34 | maximum m·n·k sent to the GPU |
| `VMB_MAX_BYTES` | 2^30 | maximum packed buffer size |
| `VMB_TEST_NO_DEVICE`, `VMB_TEST_FAULT`, `VMB_TEST_AUTO_MIN_WORK` | off | test-only hooks |

## Standalone results (Apple M3 Max)

- `build/test_zgemm`: 138 cases.
- `build/test_fortran_abi`: 12 calls, against a quad-precision oracle.
- `tests/test_rank_reports.py`: 13 cases (v0.3 per-rank route and aggregation).
- `tests/test_compare_runs.py`: 20 cases (15 single-rank, 5 multi-rank).

All pass. The tolerance per element, fixed before testing, is
`1e-18·max S + 8(k+2)·2^-48·S_ij`.

| Group | Worst err/tol | Worst max\|err\|/S |
| --- | --- | --- |
| 81-case sweep: all N/T/C pairs (mixed case), awkward sizes, padded ld, 4 alpha/beta pairs | 1.09e-1 | 9.3e-15 |
| Values spanning 1e±30, 1e-150, 1e+120 | 8.3e-3 | 1.0e-14 |
| Cancellation (result about 2^-30 of S) | 3.1e-4 | 5.9e-16 |
| Mixed scales up to 2^±200, beta·C imbalance 2^±300 (v0.2) | 1.6e-2 | — |
| Fortran ABI, quad-precision oracle | 2.7e-2 | — |

The following cases route to OpenBLAS exactly once and are bit-identical to
calling OpenBLAS directly:
- out-of-envelope operands: the diag(2^±90) regression, products near FP64
  underflow or overflow, unscaled products or partial sums that overflow under
  a tiny alpha, `inf - inf`, all-zero products;
- non-finite input, quick returns and invalid lda;
- a call below the size threshold, no device, or mode off or auto;
- injected faults: a GPU error, a non-finite GPU output, and a perturbed GPU
  output caught by the shadow check.

Repeated GPU calls are bitwise deterministic.

## Limitations

- One operation (ZGEMM). VASP integration has been tested only on one tiny
  system (Si2).
- The per-rank route supports Open MPI launches only; other launchers are not
  yet scoped. The literal `VMB_REPORT` and `VMB_TELEMETRY` paths remain
  single-process only.
- Float-float arithmetic is not IEEE FP64; see the tolerance policy. The range
  envelope is conservative.
- Timing so far is exploratory: busy host, single-thread CPU baseline,
  synthetic shapes.
- Paths are pinned to the validated Homebrew toolchain.

## License

The original work in this repository is released under the MIT License (see
`LICENSE`). It grants no rights to VASP software, source, executables or
PAW/POTCAR datasets.
