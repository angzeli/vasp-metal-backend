# v0.2 results

**Status:** v0.2 is a safety and measurement release. v0.1 (`d3be811`)
results are in [RESULTS-v0.1.md](RESULTS-v0.1.md).

| Claim | Status | Build |
| --- | --- | --- |
| Standalone numerical safety: `test_zgemm` 138 cases, `test_fortran_abi` 12 calls, `test_compare_runs` 15 cases | **passed** | `afb272e` (final safety fix) |
| VASP Si2 integration: both SCFs converged, 802/802 GPU results consumed, no TOTEN difference at reported precision (section 6) | **verified** | `0344fdb` (earlier v0.2 build) |
| VASP integration of the final safety-fix build | **not tested** | `afb272e` |
| VASP performance acceleration | **not established** | — |
| Synthetic benchmark speedups (section 3) | exploratory only (busy host) | v0.2 benchmark builds, before and after the pack/unpack optimization |

The `afb272e` fix closed an FP64 overflow hole in unscaled intermediates found
in PR #1 review. It adds `range` fallbacks and changes no GPU arithmetic. Si2's
operands are far from FP64 overflow (`range` = 0 in section 6), so the fix is
**expected** to leave Si2's routing and results unchanged. No new VASP run has
confirmed this.

## Hardware and host state

| Item | Value |
| --- | --- |
| Machine | Apple M3 Max: 14 CPU cores (10 performance + 4 efficiency), 30-core GPU (Metal 4), 36 GiB |
| CPU baseline | OpenBLAS 0.3.34 (`zgemm_`), one thread (`openblas_get_num_threads() == 1`, OpenMP build with `OMP_NUM_THREADS=1`) |
| GPU path | Forced Metal through the interposed Fortran-ABI `zgemm_`, float-float kernel, shadow check off while timing |
| Initialization | Metal device, runtime kernel compilation and pipeline: 0.05 to 0.07 s, timed separately |
| Host during timing | **Busy:** a separate 8-rank CPU job was running throughout; 1-minute load 11.6 to 13.7; 11 to 12 processes above 20% CPU |
| Memory | Normal pressure at both timed runs (one later attempt was refused at WARN and not overridden) |

Because the host was busy, **every timing below is exploratory.** Under the v0.2
rules, exploratory timing cannot be used to ship an auto-mode rule.

## 1. Numerical safety

The v0.1 kernel scaled each matrix by one power of two. For
`diag(2^90, 2^-90) · diag(2^-90, 2^90)` (alpha = 1, beta = 0, N,N), a forced
call with the shadow check off **consumed a zero matrix instead of I** on
`d3be811`. v0.2 adds an exponent-range safety envelope that covers
intermediates and fails closed ([PRECISION.md](PRECISION.md)). Each call is
either accurate or a counted `range` fallback to the original OpenBLAS before
any GPU submission or write to C.

| Case (forced mode, shadow check off) | Outcome |
| --- | --- |
| diag(2^±90) 2×2 regression | range fallback; C is exactly I; direct attempt leaves C untouched |
| Same pair inside an ordinary 64×64×64 call | range fallback, bit-identical to OpenBLAS |
| Mixed scales 2^±10 / 2^±40 / 2^±200 | GPU consumed; err/tol 1.6e-2 / 1.3e-2 / 7.1e-3 |
| Products near FP64 underflow (2^-1040) | range fallback, bit-identical |
| Products that overflow; `inf - inf` on the CPU | range fallback; CPU non-finite semantics reproduced bit for bit |
| beta·C ≈ 2^300 or 2^-300 against O(1) products; one C element 2^200 | GPU consumed, within tolerance |
| All-zero A (every S_ij = 0) | range fallback, bit-identical |
| Unscaled products 2^1040, alpha = 2^-500 (exact 2^541); `0344fdb` consumed 0 | range fallback; deltas range +1, submitted 0, consumed 0, CPU +1; bit-identical to OpenBLAS (inf, NaN) |
| Unscaled partial sum 2^1024 from finite 2^1023 products, alpha = 2^-500 | same as above |
| Positive controls, same shape: alpha = 2^-500 with moderate entries; alpha = 1 | GPU submitted and consumed, bit-identical to OpenBLAS |
| Perturbed GPU result with `VMB_CHECK=1` | shadow check fails closed, CPU result used |
| Fortran ABI, forced, check off, diag regression | C = I exactly through one counted CPU fallback |

The comparison helper `vmb_tolerance_ratio` now fails closed:
- an exact match against zero tolerance passes;
- a positive error against zero tolerance fails;
- a non-finite or negative input, scale or difference fails;
- once a failure is seen, later elements cannot overwrite it.

The same bug also existed in the test harness's own ratio loop and is fixed
there too. Suites: `test_zgemm` 138 cases (134 before the overflow fix), `test_fortran_abi` 12 calls
(quad-precision oracle), `test_compare_runs` 15 cases, all passing.

## 2. Modes and telemetry

- `VMB_MODE=off` (default), `auto` or `force`. `VMB_ENABLE=1` without
  `VMB_MODE` keeps its v0.1 meaning, `force`. Neither `auto` nor `force` can
  bypass the safety envelope.
- `auto` has **no measured winning region** and serves every call on the CPU
  (counted as `auto_cpu`).
- `VMB_TELEMETRY=path` writes one row per distinct
  (m, n, k, trans flags, lda, ldb, ldc, beta class, outcome), with count and
  total/min/max seconds. It records no operand values or addresses.

## 3. Shape-replay benchmark

Command: `scripts/run_bench.py --output private/<new dir>`. It runs
`build/shape_replay` serially with:
- 7 repetitions, alternating CPU-first and Metal-first;
- caps of 20 s per shape, 300 s in total and 128 MiB working set;
- correctness checked outside the timed regions, against OpenBLAS, using the
  fixed tolerance.

Every repeated Metal result was bit-identical. Each run took under 2 s of wall
time.

**Provenance.** The only observed data are the v0.1 Si2 counts of ZGEMM work
m·n·k by decade: 339 calls in [1e3, 1e4), 419 in [1e4, 1e5) and 44 in
[1e5, 1e6). Every shape below is synthetic:
- `si2-bin` shapes were synthesized to fall in those bins;
- `cds` shapes are inspired by plausible small-CdS plane-wave and band counts.

None is a trace. `holdout` rows were reserved to test a candidate rule.

Results after the single optimization, host call to C ready, in ms:

| Shape | Provenance | Role | op | m×n×k | m·n·k | CPU ms, median [IQR] | Metal ms, median [IQR] | CPU/Metal | Verdict | err/tol |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| si2-1e3-nn | synth. si2-bin | select | NN | 130×8×8 | 8.3e+03 | 0.003 [0.002–0.003] | 0.272 [0.243–0.311] | 0.01 | cpu-faster | 1.7e-02 |
| si2-1e3-cn | synth. si2-bin | select | CN | 8×8×130 | 8.3e+03 | 0.004 [0.002–0.004] | 0.348 [0.343–0.364] | 0.01 | cpu-faster | 9.7e-04 |
| si2-1e4-cn | synth. si2-bin | select | CN | 16×16×130 | 3.3e+04 | 0.009 [0.008–0.020] | 0.385 [0.376–0.400] | 0.03 | cpu-faster | 1.2e-03 |
| si2-1e4-nn | synth. si2-bin | select | NN | 130×16×16 | 3.3e+04 | 0.008 [0.007–0.009] | 0.257 [0.254–0.272] | 0.03 | cpu-faster | 1.2e-02 |
| si2-1e5-nn | synth. si2-bin | select | NN | 130×32×32 | 1.3e+05 | 0.032 [0.032–0.033] | 0.276 [0.274–0.282] | 0.12 | cpu-faster | 6.4e-03 |
| sq-128 | synth. generic | select | NN | 128×128×128 | 2.1e+06 | 0.442 [0.440–0.448] | 0.630 [0.628–0.643] | 0.70 | cpu-faster | 1.9e-03 |
| sq-256 | synth. generic | select | NN | 256×256×256 | 1.7e+07 | 3.244 [3.226–3.254] | 2.643 [2.553–2.861] | 1.23 | gpu-faster | 1.1e-03 |
| sq-512 | synth. generic | select | NN | 512×512×512 | 1.3e+08 | 25.336 [25.163–25.493] | 7.566 [6.917–9.187] | 3.35 | gpu-faster | 8.3e-04 |
| skinny-6000x16 | synth. generic | select | NN | 6000×16×128 | 1.2e+07 | 2.760 [2.728–2.848] | 2.680 [2.403–2.800] | 1.03 | overlap | 2.5e-03 |
| longk-128-3000 | synth. generic | select | CN | 128×128×3000 | 4.9e+07 | 9.764 [9.705–9.803] | 5.119 [4.492–6.032] | 1.91 | gpu-faster | 7.5e-05 |
| cds-rotate | synth. cds | select | NN | 3000×128×128 | 4.9e+07 | 9.300 [9.251–9.519] | 4.921 [4.528–5.730] | 1.89 | gpu-faster | 2.8e-03 |
| cds-overlap | synth. cds | select | CN | 64×256×1500 | 2.5e+07 | 4.762 [4.716–4.880] | 2.556 [2.491–3.167] | 1.86 | gpu-faster | 2.0e-04 |
| padded-1000 | synth. generic | select | NN | 1000×100×100 | 1e+07 | 1.978 [1.970–1.982] | 1.665 [1.602–1.686] | 1.19 | gpu-faster | 2.7e-03 |
| holdout-tn | synth. generic | holdout | TN | 200×200×2000 | 8e+07 | 15.373 [15.323–15.454] | 7.221 [4.855–8.600] | 2.13 | gpu-faster | 1.9e-04 |
| holdout-nc | synth. generic | holdout | NC | 2048×64×64 | 8.4e+06 | 1.641 [1.634–1.652] | 1.574 [1.519–1.703] | 1.04 | overlap | 4.5e-03 |
| holdout-sq384 | synth. generic | holdout | NN | 384×384×384 | 5.7e+07 | 10.775 [10.690–10.864] | 4.435 [3.912–5.720] | 2.43 | gpu-faster | 7.8e-04 |
| holdout-cn-longk | synth. cds | holdout | CN | 32×32×4000 | 4.1e+06 | 0.886 [0.855–0.943] | 3.045 [2.607–3.104] | 0.29 | cpu-faster | 4.7e-05 |

The verdict `gpu-faster` means every Metal repetition was faster than every
CPU repetition, and `cpu-faster` means the reverse. `overlap` means neither.

Medians by phase, in ms. `wait` is host time blocked on the command buffer and
includes GPU execution; `finish` is validation plus the write into C:

| Shape | pack | encode | GPU exec | wait (incl. exec) | unpack | finish | Metal total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| si2-1e3-nn | 0.002 | 0.008 | 0.022 | 0.253 | 0.001 | 0.001 | 0.272 |
| si2-1e4-cn | 0.003 | 0.014 | 0.133 | 0.359 | 0.001 | 0.000 | 0.385 |
| si2-1e5-nn | 0.007 | 0.006 | 0.048 | 0.247 | 0.002 | 0.003 | 0.276 |
| sq-128 | 0.021 | 0.010 | 0.319 | 0.543 | 0.009 | 0.014 | 0.630 |
| sq-512 | 0.353 | 0.055 | 4.886 | 5.726 | 0.145 | 0.298 | 7.566 |
| skinny-6000x16 | 0.565 | 0.028 | 0.499 | 1.284 | 0.050 | 0.086 | 2.680 |
| cds-rotate | 0.491 | 0.023 | 2.453 | 3.269 | 0.170 | 0.404 | 4.921 |
| holdout-cn-longk | 0.158 | 0.025 | 1.730 | 2.440 | 0.001 | 0.001 | 3.045 |

### Profile and the one optimization

- **Si2-bin shapes cannot win.** OpenBLAS takes 3 to 32 µs; each Metal call
  spends about 0.25 ms in submit-to-completion latency alone, even when the
  kernel runs for 0.02 ms. This holds 10 to 100× apart, far beyond the noise.
- **Host conversion was the main mid-size cost.** In the baseline, per-element
  `ldexp` in packing and unpacking was 3.2 + 1.6 ms of 10.5 ms for
  `cds-rotate`. The one optimization replaces it with a multiply by an exact
  power of two, falling back to `ldexp` for non-normal factors. GPU outputs are
  **bit-identical** (17 dumped cases, including the fallback branch), and err/tol
  is identical for every shape.

| Shape | Metal ms before → after | pack ms before → after | unpack ms before → after | Verdict before → after |
| --- | --- | --- | --- | --- |
| sq-256 | 3.731 → 2.643 | 0.428 → 0.086 | 0.296 → 0.035 | overlap → gpu-faster |
| sq-512 | 10.949 → 7.566 | 1.844 → 0.353 | 0.950 → 0.145 | gpu-faster → gpu-faster |
| skinny-6000x16 | 6.026 → 2.680 | 3.326 → 0.565 | 0.473 → 0.050 | cpu-faster → overlap |
| cds-rotate | 10.530 → 4.921 | 3.239 → 0.491 | 1.643 → 0.170 | overlap → gpu-faster |
| cds-overlap | 4.408 → 2.556 | 1.490 → 0.294 | 0.063 → 0.010 | overlap → gpu-faster |
| padded-1000 | 3.962 → 1.665 | 0.956 → 0.139 | 0.508 → 0.053 | overlap → gpu-faster |
| holdout-nc | 3.358 → 1.574 | 0.868 → 0.163 | 0.469 → 0.056 | overlap → overlap |

Small-shape Metal totals also moved, for example 0.655 → 0.272 ms for
`si2-1e3-nn`, but their conversion cost was about 0.01 ms. That change is
run-to-run variation in submit/sync latency on a busy host, not the
optimization. It is why latency-bound timings here are treated as noisy.

### Auto-mode decision

**Shipped: no region.** `auto` serves every call on the CPU, because all timing
was exploratory.

A candidate rule for a future quiet-host confirmation, *not shipped*, would use
the GPU only when **m·n ≥ 16384 and m·n·k ≥ 1.5e7**:
- on the select shapes it admits only `gpu-faster` cases (sq-256, sq-512,
  long-k, cds-rotate, cds-overlap), and it conservatively excludes padded-1000;
- it classifies all four holdouts correctly: tn and sq384 run on the GPU,
  while nc (work 8.4e6, overlap) and cn-longk (m·n = 1024, CPU 3.4× faster)
  stay on the CPU.

Shapes with a long k and a small m·n lose because the one-thread-per-element
kernel has too few threads. A tiled kernel was not attempted, because v0.2
allowed only one optimization.

## 4. Integration tooling

- `scripts/compare_runs.py` checks, for both runs: normal exit and normal
  termination; SCF convergence; finite and complete forces; matched staged
  inputs (by digest); counter consistency; and shadow-check failures.
  - Counter consistency means: calls equal consumed plus CPU calls, statuses sum
    to calls, and submitted ≥ completed ≥ consumed, with every gap explained by
    a counted outcome.
  - Its verdict is **VASP-verified** (GPU results consumed), **CPU-only
    consistent** (a valid all-CPU `auto` run, not integration) or **FAILED**.
  - The synthetic-fixture tests found and fixed a parser bug: a truncated force
    block was read as complete.
- Re-checked with the hardened script, the v0.1 Si2 pair still classifies as
  VASP-verified: 802/802 consumed, 0 shadow failures, |ΔTOTEN| = 0.
- `scripts/run_si2.py --backend off|auto|force [--telemetry]`; `on` remains an
  alias for `force`.

## 5. Private relink

The v0.2 library relinks against the existing VASP 6.6.1 std objects with the
same script:
- 321 objects;
- an empty link log;
- every reference global symbol present, with only backend symbols added.

The binary used in section 6 is `private/relink-v0.2-20261102-125116`. It was
linked from the library at `fd8f98d`; library sources are unchanged since.

```sh
make -j1 all test
scripts/relink-vasp.sh /path/to/vasp/build/std private/relink-v0.2-NEW
```

## 6. VASP integration run (v0.2 build at `0344fdb`)

Two sequential launches, authorized explicitly, used a new ledger
(`private/launch-ledger-v0.2.txt`, now holding its two entries) and the same
Si2 inputs as v0.1 (ENCUT 300, 2×2×2 Γ mesh, EDIFF 1e-6):
- 1 MPI rank (synthetic-topology OpenMPI), one OpenMP/BLAS thread,
  NCORE = KPAR = 1;
- 60 s hard timeout, no retries.

The first attempt was refused by the default memory gate (WARN pressure,
load average about 185 while an unrelated archive job ran). Both launches then
used the user-authorized `--allow-warn-pressure` override. Pressure was WARN
(38 to 44% free, never critical) before, between and after them; the separate
8-rank job kept running and was only observed.

| | Backend off | Force, `VMB_CHECK=1`, telemetry |
| --- | --- | --- |
| Exit / termination | 0 / normal | 0 / normal |
| SCF | converged (EDIFF reached), 12 iterations | converged (EDIFF reached), 12 iterations |
| Final TOTEN, eV | −8.28298469 | −8.28298469 |
| ZGEMM calls | 802 (all OpenBLAS, mode off) | 802 |
| GPU submitted / completed / consumed | 0 / 0 / 0 | **802 / 802 / 802** |
| Range fallbacks, CPU calls | 0, 802 | **0**, 0 |
| Shadow check | — | 802 checked, 0 failures, worst err/tol 4.39e-2 |
| VASP elapsed, s | 0.296 | 1.468 |

- |ΔTOTEN| / N_atoms = 0 at OUTCAR precision. The largest per-iteration energy
  difference in OSZICAR is 4.2e-10 eV.
- Maximum force-component difference: 0 eV/Å. Both forces are zero by symmetry
  here, so this carries no information.
- `scripts/compare_runs.py` verdict: **VASP-verified**. All of its checks
  passed: normal exits, convergence, finite and complete forces, matched
  inputs, consistent counters, and no shadow failures.
- The safety envelope declined none of VASP's calls (`range` = 0). The worst
  shadow-check ratio equals v0.1's to the printed digits.
- No performance claim: the forced GPU path is slower here. Its time includes
  the shadow check and one cold first call of 0.44 s (Metal device creation and
  runtime kernel compilation inside the VASP process).

### First observed ZGEMM shapes (telemetry, Si2)

These are **observed** shapes from this run: dimension metadata only, without
values or addresses. They are not the synthetic `si2-bin` shapes of section 3.
- 802 calls fall into 35 distinct (trans, m, n, k) combinations, with 8 distinct
  leading-dimension triples. beta = 1 in 730 calls and beta = 0 in 72.
- Every call is CN or NN. m·n·k ranges from 1,024 to 149,760, and k is at most
  476. The decade counts match v0.1 (339 / 419 / 44).

| op | m×n×k | calls |
| --- | --- | --- |
| CN | 8×8×16 | 127 |
| NN | 16×8×8 | 88 |
| CN | 8×8×468 | 62 |
| CN | 8×8×476 | 51 |
| CN | 8×8×459 | 50 |
| NN | 468×8×8 | 37 |
| CN | 16×8×16 | 36 |
| CN | 24×8×16 | 35 |

Every observed shape is at least 100× below the work of the smallest synthetic
shape that the GPU won (about 1e7). Even the unshipped candidate auto rule
(m·n ≥ 16384) would select none of them. For this system, `auto` correctly
serves everything on the CPU.

### Commands used

```sh
python3 scripts/run_si2.py --binary private/relink-v0.2-20261102-125116/vasp_std_vmb \
  --fixture "$SI2_FIXTURE" --output private/si2-v0.2-off \
  --ledger private/launch-ledger-v0.2.txt --backend off --allow-warn-pressure
python3 scripts/run_si2.py --binary private/relink-v0.2-20261102-125116/vasp_std_vmb \
  --fixture "$SI2_FIXTURE" --output private/si2-v0.2-force \
  --ledger private/launch-ledger-v0.2.txt --backend force --check --telemetry \
  --allow-warn-pressure
python3 scripts/compare_runs.py private/si2-v0.2-off private/si2-v0.2-force
```

`--allow-warn-pressure` was used only because it was explicitly authorized for
this pair. A future run should leave it out unless it is authorized again.

## Tested ranges

- **Correctness (tests plus the benchmark's untimed checks):**
  - m and n from 1 to 6000, k from 1 to 4000;
  - all nine N/T/C pairs, in upper and lower case;
  - padded leading dimensions;
  - alpha and beta pairs, including beta = 0 with C filled with NaN;
  - operand magnitudes from 2^-521 to 2^600, i.e. products from 2^-1042 to
    2^1200. The extremes are exercised through the counted fallback.
- **Timing:** 17 synthetic shapes, m·n·k from 8.3e3 to 1.3e8, working sets up to
  31 MB, 7 repetitions each.

## Limitations

- All timing was exploratory, on a busy host, with one CPU thread as the
  baseline. A multi-rank VASP run would share one GPU across ranks; that
  contention was not measured.
- VASP verification covers one tiny system on one rank, with zero forces by
  symmetry. Real shapes are known only for that system; the benchmarked large
  shapes remain synthetic.
- The envelope is conservative. Operands with a very large dynamic range fall
  back even when a per-row or per-column scaling could have served them.
- The overflow bound covers the intermediates of a conventional ZGEMM
  evaluation only, not 3M/Gauss multiplication or other reassociations
  ([PRECISION.md](PRECISION.md)).
- The final safety-fix build (`afb272e`) has not been run in VASP.

## Next decision

1. Collect telemetry from a larger, representative system (off mode, CPU
   only) to learn whether any real ZGEMM shapes reach the region where the GPU
   won (m·n·k ≥ about 1e7). If none do, the GPU path has no use case for these
   workloads, whatever the kernel speed.
2. If some do: repeat the benchmark on a quiet host using those shapes, and ship
   an auto rule only if it reproduces, select and holdout alike.
3. Only after that: a tiled or k-split kernel, or batching several small calls
   into one submission. Batching would need explicit cross-call design and was
   out of scope here.
