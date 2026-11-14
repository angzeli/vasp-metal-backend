# v0.6 results: baselines, time to solution, and FFTW

**Status: in progress.** The decision rules below were written on 13 Nov 2026, before
the weekend VASP runs (phases B–D). Sections marked *pending* fill in as those
runs finish. Same system, host and toolchain as v0.5 (anatase TiO2, MacBook Pro
M3 Max). v0.6 is CPU-only: no GPU work is planned (v0.4 and v0.5 found no
library-level work the GPU can win).

| Question | Answer so far |
| --- | --- |
| Can VASP be linked directly against Accelerate? (#9) | **Yes, with two fixes.** `zdotc_`/`cdotc_` crash (f2c convention) and need a shim; and with Open MPI's flat namespace, Accelerate must load before ScaLAPACK, or VASP silently runs OpenBLAS (§2) |
| Direct link against the interposer route (#9) | *Pending (phase C)* |
| Converged HSE06, time to solution (#10) | **1.51× faster on the Accelerate route** (14.5 h → 9.6 h). Both converge in 12 SCF steps to the same energy, forces and band edges: `CPU A/B consistent`. Per the decision rule, Accelerate becomes the default for DGEMV/DGEMM (§3) |
| FFTW plan cache (#8) | Works (294,576 hits, 16 misses per PBE run, results bit-identical); PBE timing inconclusive on a busy host (§4). HSE06 *pending (phase B)* |
| FFTW built with NEON | Homebrew's FFTW has no SIMD on Apple Silicon. A NEON build executes VASP's transforms **1.29× faster** (HSE06 shapes), 1.26× (PBE), with the same accuracy (§5). End to end *pending (phase D)* |

**Scope, frozen on 14 Nov 2026** (during phase B, before any phase B–D pair had
finished):

- **Headline results: four decisions.** The converged HSE06 pair (T1/T2), the plan
  cache on HSE06 (K1–K4), direct Accelerate against the interposer route on HSE06
  (M1–M4), and NEON FFTW on HSE06 (G1–G4). The PBE pairs back these up.
- **Supporting data only.** The quiet PBE plan-cache rerun (Q1–Q6) and the
  8-process BLAS replay run because they were already queued, but no decision
  depends on them.
- **No new experiments in v0.6,** whatever the results suggest. In particular, a
  converged HSE06 run on the direct binary, if direct linking wins, moves to v0.7.
  So do ScaLAPACK built on Accelerate and a second system.

## 1. Decision rules (fixed before the timed runs)

- **Direct Accelerate (D) against the interposer route (I),** HSE06 fixed work, D I I D:
  - D faster by more than 3% in both pairs: recommend direct linking; the
    interposer stays for evidence, profiling and the plan cache.
  - Within 3%: equivalent; the interposer route stays the recommended one, as it
    carries per-rank evidence of the library used.
  - D slower: the interposer route is recommended; record why.
- **Converged HSE06 (T1 OpenBLAS, T2 Accelerate route):** both must converge and be
  `CPU A/B consistent`. If their SCF step counts differ by more than one, report
  time per step beside time to solution. If this holds, Accelerate becomes the
  default for DGEMV/DGEMM.
- **FFTW plan cache:** a gain of at least 1% in both HSE06 ABBA pairs, with results
  unchanged: keep it and turn it on by default. Below 1%: remove the cache code.
- **NEON FFTW:** at least 3% end to end in both orders, `CPU A/B consistent`:
  document the NEON build as the recommended FFTW.
- All pairs: `compare_runs.py` must class them `CPU A/B consistent`
  (|dTOTEN|/N_atoms ≤ 1e-5 eV/atom); one VASP run at a time, memory gate, no retries.

## 2. Linking VASP directly against Accelerate (#9): ABI and link order

**ABI.** `vasp_std` imports 72 BLAS/LAPACK symbols; Accelerate exports all but
`dlasrt2_` and `iceil_`, which come from ScaLAPACK. The value-returning ones were
called from gfortran exactly as VASP declares them (`tests/test_accelerate_abi.f90`,
in `make test`):

| Functions | OpenBLAS | Accelerate (legacy interface) | Accelerate + shim |
| --- | --- | --- | --- |
| `ddot dasum dnrm2 dzasum zlange idamax izamax ilaenv` | ok | ok | ok |
| `zdotc cdotc` | ok | **segfault** | ok |

Accelerate's legacy Fortran BLAS returns complex results through a hidden first
argument (the f2c convention); gfortran expects them in registers.
`scripts/accelerate-shim.c` defines both through `cblas_*dotc_sub`. The
`ACCELERATE_NEW_LAPACK` symbols cannot be reached by relinking existing objects, so
the legacy interface is what a relinked VASP gets.

**Link order.** Open MPI's `mpifort` links with `-flat_namespace`, so each BLAS
reference resolves at run time to the first loaded image that defines it.
Homebrew's ScaLAPACK loads OpenBLAS itself. With `-framework Accelerate` on the
link line in the usual place, the gfortran driver moves it after ScaLAPACK, and a
sampled rank of the PBE single point spent all its BLAS time in OpenBLAS (155
samples in `libopenblasp-r0.3.34`, none in Accelerate). Passed as
`-Wl,-framework,Accelerate` ahead of the objects, Accelerate loads first, and the
same sample shows 656 frames in Accelerate's `libBLAS` and none in OpenBLAS.
`scripts/relink-vasp.sh --direct-accelerate` builds it this way and records the
load order.

**Result agreement (PBE single point).** The direct binary gives TOTEN
−107.30644851 eV, identical to the interposer route, forces identical, stress
within 1e-5 kB: `CPU A/B consistent` (`compare_runs.py --no-backend on`).

*Timing: pending (phase C, P1–P6 and M1–M4).*

## 3. Converged HSE06, time to solution

TiO2 HSE06 single point run to convergence (EDIFF 1e-6, NELM 100), with the inputs
of v0.4 H1/H1b, restarted from the v0.4 PBE single point's WAVECAR. 8 ranks. One
pair, run back to back on 13–14 Nov: T1 on OpenBLAS (`v06a-off`), then T2 on the
Accelerate route (`v06a-accel`, DGEMV/DGEMM to Accelerate). The two binaries are
built from v0.5.0 and differ only in `DBLAS_ACCELERATE_DEFAULT`.

| | T1, OpenBLAS | T2, Accelerate route | Speedup |
| --- | --- | --- | --- |
| **Time to solution (LOOP+)** | **52,056 s (14.5 h)** | **34,433 s (9.6 h)** | **1.51×** |
| SCF steps to convergence | 12 | 12 | |
| Per SCF step (mean; range) | 3,806 s (3,801–3,825) | 2,636 s (2,634–2,640) | 1.44× |
| SCF loop, total | 45,668 s | 31,638 s | 1.44× |
| After SCF (exact-exchange forces) | 6,388 s | 2,796 s | 2.28× |
| Final TOTEN | −132.28535600 eV | −132.28535598 eV | |
| VBM / CBM / gap | 0.8696 / 4.4226 / 3.5530 eV | 0.8696 / 4.4226 / 3.5530 eV | |

- **Results agree.** `compare_runs.py` classes the pair `CPU A/B consistent`:
  - |dTOTEN|/N_atoms = 1.7e-9 eV/atom;
  - forces identical (max |dF| = 0);
  - both runs show 927,780 BLAS calls, complete evidence on all 8 ranks, and the
    library recorded per rank.

  The energies agree to about 1e-8 eV at every SCF step, so the two runs follow the
  same SCF path.
- **Why 1.51× and not v0.5's 1.76×.** v0.5's runs stopped after 2 SCF steps, so the
  forces, where Accelerate gains most (2.3×), were a larger share of the time. A
  converged run is 88% SCF steps on OpenBLAS, where the gain is 1.44×. The parts
  agree with v0.5: 1.46× per step and 2.31–2.32× on the forces there.
- **Against v0.4.** The same calculation on OpenBLAS took 13 SCF steps over two runs
  (H1, H1b), about 15.7 h. T1 converged in one run, in 12 steps.
- **Host.** Every SCF step lies within 0.7% of its run's mean, so neither run was
  disturbed. The 1-minute load of 8.9 recorded at T2's launch is T1's 8 ranks, which
  had just exited, still in the average.
- **Decision.** Both runs converge, the step counts are equal and the pair is
  `CPU A/B consistent`, so the rule in §1 applies: **Accelerate becomes the default
  for DGEMV/DGEMM.** The default changes in the code at release, together with the
  other v0.6 decisions.
- **Limits.** One pair, not ABBA (four converged runs would take about 48 h). Run
  order is covered by the fixed-work pairs (v0.5, and K1–K4 here). Stress is not in
  the output of this case, so it was not compared.

## 4. FFTW plan cache (#8)

**PBE single point, phase A** (binaries from v0.5.0, `8cfb4a4`, differing only in
`PLAN_CACHE_DEFAULT`; Accelerate route):

| Pair | Off | Cache | Ratio |
| --- | --- | --- | --- |
| C1 / C2 | 42.76 s | 41.40 s | 1.033× |
| C4 / C3 | 41.53 s | 42.81 s | 0.970× |
| C5 / C6 | 38.26 s | 35.65 s | 1.073× |

- **The cache works as designed.** 294,576 hits and 16 misses per run over 8 ranks,
  no bypasses; TOTEN, forces and stress bit-identical in every pair.
- **The timing is inconclusive.** The ratios disagree in sign. The host was busy
  (1-minute load 5–17, Spotlight indexing and other applications at 50–120% CPU).
  The ceiling here is the 3.7% planning share. A quiet rerun is in phase C (Q1–Q6).

*HSE06 fixed work: pending (phase B, K1–K4).*

## 5. FFTW built with NEON

**Homebrew's FFTW has no SIMD on Apple Silicon.** Its formula passes SIMD flags only
on Intel, and its `libfftw3` holds none of FFTW's vector codelets. FFTW 3.3.11 was
built with `--enable-neon` (otherwise as Homebrew: clang, `--enable-threads
--enable-armv8-cntvct-el0`) into a private prefix: 332 vector codelets. The
source tarball matched Homebrew's pinned SHA-256.

**Standalone replay** (`bench/fftw_replay`, one thread, ABBA, busy-host
exploratory): each shape planned with `FFTW_ESTIMATE`, executed once and destroyed,
as VASP does, on the shapes and weights of the v0.5 profiles.

| Shapes (v0.5 profile) | Execute, Homebrew → NEON | Weighted speedup | Planning NEON / Homebrew | Max error against a direct DFT |
| --- | --- | --- | --- | --- |
| HSE06 (H1): 24×24×60 c2c, 99.9% of FFT time | 193 → 149 µs | **1.29×** | 1.05 | 7.5e-16 (both libraries) |
| PBE (A1): 26 shapes | — | **1.26×** | 1.29 | 8.7e-16 |

- The two libraries agree to ≤ 4.3e-16 on every shape, including the real-to-complex
  ones, which the direct DFT does not cover.
- **What it predicts end to end:** FFT execution is about 45% of an Accelerate-route
  HSE06 run and 35% of PBE, so about 1.11× and 1.07×.
- **First VASP pair (smoke test):** PBE 33.8 s (NEON) against 38.9 s (Homebrew),
  `CPU A/B consistent`. One pair; ABBA pending (phase D).

## Limitations

- One system, one host, 8 ranks, NCORE 1, KPAR 1.
- The direct-Accelerate baseline still runs ScaLAPACK's own BLAS calls on OpenBLAS
  (Homebrew's ScaLAPACK links it); a ScaLAPACK built on Accelerate was not tried.
- Run directories sit under the iCloud-synced Desktop (as in v0.5).
