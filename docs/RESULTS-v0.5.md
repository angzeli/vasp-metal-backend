# v0.5 results: where the CPU time goes, and DGEMV/DGEMM on Accelerate

**Status:** v0.5 follows the v0.4 finding that no TiO2 ZGEMM call can be won on the
GPU. It times the two larger library targets, DGEMV/DGEMM and FFTW, call by call
on every rank. It then tests one CPU-side change end to end: routing DGEMV and
DGEMM to Apple Accelerate instead of OpenBLAS. Same system, host and toolchain as
v0.4 (anatase TiO2, MacBook Pro M3 Max).

| Question | PBE (single point, relaxation) | HSE06 (fixed work, 2 electronic steps) |
| --- | --- | --- |
| DGEMV + DGEMM share of rank time (call timers, all 8 ranks) | **17%** (DGEMM < 0.1%) | **48%** (DGEMV 40%, DGEMM 7.5%) |
| FFTW share: planning + execution | 3.5–3.7% + 35% | 6.1% + 25% |
| FFTW plans per execute | 1.00 (every transform is planned again) | 1.00 |
| End to end, DGEMV/DGEMM on Accelerate against OpenBLAS | not yet measured (pending, §8) | **1.76× faster** (pair 1); pair 2 pending |
| Results agree? | n/a | **Yes.** `CPU A/B consistent`: \|dTOTEN\|/N_atoms 7.5e-9 eV/atom, forces and band edges identical at printed precision |

**Decisions.**
- **The Accelerate route ships opt-in** (`VMB_DBLAS_LIBRARY=accelerate`, or a
  binary built with `DBLAS_ACCELERATE_DEFAULT=1`). The default stays OpenBLAS:
  Accelerate's results differ from OpenBLAS's in the last bits. The speedup was
  measured on one system, and with HSE06 only.
- **The FFTW plan cache ships opt-in and unmeasured on VASP.** Its upper bound is
  6.1% of HSE06 rank time and 3.7% of PBE rank time (§3). Measuring it is v0.6
  ([#8](https://github.com/angzeli/vasp-metal-backend/issues/8)).
- **GPU policy is unchanged:** ZGEMM `auto` stays empty and the default mode stays
  `off`.

## 1. Setup

**Host, toolchain, route:** as in v0.4 (MacBook Pro M3 Max, 10 P + 4 E cores, 36 GiB,
macOS 27.2). Toolchain: GCC 16.1.0, Open MPI 5.0.9, OpenBLAS 0.3.34, ScaLAPACK 2.2.3,
FFTW 3.3.11. The VASP port runner's reviewed `--binary-manifest` route, one VASP run
at a time, a memory gate before each launch, no retries. At each launch memory
pressure was normal; the CPU was 56–85% idle, the rest mostly the previous run's
load decaying.

**Binaries.** All are link-only relinks of the same 321 licensed VASP 6.6.1 std
objects, from clean committed trees.

| Family | Commit | Build defaults | Used for |
| --- | --- | --- | --- |
| v05b-off / -profile / -reuse | `7e64c38` | `PROFILE_DEFAULT` 0 / 1 / 2 | phase A: census, profiles, reuse probe |
| v05f-off / -accel | `2c050d4` | `DBLAS_ACCELERATE_DEFAULT` 0 / 1 | phases F and G: the BLAS A/B |

Within a family the binaries differ only in the build default. Every rank report
records the DGEMV/DGEMM library actually resolved, and each A/B launch checked it
(`openblas` or `accelerate`) before it was accepted.

**Raw results** stay in the private v0.5 work directory and are never committed.

## 2. Cases and jobs

| Case | Settings | Launches (all exit 0, COMPLETE per-rank evidence, 8 ranks, NCORE 1, KPAR 1) |
| --- | --- | --- |
| `tio2-pbe-sp` | v0.4 PBE single point | A1–A3 profile, alternated with A4–A6 off; A7 reuse probe; A9 profile with `sample` |
| `tio2-pbe` | v0.4 PBE relaxation, ISIF = 3 | A8 profile (11 ionic steps) |
| `tio2-hse06-fw` | v0.4 HSE06 single point with NELMIN = NELM = 2, restarted from the v0.4 HSE06 WAVECAR | H1 profile with `sample`, H2 off (phase A); H7 OpenBLAS, H8 Accelerate (phase F); H9 Accelerate, H10 OpenBLAS (phase G, pending) |
| `si2` | 1 rank, smoke test | S1 OpenBLAS, S2 Accelerate: same TOTEN, each declaring its library |

**Why fixed work.** A converged HSE06 single point took 13.5 h in v0.4. Capping it at
two electronic steps from the same converged WAVECAR gives the same work in each
run: the two SCF steps, then the exact-exchange forces. Runs are compared with
`compare_runs.py --fixed-work`, which requires exactly NELM steps in both.

Scripted phases B1 (an NSIM sweep) and E (plan-cache pairs) were not run.

## 3. Where the time goes (call timers, all ranks)

Every DGEMV, DGEMM and FFTW call is timed on every rank (v05b-profile, mean over 8
ranks).

| | PBE single point (A1) | PBE relaxation (A8) | HSE06 fixed work (H1) |
| --- | --- | --- | --- |
| Rank time | 45 s | 227 s | 14,416 s |
| DGEMV | 16.9% | 17.1% | **40.3%** (5,283–6,344 s across ranks) |
| DGEMM | < 0.1% | < 0.1% | 7.5% |
| FFTW execution | 34.8% | 34.7% | 24.9% |
| FFTW planning | 3.7% | 3.5% | 6.1% |
| DGEMV calls per rank | 7.7e5 | 4.3e6 | 8.1e8 |
| FFTW plans / executes per rank | 3.75e4 / 3.74e4 | 1.94e5 / 1.94e5 | 2.0e7 / 2.0e7 |

- **HSE06 is BLAS-bound.** 99.6% of its BLAS time is in 11 shapes. By call site:
  `dllmm_kernel` 66%, `rholm_kernel` 18%, and the exact-exchange projectors
  `rpromu_hf` 11% and `racc0mu_hf` 5%. The largest shapes are DGEMV 324×25 (38% of
  BLAS time) and 324×9 (28%), at 5–14 µs per call.
- **PBE's DGEMV is a different shape mix.** It is tall (about 4,700 × 8 to 18) and
  spread over many sites; no shape exceeds 3% of BLAS time.
- **FFTW plans every transform again**, in both series. One plan per execute confirms
  the v0.4 hint (`fftw_md5putc`), and sets the plan cache's ceiling at the planning
  share: 6.1% (HSE06) and 3.7% (PBE).

**The two HSE06 phases differ.** `sample` windows on H1 (60 s, 2 of 8 ranks):

| | SCF step (10 min) | Exact-exchange forces (150 min) |
| --- | --- | --- |
| DGEMV | 30.4% | **59.2%** |
| DGEMM | 7.7% | 8.4% |
| FFTW | 41.8% | 19.2% |
| VASP's own code | 15.8% | 10.8% |

v0.4 sampled only the SCF phase, hence its 29–32% DGEMV. Weighted by phase time, the
samples give 44% DGEMV over the run, against 40% from the call timers.

**Reuse probe (A7, PBE).** About 83% of projector DGEMV calls (`proj1`, `vnlac0`)
reuse the matrix of the previous call, in runs of 3–8 calls. Batching them into DGEMM
would need a change inside VASP, not at the library boundary. It is recorded as an
option, not pursued in v0.5.

## 4. Standalone replay: OpenBLAS against Accelerate

`bench/blas_replay` replays the H1 shapes with one thread, in ABBA order, on a quiet
host (by `run_bench.py`'s rule). Medians per call, in µs, for the five largest shapes:

| Shape (share of HSE06 BLAS time) | OpenBLAS warm / cold | Accelerate warm / cold | 8 processes at once*: OpenBLAS / Accelerate |
| --- | --- | --- | --- |
| DGEMV N 324×25 (38%) | 12.4 / 12.2 | 0.8 / 1.4 | 13.5 / 5.8 |
| DGEMV N 324×9 (28%) | 4.4 / 4.4 | 0.6 / 0.6 | 4.9 / 3.5 |
| DGEMV T 324×25 (11%) | 8.1 / 8.0 | 0.7 / 1.3 | 9.0 / 3.1 |
| DGEMV T 324×9 (8%) | 2.9 / 2.9 | 0.4 / 0.5 | 3.2 / 0.6 |
| DGEMM T 25×88×2269 (5%) | 298 / 301 | 57 / 62 | 341 / 137 |

\* Exploratory: the host was busy, and the series was stopped after 5 of its steps.

- **Correctness.** Every shape matched the long-double reference exactly with both
  libraries. Replay inputs are exactly representable, so this checks that each
  result is right, not how it rounds; VASP rounding is checked in §5.
- **Raw wall time misleads.** The replay's total wall time is 4× in Accelerate's
  favour, but it weights shapes by replay count. Weighted by VASP's time per shape,
  Accelerate's BLAS is 8.3× faster warm and 6.6× cold.
- **Under 8-way contention**, the gain falls to 2.1× on the same weighting: eight
  copies of pure BLAS share each cluster's matrix units, and OpenBLAS barely notices.

## 5. End to end: DGEMV/DGEMM on Accelerate (HSE06, fixed work)

| Run | Library | Order | SCF step 1 | SCF step 2 | Post-SCF (forces) | Total |
| --- | --- | --- | --- | --- | --- | --- |
| H2 (phase A) | OpenBLAS | — | — | — | — | 14,288 s |
| H7 | OpenBLAS | 1st | 3,830 s | 3,826 s | 6,456 s | 14,115 s |
| H8 | Accelerate | 2nd | 2,620 s | 2,611 s | 2,790 s | **8,024 s** |
| H9 | Accelerate | 1st | pending | | | |
| H10 | OpenBLAS | 2nd | pending | | | |

- **Pair 1: 1.76× faster end to end.** 1.46× per SCF step, and 2.31× on the post-SCF
  exact-exchange forces, the more DGEMV-bound phase (§3).
- **Correctness.** `compare_runs.py --fixed-work` classes H7/H8 `CPU A/B consistent`:
  - |dTOTEN|/N_atoms = 7.5e-9 eV/atom, against a 1e-5 threshold;
  - forces identical at printed precision;
  - VBM 0.8697 eV, CBM 4.4227 eV and gap 3.5530 eV in all four HSE06 runs (H1, H2,
    H7, H8). The gap matches v0.4's converged 3.55 eV.
- **The OpenBLAS runs differ among themselves by as much.** TOTEN across H1, H2 and
  H7 spans 6e-8 eV, and H8 differs from H7 by 9e-8 eV.

## 6. Cross-checks

| Check | Expected | Observed | Holds? |
| --- | --- | --- | --- |
| Replay predicts the end-to-end gain | 1.75× (warm), 1.70× (cold) | 1.77× (H8 against mean of H2, H7) | **Yes** |
| … per phase (Amdahl, 8.3× BLAS, sampled shares) | SCF 1.51×, forces 2.47× | 1.46×, 2.31× | Yes, slightly below |
| 8-process replay predicts it | 1.33× | 1.77× | No: a pessimistic bound (§4) |
| OpenBLAS baseline reproduces | | H2 14,288 s, H7 14,115 s (1.2% apart, different commits) | Yes |
| Profiling overhead | small | PBE: profile 45.3 s vs off 46.0 s (within spread); HSE06: H1 +0.9% over H2, +2.1% over H7 | Yes, ≤ 2% |
| Sampled and timed shares agree (HSE06) | | DGEMV 44% sampled (phase-weighted), 40% timed | Yes |
| Every A/B launch used the declared library | | Checked in each rank report before acceptance | Yes |

## 7. FFTW plan cache: not measured on VASP

The cache (`VMB_FFTW_PLAN_CACHE=1`, commit `dc394d6`) reuses repeated FFTW_ESTIMATE
3D plans. It passes its unit tests, but phase E (its VASP pairs) was not run.
- **What it could save:** since every transform is planned again (§3), at most the
  planning share: 6.1% of HSE06 rank time and 3.7% of PBE.
- **Shipping:** opt-in, default off, with no speedup claimed. Measuring it, and keeping
  or removing it, is v0.6
  ([#8](https://github.com/angzeli/vasp-metal-backend/issues/8)).

## 8. Pending

- **Phase G (H9, H10).** The same HSE06 pair in reverse order, Accelerate first, to
  rule out an order or drift effect. Running.
- **PBE A/B.** DGEMV is 17% of PBE rank time, so the gain is capped at 1.2×. The
  replay can't predict it, because it covers none of PBE's shapes. Three ABBA pairs
  on the PBE single point (about 45 s each) would settle it.

## Limitations

- **Scope:** one system on one host. 8 ranks, NCORE 1, KPAR 1, LREAL .FALSE.
- **HSE06 fixed work, not a converged run.** The speedup is per unit of work at a
  fixed point, from one pair so far (a second is running). A converged HSE06 run on
  Accelerate was not made.
- **Accelerate is a different BLAS.** Its results differ from OpenBLAS's in the last
  bits. They stay within the run-to-run spread of OpenBLAS here, but other systems
  need their own check.
- **Profiles:** call timers on all ranks; `sample` windows of 20–60 s on 2 of 8 ranks.

## Next decision (v0.6)

1. **FFTW plan cache** ([#8](https://github.com/angzeli/vasp-metal-backend/issues/8)):
   measure it, with a ceiling of 6.1% (HSE06) and 3.7% (PBE).
2. **Accelerate route:** the PBE A/B and a converged HSE06 run. If both hold, consider
   making Accelerate the default for DGEMV/DGEMM.
3. **FFTW execution** (25–35% of rank time) is now the largest library target.
4. **Larger systems** (the Co3O4 case), where shapes and shares may differ.
