# v0.4 results: real-workload shapes, GPU correctness and end-to-end timing

**Status:** v0.4 is a measurement release on one realistic system, TiO2 PBE, run on an
otherwise idle MacBook Pro: a variable-cell relaxation, and a single point on the
relaxed cell (§6).

| Question | Result |
| --- | --- |
| Do real ZGEMM calls reach the GPU-win region (m·n ≥ 16384 and m·n·k ≥ 1.5e7)? | **No.** 0 calls at 4 or 8 ranks, NCORE 1 or 4. |
| Most any ZGEMM offload could save | **7.0–7.6%** of rank time (all ZGEMM time) |
| Are forced GPU results correct through a full relaxation? | **Yes.** 4,891,290 of 4,891,290 calls on the GPU, 0 shadow failures, identical ionic path (VASP-verified) |
| End-to-end, all ZGEMM forced onto the GPU | **2.60× slower:** 557.3 ± 6.7 s against 214.1 ± 1.5 s on the CPU, every force run slower than every off run |
| Where the time goes | FFTW 38–39%, VASP's own code 18–20%, BLAS-2 DGEMV 15–16%, MPI 11–12%, ZGEMM 7–8% |
| Single point on the relaxed cell (§6) | Same picture: ZGEMM 8.2%, 0 calls in the region, FFTW 40–41%; forced GPU VASP-verified and 2.54× slower |

**Decisions.**
- The pre-registered Gate A (an upper bound of at least 5% inside the GPU-win region) failed.
- `auto` stays empty, the default stays `off`, and no VASP speedup is claimed.
- The numerical policy is unchanged from v0.2 and v0.3.

## 1. Setup

**Host.** MacBook Pro M3 Max (10 P + 4 E cores, 30-core GPU, 36 GiB), macOS 27.2.
- No other computation ran. Desktop apps and a CMW watcher used a small amount of CPU.
- At each launch the CPU was 66–83% idle and memory pressure was normal. The load
  average was still decaying from the previous run.
- Each run's `host-before` and `host-after` record shows the top processes.

**Toolchain.** GCC 16.1.0, Open MPI 5.0.9 (synthetic topology, `--bind-to none`),
OpenBLAS 0.3.34, ScaLAPACK 2.2.3, FFTW 3.3.11. This is the VASP port's tracked default,
read through its own `toolchain.py`.

**Binary.**
- `libvmb.a` was built from the committed v0.4 tree `71de2e3` (clean; source digest
  `aab2b2cd…`).
- Link-only relink of the 321 licensed std objects already on this MacBook gave
  `vasp_std_vmb` `3d5bbf16…`, with an empty link log. The stock `vasp_std` is unchanged.

**Route.**
- The published VASP port runner (`4fd6fa7`, private copy) uses the
  `--binary-manifest` route. Three reviewed manifests select the same executable:
  `off`, `force-check` (force with the shadow check) and `force` (no check).
- One thread per rank, NCORE and KPAR set explicitly, per-rank reports and per-rank
  shape telemetry.
- One VASP run at a time, a memory gate before each launch, no retries.

**Case** (user-provided, inputs byte-identical; the runner adds only NCORE and KPAR):

| System | k-mesh | Settings |
| --- | --- | --- |
| Ti4O8 | 8×8×4 Γ | PBE, ENCUT 600, PREC Accurate, LREAL .FALSE., ISIF=3 relaxation (11 ionic steps) |

**Jobs.** 11 launches, all exit 0 with complete per-rank evidence:
- the census: A1–A3;
- the profiled run: P1;
- correctness: A6;
- timing: three alternating off/force pairs, T1–T3 (off, force, force, off, off, force).

Runs made earlier on a host shared with another workload are superseded by these and
are not used.

## 2. Shape census (CPU only, backend off)

`scripts/vmb_shapes.py` merged the per-rank telemetry. **Every census reconciles
exactly** with the attempt evidence: the telemetry calls equal the evidence total, no
shapes were dropped, and the evidence is COMPLETE.

| Launch | Ranks | NCORE | NBANDS | Elapsed s | ZGEMM calls | Distinct shapes | ZGEMM share of rank time | Calls in GPU-win region |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | 4 | 1 | 52 | 353 | 2,515,192 | 887 | 7.6% | 0 |
| A2 | 8 | 1 | 56 | 227 | 4,891,290 | 575 | 7.6% | 0 |
| A3 | 8 | 4 | 52 | 198 | 4,970,624 | 2037 | 7.0% | 0 |

- **Call sizes.** All calls have m·n·k below 1e7. At 8 ranks almost all are below 1e6:
  1.8 million in [1e3, 1e4), 1.0 million in [1e4, 1e5), 2.0 million in [1e5, 1e6).
- **Where the ZGEMM time is.** It is concentrated in band blocks with n = 8, for
  example 56×8×574 and 574×8×56 at 64–85 µs per call. The largest repeated shape is
  256×56×56, at 188 µs. The GPU's fixed cost is about 250 µs per call (v0.2), before
  any arithmetic.
- **Decomposition changes NBANDS** (56 at NCORE=1 on 8 ranks, 52 otherwise). The
  relaxed cell then differs by up to about 1.7e-3 Å between decompositions, on the CPU
  alone. Off/on comparisons therefore always use the same ranks and NCORE.
- **Telemetry overhead.** 2 million off-mode calls cost 83–87 ns each without
  telemetry and 92–101 ns with it. That is about 0.01 s per rank here, so it is
  negligible.

## 3. Where the time goes

VASP's own timers, for A2: EDDAV (Davidson) 80.7%, STRESS 7.1%, CHARGE 5.3%,
SETDIJ 1.7%.

`scripts/vmb_profile.py` on macOS `sample` reports of P1 (two of eight ranks, 10 s at
1 ms each, idle waits excluded):

| Work | Rank 1 | Rank 5 |
| --- | --- | --- |
| FFTW | 37.7% | 39.4% |
| VASP's own code | 19.9% | 17.5% |
| BLAS DGEMV (non-local projectors) | 15.6% | 15.4% |
| MPI and runtime | 11.3% | 12.3% |
| BLAS ZGEMM | 6.7% | 7.7% |
| Other BLAS/LAPACK and system | 8.9% | 7.7% |

**FFT is the largest piece of work a library-level backend could reach**, about five
times ZGEMM's share. Like ZGEMM, VASP issues it in small per-band transforms. The
wavefunction grid is 30×30×80 and the density grid 60×60×160.

## 4. Correctness with every ZGEMM on the GPU (A6)

A6 is TiO2 at 8 ranks with NCORE=1 and `force-check`. Every call ran on the GPU in
float-float arithmetic, and each result was also recomputed with OpenBLAS and checked
against the v0.1 tolerance. It was compared with A2 using thresholds fixed before any
run:

| Check | A6 vs A2 | Threshold |
| --- | --- | --- |
| Ionic steps; electronic steps per ionic step | 11; identical (15, 10, 10, 5, 6, 6, 6, 7, 7, 7, 6) | equal |
| Final energy | 8.3e-10 eV/atom | 1e-5 |
| Final forces | 0 at printed precision | 1e-3 eV/Å |
| Final stress | 2e-5 kB | 0.1 kB |
| Final lattice | 1.2e-7 Å | 1e-4 Å |
| GPU calls submitted / consumed | 4,891,290 / 4,891,290, on all 8 ranks | — |
| Shadow-check failures; worst err/tol | 0; 0.072 | 0; ≤ 1 |

**Verdict: VASP-verified.** This is the backend's first evidence on a variable-cell,
multi-k-point relaxation with non-zero forces and stress.

A6 took 829 s, because the shadow check recomputes every call on the CPU. It is a
correctness result, not a timing. Its hard limit was raised from 20 to 90 minutes for
that reason.

## 5. End-to-end timing: off against force (no shadow check)

Three alternating pairs at 8 ranks, NCORE=1, run in the order off, force, force, off,
off, force:

| Pair | Off, s | Force, s | Numerical agreement (off vs force) |
| --- | --- | --- | --- |
| T1 | 215.8 | 561.0 | VASP-verified: ΔE 8.3e-10 eV/atom, ΔF 0, Δstress 1e-5 kB, Δlattice 5.8e-8 Å |
| T2 | 212.8 | 549.6 | VASP-verified: ΔE 0, ΔF 0, Δstress 2e-5 kB, Δlattice 7.1e-8 Å |
| T3 | 213.7 | 561.5 | VASP-verified: ΔE 8.3e-10 eV/atom, ΔF 0, Δstress 1e-5 kB, Δlattice 5.7e-8 Å |
| **Mean ± s.d.** | **214.1 ± 1.5** | **557.3 ± 6.7** | |

**Forcing ZGEMM onto the GPU makes the whole run 2.60× slower.** Every force run is
slower than every off run, and the two other off runs agree (A2 227 s, P1 214 s).

The cause is call granularity. Forcing added 343 s per run over about 611,000 calls per
rank, which is **about 0.56 ms per call** against 64–188 µs on a CPU core. That is more
than the 0.25 ms single-process submit-to-completion floor measured in v0.2, which
suggests that 8 processes contending for one GPU add latency.

## 6. Single point on the relaxed cell

**Why.** A relaxation repeats SCF cycles from a good starting point. A single point
from scratch weights the first, harder steps more. This checks whether that changes the
rate-limiting step.

**Case** (user-provided `tio2-pbe-sp`): the relaxed Ti4O8 cell, with the same KPOINTS,
POTCAR, ENCUT, PREC and LREAL as §1.
- **INCAR settings:** IBRION = −1, NSW = 0, ISTART = 0, ICHARG = 2, LVHAR = .TRUE.
- **One title edit:** the runner's INCAR parser refuses parentheses outside comments,
  so they were removed from the SYSTEM title, with the user's approval. SYSTEM is only
  a label.
- **The four refused attempts** (preflight only, no VASP launched) are kept apart from
  the results.

**Jobs.** 7 launches, all at 8 ranks, NCORE=1, KPAR=1, one at a time. All exited 0
with complete per-rank evidence:
- three census repeats, S1a–S1c;
- S2, profiled;
- S3, force-check;
- S4, an off and force pair.

NBANDS is 56, with 45 irreducible k-points. Every run converged in 16 electronic steps
to E = −107.30644851 eV.

| Measure | Result |
| --- | --- |
| Elapsed, off (S1a, S1b, S1c, S4-off) | 42.8, 42.6, 42.3, 44.4 s → **43.0 ± 0.9 s** |
| VASP timers (S1a) | EDDAV 84.9%, CHARGE 4.0%, STRESS 3.5%, SETDIJ 1.3%; about 1.4 s of setup |
| ZGEMM calls; distinct shapes | 972,376; 635, identical in all three census runs |
| ZGEMM share of rank time | 8.2%, 8.3%, 8.3% |
| Calls in GPU-win region | **0**; m·n·k < 1e6 for every call. Gate A fails. |
| Top shapes | 56×8×592 and 592×8×56 (n = 8 band blocks, 46% of ZGEMM time); 256×56×56 at 185 µs |
| Profile, S2 (2 ranks, 20 s from 15 s in) | FFTW 40.9 / 40.4%, DGEMV 15.6 / 14.9%, MPI 15.2 / 16.2%, VASP's own code 12.5 / 11.7%, BLAS gemm 8.9 / 9.2% |

The census reconciles exactly with the evidence in all three runs.

**Correctness, S3 against S1a.**
- 972,376 of 972,376 calls ran on the GPU, on all 8 ranks.
- Shadow check: 0 failures, worst err/tol 0.075.
- ΔE, ΔF and Δstress are all 0 at printed precision. **Verdict: VASP-verified.**
- 169.0 s, a correctness result rather than a timing.

**Timing, S4.**
- Off 44.4 s against force 112.6 s, so **2.54× slower**. The run is VASP-verified
  (Δstress 1e-5 kB).
- The added 68 s over about 121,500 calls per rank is **about 0.56 ms per call**, the
  same as in the relaxation.
- This is one pair. The three alternating pairs in §5 carry the main timing result.

**Conclusion.** The rate-limiting work is the same as in the relaxation.
- Davidson dominates. FFTW is the largest library-level piece, at about 40% of busy
  time.
- DGEMV and MPI follow, at about 15% each.
- ZGEMM is a fixed, small share (about 8%) of thin calls.
- On this system the relaxation's ionic steps do not change which work limits the run.

## Limitations

- **Scope:** one system on one host. The single point (§6) uses the same system and
  settings. Open MPI synthetic topology, NCORE 1 or 4, KPAR 1,
  LREAL .FALSE.
- **Host:** desktop apps were running, though no other computation. The timing
  comparison is internally controlled (alternated, same binary), not a benchmark of the
  host.
- **Profile:** statistical 10 s windows on 2 of 8 ranks, not a full trace.
- **Other systems:** larger systems and hybrid functionals were not measured in v0.4.

## Next decision (v0.5)

ZGEMM offload is capped at about 7.6% for this workload, and forcing it costs 2.6×.
ZGEMM kernel work is not justified by this case. Candidates for v0.5, each to be
measured before any GPU work:
1. **FFT.** Record FFTW plan sizes and transforms per call (where it is 38–39% of busy
   time here). A GPU FFT only pays off if many bands are transformed per call, against
   about 250 µs of fixed cost per GPU call.
2. **Larger systems** (the Co3O4 case) **and hybrid functionals,** where calls are
   heavier, to see whether any library-level work grows into the GPU's range.
