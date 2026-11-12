# v0.4 results: real-workload shapes, GPU correctness and end-to-end timing

**Status:** v0.4 is a measurement release on one realistic system, anatase TiO2
(Ti4O8), run on an otherwise idle MacBook Pro. It covers two parallel series on that
system:
- **PBE:** a variable-cell relaxation, and a single point on the relaxed cell;
- **HSE06:** a hybrid single point on the same relaxed cell.

| Question | PBE | HSE06 |
| --- | --- | --- |
| Do real ZGEMM calls reach the GPU-win region (m·n ≥ 16384 and m·n·k ≥ 1.5e7)? | **No.** 0 calls at 4 or 8 ranks, NCORE 1 or 4 | **No.** 0 calls |
| Most any ZGEMM offload could save (all ZGEMM time) | **7.0–8.3%** of rank time | **0.008%** of rank time |
| Where the time goes | FFTW 38–41%, DGEMV 15–16%, MPI 11–16%, VASP's own code 12–20%, ZGEMM 7–9% | FFTW 38–41%, DGEMV 29–32%, VASP's own code 15–16%, DGEMM 7–8% |
| Are forced GPU results correct? | **Yes.** Relaxation and single point both VASP-verified, 0 shadow failures | Not run: no ZGEMM time to save |
| End-to-end, all ZGEMM forced onto the GPU | **2.60× slower** (relaxation, 3 pairs); 2.54× (single point, 1 pair) | Not run |

**Decisions.**
- The pre-registered Gate A (an upper bound of at least 5% inside the GPU-win region)
  failed in both series.
- `auto` stays empty, the default stays `off`, and no VASP speedup is claimed.
- The numerical policy is unchanged from v0.2 and v0.3.

## 1. Setup

**Host.** MacBook Pro M3 Max (10 P + 4 E cores, 30-core GPU, 36 GiB), macOS 27.2.
- No other computation ran. Desktop apps and a CMW watcher used a small amount of CPU.
- At each launch memory pressure was normal and the CPU mostly idle (66–83% idle for
  the PBE runs). The load average was still decaying from the previous run.
- Each run's `host-before` and `host-after` record shows the top processes.

**Toolchain.** GCC 16.1.0, Open MPI 5.0.9 (synthetic topology, `--bind-to none`),
OpenBLAS 0.3.34, ScaLAPACK 2.2.3, FFTW 3.3.11. This is the VASP port's tracked default,
read through its own `toolchain.py`.

**Binary.**
- `libvmb.a` was built from the committed v0.4 tree `71de2e3` (clean; source digest
  `aab2b2cd…`).
- A link-only relink of the 321 licensed std objects already on this MacBook gave
  `vasp_std_vmb` `3d5bbf16…`, with an empty link log. The stock `vasp_std` is unchanged.

**Route.**
- The published VASP port runner (`4fd6fa7`, private copy) uses the
  `--binary-manifest` route. Three reviewed manifests select the same executable:
  `off`, `force-check` (force with the shadow check) and `force` (no check).
- One thread per rank, NCORE and KPAR set explicitly, per-rank reports and per-rank
  shape telemetry.
- One VASP run at a time, a memory gate before each launch, no retries.

**Raw results.** The run directories stay in the private v0.4 work directory and are
never committed:
- per launch, under `runs/<label>/`: OUTCAR, OSZICAR, vasprun.xml, CONTCAR, EIGENVAL,
  WAVECAR, CHGCAR, `RUN_METADATA.txt` and `vmb-evidence.json`;
- next to it, `runs/<label>.census.json` and `runs/<label>.host-{before,after}.txt`;
- the profiles, in `runs/profiles/`.

## 2. Cases and jobs

All inputs are user-provided and byte-identical; the runner adds only NCORE and KPAR.
Common settings: 8×8×4 Γ k-mesh (45 irreducible k-points), ENCUT 600, PREC Accurate,
LREAL .FALSE.

| Series | Case | Settings | Launches (all exit 0, complete per-rank evidence) |
| --- | --- | --- | --- |
| PBE | relaxation (`tio2-pbe`) | ISIF = 3, 11 ionic steps | census A1–A3; profiled P1; correctness A6; timing pairs T1–T3 (off, force, force, off, off, force) |
| PBE | single point (`tio2-pbe-sp`) | relaxed cell; IBRION = −1, NSW = 0, ISTART = 0, ICHARG = 2 | census S1a–S1c; profiled S2; correctness S3; timing pair S4 |
| HSE06 | single point (`tio2-hse06-sp`) | same cell; LHFCALC, HFSCREEN = 0.2, AEXX = 0.25, PRECFOCK = Normal, LFOCKACE, ALGO = Normal, EDIFF = 1E-6 | H1 (census, CPU only) |

**Input edits**, each made with the user's approval because the runner's INCAR parser
refused the original:
- **PBE single point:** parentheses removed from the SYSTEM title (only a label).
- **HSE06:** an empty `NBANDS =` set to 56, the PBE value.

The refused attempts were preflight only (no VASP launched) and are kept apart from
the results.

**HSE06 specifics.**
- **Size:** NKDIM = 256 full-zone k-points for the Fock exchange, 56 bands.
- **Start:** H1 starts from a partly converged HSE06 WAVECAR (ISTART = 1, ICHARG = 0,
  through the runner's `--restart wavecar`). That WAVECAR came from an earlier launch
  with identical inputs, which started from the PBE single-point WAVECAR and was
  stopped after two electronic steps to lift its 12 h limit.
- **Limit:** H1 ran with a 7-day limit.

Runs made earlier on a host shared with another workload are superseded and not used.

## 3. VASP results

| | PBE relaxation (A2) | PBE single point (S1a) | HSE06 single point (H1) |
| --- | --- | --- | --- |
| Electronic steps | 11 ionic steps | 16 | 11 (final dE −5.4e-7 eV) |
| Total energy (TOTEN) | −107.29573806 eV | −107.30644851 eV | **−132.28535645 eV** |
| Lattice a, c | 3.80597, 9.71736 Å | 3.80550, 9.71912 Å | same cell as S1a |
| Band gap (from EIGENVAL) | 2.09 eV | 2.09 eV | **3.55 eV** |
| Largest force component | 0.007 eV/Å | 0.004 eV/Å | 0.23 eV/Å |
| External pressure | −1.04 kB | −0.81 kB | not computed |

- **The HSE06 numbers are as expected.** The gap opens from 2.09 eV with PBE to
  3.55 eV with HSE06, and the PBE-relaxed cell is no longer in equilibrium under HSE06
  (0.23 eV/Å).
- **HSE06 computed forces but not stress.** Its post-SCF exact-exchange forces (FORHF)
  took 6476 s.

## 4. Shape census (CPU only, backend off)

`scripts/vmb_shapes.py` merged the per-rank telemetry. **Every census reconciles
exactly** with the attempt evidence: the telemetry calls equal the evidence total, no
shapes were dropped, and the evidence is COMPLETE.

| Series | Launch | Ranks | NCORE | NBANDS | Elapsed s | ZGEMM calls | Distinct shapes | ZGEMM share of rank time | Calls in GPU-win region |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PBE relax | A1 | 4 | 1 | 52 | 353 | 2,515,192 | 887 | 7.6% | 0 |
| PBE relax | A2 | 8 | 1 | 56 | 227 | 4,891,290 | 575 | 7.6% | 0 |
| PBE relax | A3 | 8 | 4 | 52 | 198 | 4,970,624 | 2037 | 7.0% | 0 |
| PBE SP | S1a–S1c | 8 | 1 | 56 | 42.3–42.8 | 972,376 | 635 | 8.2–8.3% | 0 |
| HSE06 SP | H1 | 8 | 1 | 56 | 48,648 | 877,597 | 633 | **0.008%** | 0 |

- **Call sizes.** All calls have m·n·k below 1e7, and at 8 ranks almost all are below
  1e6. In A2, for example, there are 1.8 million calls in [1e3, 1e4), 1.0 million in
  [1e4, 1e5) and 2.0 million in [1e5, 1e6).
- **The same shapes carry the ZGEMM time in every case:** band blocks with n = 8, such as
  56×8×592 and 592×8×56 (56×8×574 during the relaxation) at 64–85 µs per call. The
  largest repeated shape is 256×56×56, at 185–188 µs. The GPU's fixed cost is about
  250 µs per call (v0.2), before any arithmetic.
- **HSE06 does not make ZGEMM heavier.**
  - H1 issues about 80,000 ZGEMM calls per electronic step (3.0 rank-seconds), against
    61,000 (1.8 rank-seconds) for the PBE single point.
  - But each HSE06 step takes about 1,700 times longer, so ZGEMM's share falls to noise.
  - The exact-exchange work does not go through ZGEMM.
- **Decomposition changes NBANDS** (56 at NCORE = 1 on 8 ranks, 52 otherwise). The
  relaxed cell then differs by up to about 1.7e-3 Å between decompositions, on the CPU
  alone. Off/on comparisons therefore always use the same ranks and NCORE.
- **Telemetry overhead.** 2 million off-mode calls cost 83–87 ns each without
  telemetry and 92–101 ns with it. That is about 0.01 s per rank, so it is negligible.

## 5. Where the time goes

**VASP's own timers** (share of LOOP+):

| | PBE relaxation (A2) | PBE single point (S1a) | HSE06 single point (H1) |
| --- | --- | --- | --- |
| EDDAV (Davidson) | 80.7% | 84.9% | 86.7% (11 × 3805–3902 s, mean 3832 s) |
| Other | STRESS 7.1%, CHARGE 5.3%, SETDIJ 1.7% | CHARGE 4.0%, STRESS 3.5%, SETDIJ 1.3% | FORHF (exact-exchange forces) 13.3% |
| Time per electronic step | — | 2.3 s | 3832 s |

**Profiles.** `scripts/vmb_profile.py` on macOS `sample` reports of two of eight
ranks, at 1 ms per sample, with idle waits excluded. The windows were:
- **P1:** 10 s.
- **S2:** 20 s, starting 15 s in.
- **HSE06:** 60 s, about 3 and 30 minutes into the first electronic step. These were
  taken during the earlier stopped launch (§2), whose inputs are identical to H1's.

| Work | PBE relaxation (P1) | PBE single point (S2) | HSE06, early | HSE06, late |
| --- | --- | --- | --- | --- |
| FFTW | 37.7 / 39.4% | 40.9 / 40.4% | 41.1 / 41.2% | 39.8 / 38.3% |
| BLAS DGEMV | 15.6 / 15.4% | 15.6 / 14.9% | 32.0 / 31.9% | 31.0 / 29.1% |
| VASP's own code | 19.9 / 17.5% | 12.5 / 11.7% | 15.8 / 15.6% | 15.9 / 14.7% |
| MPI and runtime | 11.3 / 12.3% | 15.2 / 16.2% | 0 / 0% | 1.8 / 6.9% |
| BLAS gemm | 6.7 / 7.7% (ZGEMM) | 8.9 / 9.2% | 7.6 / 7.6% (all DGEMM) | 7.7 / 7.1% (all DGEMM) |
| Other BLAS/LAPACK and system | 8.9 / 7.7% | — | 3.6 / 3.7% | 3.8 / 3.9% |

- **FFT is the largest piece of work a library-level backend could reach**, at 38–41%
  of busy time in both series.
  - Like ZGEMM, VASP issues it in small per-band transforms.
  - The wavefunction grid is 30×30×80 and the density grid 60×60×160.
- **DGEMV comes from different code in the two series:**
  - with PBE, the non-local projectors;
  - with HSE06, the real-space projector routines of the exchange (`rpromu_hf`,
    `racc0mu_hf`), which double its share to 29–32%.
- **HSE06 may plan its FFTs again.** `fftw_md5putc` is among the most-sampled
  functions, which suggests FFTW is planning again inside the exchange loop.

## 6. Correctness with every ZGEMM on the GPU (PBE)

A6 (relaxation) and S3 (single point) ran at 8 ranks with NCORE = 1 and
`force-check`. Every call ran on the GPU in float-float arithmetic, and each result
was also recomputed with OpenBLAS and checked against the v0.1 tolerance. Each was
compared with its CPU run (A2, S1a), using thresholds fixed before any run:

| Check | A6 vs A2 | S3 vs S1a | Threshold |
| --- | --- | --- | --- |
| Ionic and electronic steps | 11 ionic; identical (15, 10, 10, 5, 6, 6, 6, 7, 7, 7, 6) | 16 electronic, identical | equal |
| Final energy | 8.3e-10 eV/atom | 0 at printed precision | 1e-5 |
| Final forces | 0 at printed precision | 0 | 1e-3 eV/Å |
| Final stress | 2e-5 kB | 0 | 0.1 kB |
| Final lattice | 1.2e-7 Å | — (fixed cell) | 1e-4 Å |
| GPU calls submitted / consumed | 4,891,290 / 4,891,290, all 8 ranks | 972,376 / 972,376, all 8 ranks | — |
| Shadow-check failures; worst err/tol | 0; 0.072 | 0; 0.075 | 0; ≤ 1 |

**Verdict: VASP-verified, both.** This is the backend's first evidence on a
variable-cell, multi-k-point relaxation with non-zero forces and stress.

A6 took 829 s and S3 169 s, because the shadow check recomputes every call on the CPU.
They are correctness results, not timings. A6's hard limit was raised from 20 to
90 minutes for that reason.

**HSE06 was not run on the GPU.** ZGEMM is 0.008% of its rank time, so a GPU run could
not save anything. Its launch budget was spent on the CPU census.

## 7. End-to-end timing: off against force (PBE, no shadow check)

At 8 ranks, NCORE = 1. The relaxation pairs ran in the order off, force, force, off,
off, force.

| Pair | Off, s | Force, s | Numerical agreement (off vs force) |
| --- | --- | --- | --- |
| T1 (relaxation) | 215.8 | 561.0 | VASP-verified: ΔE 8.3e-10 eV/atom, ΔF 0, Δstress 1e-5 kB, Δlattice 5.8e-8 Å |
| T2 (relaxation) | 212.8 | 549.6 | VASP-verified: ΔE 0, ΔF 0, Δstress 2e-5 kB, Δlattice 7.1e-8 Å |
| T3 (relaxation) | 213.7 | 561.5 | VASP-verified: ΔE 8.3e-10 eV/atom, ΔF 0, Δstress 1e-5 kB, Δlattice 5.7e-8 Å |
| **Relaxation, mean ± s.d.** | **214.1 ± 1.5** | **557.3 ± 6.7** | **2.60× slower** |
| S4 (single point) | 44.4 | 112.6 | VASP-verified: Δstress 1e-5 kB; **2.54× slower** |

- **Relaxation.** Every force run is slower than every off run, and the other off runs
  agree (A2 227 s, P1 214 s).
- **Single point.** The four off runs (S1a–S1c, S4) give 43.0 ± 0.9 s. S4 is one pair,
  so the three relaxation pairs carry the main timing result.

**The cause is call granularity.**
- Forcing added 343 s per relaxation over about 611,000 calls per rank, and 68 s per
  single point over about 121,500. Both come to **about 0.56 ms per call**, against
  64–188 µs on a CPU core.
- That is more than the 0.25 ms single-process submit-to-completion floor measured in
  v0.2. It suggests that 8 processes contending for one GPU add latency.

## Limitations

- **Scope:** one system on one host. Open MPI synthetic topology, NCORE 1 or 4 (HSE06:
  1), KPAR 1, LREAL .FALSE.
- **Host:** desktop apps were running, though no other computation.
  - The PBE timing comparison is internally controlled (alternated, same binary), not a
    benchmark of the host.
  - In H1's first hour another process competed for CPU. Its later steps were at most
    2.5% faster than its first, so the HSE06 per-step figures stand as observed.
- **HSE06:** one CPU-only run, which started from a partly converged HSE06
  wavefunction (§2). Its profile was sampled during the earlier launch with identical
  inputs.
- **Profile:** statistical windows (10–60 s) on 2 of 8 ranks, not a full trace.
- **Other systems:** larger systems were not measured in v0.4.

## Next decision (v0.5)

ZGEMM offload is capped at about 8% of rank time with PBE, and at almost nothing with
HSE06. Forcing it costs 2.6×, so ZGEMM kernel work is not justified by this system.
Candidates for v0.5, each to be measured before any GPU work:
1. **FFT** (38–41% of busy time in both series).
   - Record FFTW plan sizes and transforms per call.
   - Check whether HSE06 plans FFTs again inside the exchange loop (`fftw_md5putc`).
   - A GPU FFT only pays off if many bands are transformed per call, against about
     250 µs of fixed cost per GPU call (0.56 ms with 8 ranks sharing the GPU).
2. **The HSE06 exchange DGEMV** (29–32%), the second-largest target for hybrids.
3. **Larger systems** (the Co3O4 case), to see whether any library-level work grows
   into the GPU's range.
