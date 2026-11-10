# v0.4 results: real-workload shapes and the auto decision (paused)

**Status: paused.** v0.4 is a measurement release, scoped to the TiO2 case. It asks
whether that case's ZGEMM calls reach the region where the Metal backend can win.
- **Census (CPU only):** done.
- **Pre-registered Gate A** (the upper bound on any gain must reach 5% to continue):
  **failed**. No TiO2 call reaches the candidate region.
- **Deferred:** the force-check correctness run (A6) and the off/force timing pairs.
  The MacBook was shared with another workload, so a fair comparison wasn't possible.
  A6 was stopped by SIGTERM with incomplete evidence, and no timing pair was launched.
  Both will be rerun when the host is otherwise idle.
- **Unchanged:** `auto` stays empty, no VASP speedup is claimed, and the numerical
  policy is the same as in v0.2 and v0.3.

| Question | Result |
| --- | --- |
| Do TiO2's real ZGEMM calls reach the GPU-win region (m·n ≥ 16384 and m·n·k ≥ 1.5e7)? | No: 0 calls at 4 or 8 ranks, NCORE 1 or 4 |
| Most any ZGEMM offload could save | 4.3–7.4% of rank time (all ZGEMM time) |
| Correctness of forced GPU results on this relaxation | deferred (A6) |
| End-to-end off versus force timing | deferred |

## 1. Setup

**Host.** MacBook Pro M3 Max (10 P + 4 E cores, 30-core GPU, 36 GiB), macOS 27.2.
- Another user workload, an ESP benchmark using about 4 cores, ran throughout.
- Every timing here is therefore *exploratory*: it is used only for shares within a run,
  never for a speed claim.

**Toolchain.** GCC 16.1.0, Open MPI 5.0.9 (synthetic topology, `--bind-to none`),
OpenBLAS 0.3.34, ScaLAPACK 2.2.3, FFTW 3.3.11. This is the port's tracked default,
read through its own `toolchain.py`.

**Binary.**
- `libvmb.a` was built from the committed v0.4 tree `71de2e3` (clean; source digest
  `aab2b2cd…`).
- Link-only relink of the 321 licensed std objects already on this MacBook gave
  `vasp_std_vmb` `3d5bbf16…`, with an empty link log. The stock `vasp_std` is unchanged.

**Route.**
- The published VASP port runner (`4fd6fa7`, private copy) uses the
  `--binary-manifest` route with manifests `vmb-v040-off`, `vmb-v040-force-check` and `vmb-v040-force`.
- One thread per rank, NCORE and KPAR set explicitly, per-rank reports and per-rank
  shape telemetry.
- No retries. Census launches A1–A3 completed; A6 was started and then stopped (above).

**Case** (user-provided, inputs byte-identical; the runner added only NCORE and KPAR):

| Case | System | k-mesh | Settings |
| --- | --- | --- | --- |
| TiO2 | Ti4O8 | 8×8×4 Γ | PBE, ENCUT 600, PREC Accurate, LREAL .FALSE., ISIF=3 relaxation |

## 2. Shape census (CPU only, backend off)

`scripts/vmb_shapes.py` merged the per-rank telemetry. **Every census reconciles
exactly** with the attempt evidence: the telemetry calls equal the evidence total, no
shapes were dropped, and the evidence is COMPLETE.

| Launch | Case | Ranks | NCORE | NBANDS | Ionic steps | Elapsed s | ZGEMM calls | ZGEMM share of rank time | Calls in candidate region | Bound for candidate region |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | TiO2 | 4 | 1 | 52 | 11 (converged) | 397 | 2,515,192 | 7.4% | 0 | 0 |
| A2 | TiO2 | 8 | 1 | 56 | 11 (converged) | 471 | 4,891,290 | 4.9% | 0 | 0 |
| A3 | TiO2 | 8 | 4 | 52 | 11 (converged) | 468 | 4,970,624 | 4.3% | 0 | 0 |

"Bound" is the region's ZGEMM time divided by total rank time: no rule selecting that
region can save more.

**Shapes.**
- **TiO2:** the calls are many and small. At 8 ranks the largest shapes are about
  8e5 work. The time is concentrated in band-block shapes such as 56×8×574 and
  574×8×56, at 40–100 µs per call on the CPU against a GPU floor of about 250 µs per
  call (v0.2).

**Decomposition matters.**
- Effective NBANDS follows the band-parallel layout: 56 at 8 ranks with NCORE=1, 52 at
  NCORE=4 or at 4 ranks.
- Between those decompositions, the CPU-only relaxations end with lattice vectors up to
  1.75e-3 Å apart.
- Off/on comparisons therefore always use the same ranks and NCORE.

**Telemetry overhead** (it replaces the planned telemetry-off run A4, which the manifest
route cannot express):
- 2 million off-mode calls cost 83–87 ns each without telemetry and 92–101 ns with it.
- That is at most about 0.01 s of A2's 471 s per rank. The overhead is negligible.

## 3. Where the time goes beyond ZGEMM

VASP's own per-phase timers put 87.5% of A2's time in EDDAV (Davidson), 4.4% in STRESS
and 3.3% in CHARGE. ZGEMM accounts for 4.9% of rank time, so most of EDDAV is other
work: FFTs, projector application and VASP's own code.

A per-library breakdown for TiO2 (`scripts/vmb_profile.py` on macOS `sample` reports)
is deferred to the rerun on an idle host. It will show how FFT and BLAS-2 shares compare
before any v0.5 FFT work.

## 4. Deferred: correctness (A6) and off/force timing

**A6** is TiO2 at 8 ranks with `vmb-v040-force-check`, compared with A2 using the
relaxation thresholds fixed in advance:
- the same number of ionic steps;
- |ΔE_final|/N ≤ 1e-5 eV/atom;
- max |ΔF| ≤ 1e-3 eV/Å;
- max |Δstress| ≤ 0.1 kB;
- max |Δlattice| ≤ 1e-4 Å.

A6 started, but was stopped by SIGTERM (exit 143) before any ionic step completed. Its
evidence is correctly INCOMPLETE: killed ranks publish no report.

**Off/force timing** is 3 alternating pairs (off, force, force, off, off, force), using
`vmb-v040-off` and `vmb-v040-force` (no shadow check). It was not launched.

**To resume on an idle host:**
1. Rerun A2 (the off baseline).
2. Run P1, the profiled off run.
3. Run A6.
4. Run the six timing launches. Use `private/v04-runs/tio2-timing.zsh` and
   `launch.sh`; each gates on memory and on no other VASP running.

## Limitations

- **Scope:** one system (TiO2), one host, Open MPI synthetic topology, NCORE 1 or 4,
  KPAR 1.
- **Host contention:** another workload was present throughout, so all timing so far is
  exploratory.
- **Not yet run:** the correctness and timing runs, and a per-library profile.

## Next decision

1. **Finish v0.4 on an idle host:** A6, the off/force timing pairs and the TiO2 profile.
2. **v0.5 candidate:** FFT. Measure FFTW call sizes and the FFT share first. A GPU FFT
   pays off only if many bands are transformed per call.
