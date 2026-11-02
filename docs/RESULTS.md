# v0.1 results

Host: Apple M3 Max (30-core GPU, Metal 4), 36 GiB, 14 physical CPU cores.
Toolchain: Apple clang 21, Homebrew GCC/gfortran 16.1.0, OpenMPI 5.0.9,
OpenBLAS 0.3.34.

## Verified

| Item | Evidence |
| --- | --- |
| Metal device and kernel | `Apple M3 Max`; kernel compiled at runtime in safe math mode |
| GPU accuracy | 100 C cases and 11 Fortran calls within the fixed tolerance; worst err/tol 0.109 |
| Dispatch accounting | 143 submitted = 141 consumed + 1 injected GPU error + 1 injected non-finite result |
| Fallback | 10 routes, each bit-identical to OpenBLAS and taken exactly once |
| Original CPU path | bound to the loaded `libopenblas` image, never to Accelerate or to the backend itself |
| Private relink | 321 VASP std objects linked with no warnings; all 7195 reference global symbols present; additions are only the 16 backend symbols; `zgemm_` defined in the executable |

## VASP integration: verified (correctness and dispatch only)

Two sequential launches on 2 November 2026 used the same privately relinked
binary and identical 2-atom Si static PBE inputs (ENCUT 300, 2×2×2 Γ mesh,
EDIFF 1e-6): 1 MPI rank (synthetic-topology OpenMPI), one OpenMP/BLAS thread,
NCORE = KPAR = 1, 60 s hard timeout, no retries.

| | Backend off | Backend on (`VMB_CHECK=1`) |
| --- | --- | --- |
| Exit / termination | 0 / normal | 0 / normal |
| SCF | converged (EDIFF reached) | converged (EDIFF reached) |
| Davidson iterations | 12 | 12 |
| Final TOTEN, eV | −8.28298469 | −8.28298469 |
| ZGEMM calls | 802 | 802 |
| GPU submitted / completed / consumed | 0 / 0 / 0 | 802 / 802 / 802 |
| Calls served by OpenBLAS | 802 (disabled) | 0 |
| Shadow-check worst err/tol | — | 4.39e-2 over 802 checks, 0 failures |

- |ΔTOTEN| / N_atoms = 0 at OUTCAR precision (1e-8 eV); the largest
  per-iteration energy difference in OSZICAR is 4.2e-10 eV. Small differences in
  the residual columns show that the GPU results fed the SCF.
- Maximum force-component difference: 0 eV/Å. In this ideal diamond structure
  both forces are exactly zero by symmetry, so the force comparison carries no
  information here.
- ZGEMM sizes (m·n·k): 339 calls in [1e3, 1e4), 419 in [1e4, 1e5), 44 in
  [1e5, 1e6).
- Verdict from `scripts/compare_runs.py`: **VASP-verified** (both converged,
  GPU results consumed, |ΔTOTEN|/N_atoms ≤ 1e-5 eV/atom).

Memory pressure was WARN (level 2, 39 to 43% free) at every check. Both launches
used the explicit, user-authorized `--allow-warn-pressure` override; the default
gate still requires normal pressure, and critical pressure or < 20% free memory
is always refused. A separate 8-rank job ran throughout and was only observed.

## Not claimed: performance

This is a correctness and dispatch result, not a speedup. VASP's elapsed time
was 0.244 s with the backend off and 1.102 s with it on. The on-run also
includes the shadow check, and each GPU call carries roughly 0.3 to 1 ms of
packing and synchronization overhead, far more than the matrix work at these
sizes. No timing study was performed.

## Reproduce

```sh
python3 scripts/run_si2.py --binary private/relink-<stamp>/vasp_std_vmb \
  --fixture "$SI2_FIXTURE" --output private/si2-off \
  --ledger private/launch-ledger.txt --backend off
python3 scripts/run_si2.py --binary private/relink-<stamp>/vasp_std_vmb \
  --fixture "$SI2_FIXTURE" --output private/si2-on \
  --ledger private/launch-ledger.txt --backend on --check
python3 scripts/compare_runs.py private/si2-off private/si2-on
```

`$SI2_FIXTURE` holds your licensed Si2 inputs. Add `--allow-warn-pressure` only
as a deliberate override.

## Limitations

- One tiny system, one rank, one k-point set; forces are zero by symmetry.
- Only ZGEMM calls from VASP's own objects are offloaded; calls inside
  ScaLAPACK and OpenBLAS stay on the CPU.
- Float-float arithmetic is not IEEE FP64; see `PRECISION.md`.
