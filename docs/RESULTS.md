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

## VASP integration: UNVERIFIED

No VASP launch was made. The gate in `scripts/run_si2.py` requires normal
kernel memory pressure (`kern.memorystatus_vm_pressure_level == 1`) and at least
20% free memory. On 2 November 2026, three readings returned pressure level 2
(WARN) with 44 to 47% free, about 12 of 13 GB of swap in use, and a large
compressed-memory pool. The gate was not relaxed. Another long-running job
was using 8 CPU cores and was left untouched.

Next local commands, once `sysctl -n kern.memorystatus_vm_pressure_level`
prints `1` (no relink or rebuild is needed):

```sh
python3 scripts/run_si2.py --binary private/relink-<stamp>/vasp_std_vmb \
  --fixture "$SI2_FIXTURE" --output private/si2-off \
  --ledger private/launch-ledger.txt --backend off
python3 scripts/run_si2.py --binary private/relink-<stamp>/vasp_std_vmb \
  --fixture "$SI2_FIXTURE" --output private/si2-on \
  --ledger private/launch-ledger.txt --backend on --check
python3 scripts/compare_runs.py private/si2-off private/si2-on
```

`$SI2_FIXTURE` is a directory holding the licensed 2-atom Si static PBE inputs
(ENCUT 300, 2×2×2 Γ mesh, EDIFF 1e-6). The comparison reports both energies,
SCF iteration counts and convergence, GPU calls consumed, the shadow-check
ratio, |ΔTOTEN|/N_atoms and the maximum force-component difference.
