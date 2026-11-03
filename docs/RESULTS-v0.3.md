# v0.3 results: managed 16-rank execution and per-rank evidence

**Status:** v0.3 is an execution and evidence release. The numerical policy is
unchanged from v0.2. No speedup claim is made, and nothing here assesses TiO2
accuracy.

| Claim | Status | Scope |
| --- | --- | --- |
| Standalone numerical gates (v0.2 range/overflow and fallback included) | **passed** on the MacBook (M3 Max) and on the Mac mini (M5 Pro GPU) | 138 C cases, 12 Fortran ABI calls |
| Per-rank report route and strict aggregation | **passed** | 13 report cases, 20 run-comparison cases, harmless 16-rank Open MPI fixtures |
| Pinned binary selection (VASP port) and CMW integration | **passed** (offline) | port: 9 manifest cases plus the existing suites (45 unit and 14 cross-repository managed tests); CMW: 11 new cases and 136 in total over the touched groups |
| 16-rank VASP qualification with the relinked binary | see section 4 | Si2 fixture only |
| VASP performance acceleration | **not established** | — |

## 1. Hardware, toolchain and identities

- **Execution host:** physical Apple M5 Pro Mac mini (18 cores, 64 GiB), macOS 27.2.
- **Toolchain:** Open MPI 5.0.11 (PRRTE 4.1.0_1), GCC 16.2.0, OpenBLAS 0.3.34,
  ScaLAPACK 2.2.3, FFTW 3.3.11. The toolchain was read from the VASP port's
  convention (`private/toolchain.env`), not from an edited script.
- **Backend source:** base commit `79659208` plus the uncommitted v0.3 working
  tree. The source digest is `85621b76…`, identical on the MacBook and in the
  mini build.
- **Library and relinked executable:**
  - `libvmb.a` SHA-256 `f1760116…`.
  - Relinked executable `vasp_std_vmb` SHA-256 `71fb2547…`, link-only: 321
    licensed objects already on the mini and an empty link log. The stock
    `vasp_std` (`4d147f16…`) is unchanged.
- **The only build warning** is the deployment-target notice for Homebrew's
  OpenBLAS (built for macOS 27, linked with minimum 26.0).

## 2. Offline validation (MacBook)

**Backend.**
- `make -j1 test`: 138 numerical cases, 12 ABI calls and 13 per-rank cases.
- `tests/test_compare_runs.py`: 20 cases.

The per-rank cases cover:
- 16 concurrent processes, including zero-call ranks;
- a rank that leaves without a report, a killed rank, and a duplicate rank;
- 15 kinds of malformed or contradictory rank and environment input;
- symbolic-link, writable, relative and absent directories;
- legacy single-process reports;
- 20 counter, identity and time tamperings, shadow failure, a non-finite ratio,
  maximum-not-sum aggregation, truncation, interrupted publication, stale
  reports, and mixed attempts.

**Real bug found by the tests.** A rank that never called ZGEMM reported mode
`off` while the other ranks reported `force`. The configuration is now read when
the rank is claimed, without initialising Metal.

## 3. Mac mini standalone and harmless 16-rank gates

**Standalone gates.** All passed on the M5 Pro GPU:
- `test_zgemm`: 138 cases; both v0.2 overflow regressions return `range`
  bit-identically to OpenBLAS, and the positive controls stay on the GPU;
- `test_fortran_abi`: 12 calls;
- `test_rank_reports.py`: 13 cases;
- `test_compare_runs.py`: 20 cases.

**Harmless fixtures.** These used `rank_probe` (no VASP), real Open MPI 5.0.11
`mpirun -np 16` through the port's supervised launcher, one thread per rank:

| Fixture | Exit | Claims / reports | Evidence | Consumed ranks | Shadow |
| --- | --- | --- | --- | --- | --- |
| Reporting (ranks 0, 5, 9 call ZGEMM) | 0 | 16 / 16 | COMPLETE, qualified | 3 | PASSED |
| All 16 ranks call ZGEMM (concurrent Metal) | 0 | 16 / 16 | COMPLETE, qualified | 16 | PASSED |
| 5 s timeout while ranks sleep | 124 | 16 / 0 | INCOMPLETE (16 missing) | 0 | — |
| Cancellation (SIGTERM to the launcher) | 143 | 16 / 0 | INCOMPLETE (16 missing) | 0 | — |

- **Cleanup:** no probe process remained after any fixture, and an independent
  sentinel process survived.
- **Rank variables:** Open MPI passed them, and the backend variables, to every
  rank unchanged.
- **First attempt:** a harness error (`python -I` hid the launcher's sibling
  module) failed before any launch. Its directory is kept as
  `fixtures-attempt1-harness-error`.

## 4. Si2 16-rank qualification (relinked binary, backend off vs force)

**Setup.**
- **Launches:** two, through a dedicated CMW queue (`vmb-v03-qual-20261102`),
  separate from TiO2. Each was gated on normal memory pressure and no other VASP
  process, and logged in a ledger capped at two. Each had a 120 s limit and no
  retries.
- **Delay:** the mini was first occupied by another session's 16-rank research
  calculation. Launches waited until it was free, and nothing was interrupted.
- **Binary and route:** both runs used the same relinked executable
  (`71fb2547…`), selected only through the reviewed manifests
  `vmb-v030-off` and `vmb-v030-force-check`. They ran as managed-foreground CMW
  payloads: 16 Open MPI ranks, one thread per rank, NCORE=4, KPAR=1.
- **Inputs:** the retained port Si2 PBE fixture, copied byte for byte. The four
  staged digests are identical in both runs, and the effective INCAR is
  identical (`e14f6adb…`). Only NCORE and KPAR were added.
- **Attempt binding:** each run's reports are bound to its CMW attempt
  (`origin=cmw`).

| | Backend off (J3.1) | Force, `VMB_CHECK=1` (J4.1) |
| --- | --- | --- |
| Exit / footer / convergence | 0 / normal / EDIFF reached | 0 / normal / EDIFF reached |
| Electronic steps; effective NBANDS; NELECT | 12; 8; 8 | 12; 8; 8 |
| TOTEN, eV | −8.28298469 | −8.28298469 |
| Rank reports (evidence) | 16/16 COMPLETE | 16/16 COMPLETE |
| ZGEMM calls / consumed on GPU / CPU | 10120 / 0 / 10120 | 10120 / **10120** / 0 |
| Ranks with consumed GPU work | 0 | **16** |
| Range fallbacks; shadow checks; failures | 0; 0; — | 0; 10120; **0** (max err/tol 0.110) |
| GPU qualified (aggregator) | no (CPU_ONLY) | **yes** |
| Runner elapsed, s (not a speed comparison) | 0.556 | 1.191 |

**Comparison verdict.** `scripts/compare_runs.py` reports **VASP-verified**:
- |ΔTOTEN|/N = 0 at printed precision, inside the fixture's 1e-5 eV/atom
  criterion;
- maximum force and stress differences of 0 at printed precision (the forces
  are zero by symmetry);
- inputs matched by digest, and identical effective NBANDS.

**CMW result binding.** The runner-record association in CMW accepted both real
attempts. It checked the attempt ID, the metadata block order and the summary
invariants.

**Not assessed.** Ionic relaxation, variable-cell behaviour, multi-k-point
scaling and any TiO2 property. The forced run includes the full shadow check, so
its time is not GPU performance.

## 5. TiO2 MPI-16 GPU case: registered on Hold, not dispatched

**Before registering.**
- The benchmark queue had no active attempt (J1.1 and J2.1 were Done, both exit 0).
- Its idle controller was stopped.
- The same queue (`2b975d09…`) was enrolled under the new config root
  `node-v03`, with the v0.3 runtime (`b669e67b…`). The previous config roots
  and runtimes are untouched.

**Revalidation with the actual new runner and binary.**
- The four input files are byte-identical to the plan, and `ISIF=3` is
  preserved.
- The intended output is absent.
- The static check is COMPATIBLE, and the dry run shows `force` with the check
  on and NCORE=1 / KPAR=1.

**Registration.** One atomic held registration created **J3.1** (attempt
`6f4019e4…`): 16 ranks × 1 thread, 86400 s, write scope
`/Users/liangze/Desktop/vasp-test/outputs`. The mini (`jobs show`) and the
MacBook portal (v0.3 observer configuration) showed the same queue, attempt,
Hold status, ranks, binary, output path and backend configuration.

**Before dispatch.** Release the job only with a v0.3-capable controller.
Older workers do not export `CMW_JOBS_ATTEMPT_ID`, and the runner refuses that
launch.

## Limitations

- **Si2 only.** The only VASP evidence is from the 2-atom Si2 fixture. It has
  zero forces by symmetry and says nothing about TiO2 accuracy, variable-cell
  relaxation or speed.
- **Open MPI only.** The per-rank route is scoped to Open MPI launches on one host.
- **Manifest provenance.** The manifests pin a dirty source tree, by digest,
  because nothing was committed.
