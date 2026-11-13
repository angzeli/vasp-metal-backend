# Changelog

## 0.4.0 (merged, untagged)

A measurement release: whether realistic VASP workloads produce ZGEMM calls the
GPU can win. Numerical policy, modes and the evidence contract are unchanged;
`auto` still has no regions. Results: [docs/RESULTS-v0.4.md](docs/RESULTS-v0.4.md).

### Added

- **Per-rank timing.** Rank reports gain optional `timing` (seconds in GPU
  attempts and in OpenBLAS) and `work_decades` (calls per m·n·k decade). The
  aggregator validates and sums them, and still accepts v0.3 reports. The compact
  summary is unchanged.
- **`scripts/vmb_shapes.py`.** A shape census of a managed run from per-rank
  telemetry. It reconciles the calls with the attempt evidence, bins work by decade,
  bounds the share of rank time any region rule could save, and exports observed
  shapes for replay.
- **`scripts/vmb_profile.py`.** Attributes the busy time of a rank sampled by macOS
  `sample` to BLAS kernels, FFTW, MPI and VASP's own code.
- **`run_bench.py --concurrency N`.** N `shape_replay` replicas, each with one CPU
  thread, start together behind a barrier (`shape_replay --start-at`), modelling N
  ranks sharing one GPU.
- **Ionic relaxations in `compare_runs.py`.** Electronic convergence per ionic step,
  the required-accuracy stop, step counts, and final energy, forces, stress and
  lattice, against thresholds fixed before any run.

### Findings

Anatase TiO2 on an idle MacBook Pro M3 Max, in two parallel series. Gate A failed in
both, so `auto` stays empty.
- **PBE** (a variable-cell relaxation at 4–8 ranks, and a single point on the relaxed
  cell):
  - **Census:** ZGEMM is 7.0–8.3% of rank time, and no call reaches the GPU-win
    region.
  - **Correctness:** with every ZGEMM forced onto the GPU and the shadow check on, the
    relaxation (4.9 million calls) and the single point (1.0 million) both match their
    CPU runs, with 0 failures (VASP-verified).
  - **Timing:** forcing ZGEMM onto the GPU without the check is 2.60× slower end to
    end on the relaxation (557 ± 7 s against 214 ± 2 s) and 2.54× on the single point.
    That comes from about 0.56 ms of per-call GPU latency.
  - **Profile:** FFTW is the largest library-reachable work (38–41% of busy time).
- **HSE06** (a hybrid single point on the same cell, CPU only, converged in 13.5 h):
  - **Census:** ZGEMM falls to 0.008% of rank time, with 0 calls in the GPU-win
    region, so no GPU run was made.
  - **Profile:** FFTW (38–41%) and the exchange DGEMV (29–32%) dominate.

## 0.3.0 (merged, untagged)

Managed MPI execution and per-rank evidence. The architecture (a link-time
`zgemm_` interposer) and the numerical policy are unchanged from 0.2, including
the `afb272e` range and overflow envelope. The contract is in
[docs/CONTRACT-v0.3.md](docs/CONTRACT-v0.3.md).

### Added

- **Per-rank reports** (`VMB_REPORT_DIR`, `VMB_ATTEMPT_ID`, optional
  `VMB_EXPECT_WORLD_SIZE`).
  - Rank and size come from the Open MPI launcher environment; no MPI library is
    linked.
  - A library constructor claims the rank, and each rank publishes a versioned
    JSON report (`vmb.rank-report.v1`) at exit. Ranks that never call ZGEMM still
    publish a zero-call report.
  - Files are created with `O_EXCL|O_NOFOLLOW` through a held directory
    descriptor and published by hard link, so nothing is overwritten or
    redirected.
  - A malformed or contradictory configuration exits with status 78 before the
    program starts.
  - Metal is never initialised for reporting.
- **Per-rank telemetry** (`VMB_TELEMETRY=rank`).
- **`scripts/vmb_evidence.py`**: strict attempt aggregation
  (`vmb.attempt-evidence.v1`).
  - It checks every per-rank invariant before summing counters, and takes the
    maximum (not the sum) of the check ratios.
  - It rejects missing, duplicate, stale, mixed-attempt and mixed-binary
    evidence.
  - It reports evidence completeness, CPU-only or GPU-consumed execution, the
    shadow-check state and GPU qualification separately.
- **`scripts/make-binary-manifest.py`**: writes the reviewed
  `vasp-port.binary-manifest.v1`. The VASP port's managed runner uses it to
  select a relinked executable explicitly.
- **`tests/rank_probe.c` and `tests/test_rank_reports.py`**: 16 concurrent ranks,
  missing, killed and duplicate ranks, malformed rank information, unsafe
  directories, and invariant tampering.

### Changed

- **`scripts/relink-vasp.sh --port-root PORT_ROOT`** reads the toolchain from the
  port's own convention (`config/toolchain-defaults.txt` plus the host-local
  `private/toolchain.env`) and records `relink.txt`. This replaces the one-off,
  per-host edited copy of the script. The two-argument form keeps its old
  defaults.
- **Makefile.**
  - `make PORT_ROOT=…` selects the same toolchain.
  - Builds embed the source commit, a source-tree digest and a dirty flag.
- **`scripts/compare_runs.py`** also reads runs made by the port runner
  (`RUN_METADATA.txt`, `vmb-reports/`). It aggregates them with the same model,
  and reports effective NBANDS and stress differences.
- **Fail-closed reporting.** A literal `VMB_REPORT` or `VMB_TELEMETRY` path is
  refused when the world size is greater than 1, because every rank would
  overwrite it.
- **Text report.** `vmb.version=0.3`.

### Unchanged

- **Modes.** `VMB_MODE=off` is the default; `force` obeys every safety check and
  falls back exactly once; `auto` still has no shape regions.
- **Single-process compatibility.** The v0.1/v0.2 report formats are still
  parsed, and the 138 numerical cases and 12 Fortran ABI calls still pass.

## 0.2 (`7965920`, merged in PR #1)

- **Range and overflow safety.** Exponent-range envelope, bounded overflow of
  unscaled intermediates (`afb272e`) and fail-closed shadow checking.
- **Modes and telemetry.** `off`, `auto` and `force` modes, and aggregated shape
  telemetry.
- **Benchmark.** A shape-replay benchmark with one measured optimization.
- **Verification.** Si2 integration was verified on the `0344fdb` build.

## 0.1 (`d3be811`)

- First opt-in Metal ZGEMM backend, with a single-rank Si2 integration.
