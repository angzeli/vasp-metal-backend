# v0.3 contract: managed MPI execution and per-rank evidence

v0.3 keeps the v0.2 architecture and numerical policy unchanged:
- `libvmb.a` is still a link-time `zgemm_` interposer;
- the exponent-range envelope, the fail-closed shadow check and the exactly-once
  OpenBLAS fallback are unchanged (see [PRECISION.md](PRECISION.md));
- `VMB_MODE=off` is still the default;
- `force` is a validation mode that still obeys every safety check;
- `auto` still has no shape regions, so it serves every call on the CPU.

v0.3 adds three things, each with its own contract below:
- **A.** per-rank reports that 16 MPI ranks can write without colliding;
- **B.** strict aggregation of those reports into one attempt verdict;
- **C.** an explicit, pinned way for the VASP port's managed runner to select a
  relinked backend binary.

Kernel work, precision schemes, tuning, GPU scheduling and new scientific modes
are out of scope.

## A. Per-rank report route (`libvmb`)

### Environment

| Variable | Meaning |
| --- | --- |
| `VMB_REPORT_DIR` | Absolute path of an existing directory: a real directory (no symbolic link anywhere in the path), owned by the effective user, not group- or world-writable. Enables the route. |
| `VMB_ATTEMPT_ID` | Required with `VMB_REPORT_DIR`. 1–64 characters from `[A-Za-z0-9._-]`, starting with a letter or digit. |
| `VMB_EXPECT_WORLD_SIZE` | Optional. If set, it must equal the world size the launcher reports. |
| `VMB_TELEMETRY` | With `VMB_REPORT_DIR`, it must be unset or exactly `rank` (per-rank telemetry in the same directory). |
| `VMB_REPORT` | Must be unset when `VMB_REPORT_DIR` is set. |

**Rank source.** v0.3 supports Open MPI only:
- rank and size come from `OMPI_COMM_WORLD_RANK` and `OMPI_COMM_WORLD_SIZE`;
- if `PMIX_RANK` is also set, it must equal the Open MPI rank;
- the local rank and local size are recorded when present.

Each value must be a canonical decimal (no sign, no leading zero), with
`0 <= rank < size <= 4096`. If neither Open MPI variable is set, the process is a
singleton (rank 0, size 1). If only one is set, the configuration is
contradictory. No MPI library is linked or called.

**Configuration errors exit before the program runs.** These include:
- a malformed or contradictory rank;
- an unsafe directory, or a missing or invalid attempt ID;
- conflicting variables;
- a legacy literal `VMB_REPORT` or `VMB_TELEMETRY` path when the world size is
  greater than 1, since every rank would overwrite the same file;
- a rank claim that already exists.

Each is checked in a library constructor. The process prints `vmb: report
configuration error: …` to stderr and exits with status 78 before the program's
`main`. Metal is never initialised for reporting.

### Files

Every file is created inside the directory through a held directory descriptor,
with `O_CREAT|O_EXCL|O_NOFOLLOW` and mode 0600. Nothing is ever overwritten.

| File | When | Meaning |
| --- | --- | --- |
| `rank-RRRRR.claim.json` | constructor | This process owns rank R for this attempt (pid, start time). A second claim for the same rank fails, and that process exits 78. |
| `rank-RRRRR.telemetry.tsv` | exit, only if `VMB_TELEMETRY=rank` | Aggregated shapes, written to a temporary file and then hard-linked into place. |
| `rank-RRRRR.report.json` | normal exit (`atexit`) | Complete final report: written to `.rank-RRRRR.report.PID.tmp`, `fsync`ed, then hard-linked to the final name (which fails if that name exists). The directory is `fsync`ed afterwards. |

`RRRRR` is the zero-padded rank, five digits.
- **No report means missing evidence.** A rank that is killed, or that leaves
  through `_exit`, keeps its claim but never publishes a report. That counts as
  missing evidence, never as a zero-call rank.
- **Only the claiming process reports.** A forked child does not publish: the
  `atexit` handler checks that it is running in the process that made the claim.
- **Zero-call ranks still report.** A rank that never calls ZGEMM publishes a
  valid report with zero counters.

### Report schema `vmb.rank-report.v1`

```json
{"schema": "vmb.rank-report.v1", "complete": true,
 "backend": {"name": "vasp-metal-backend", "version": "0.3.0",
             "source_commit": "<git HEAD at build>", "source_digest": "<sha256 of source tree>",
             "source_dirty": true},
 "attempt_id": "…", "rank": 7, "world_size": 16, "local_rank": 7, "local_size": 16,
 "rank_source": "openmpi",
 "process": {"pid": 1234, "ppid": 1200, "hostname": "…", "executable": "/abs/path",
             "executable_stat": [dev, inode, bytes, mtime_ns]},
 "times": {"start_unix": 0.0, "end_unix": 0.0},
 "mode": "force", "check": true,
 "device": {"initialized": true, "name": "Apple M5 Pro"},
 "cpu_library": "/…/libopenblas.0.dylib",
 "counters": {"calls": 0, "gpu_submitted": 0, "gpu_completed": 0, "consumed": 0,
              "cpu_calls": 0, "checked": 0},
 "statuses": {"ok": 0, "disabled": 0, "…": 0},
 "max_check_ratio": 0.04, "max_check_ratio_state": "finite",
 "telemetry": {"file": "rank-00007.telemetry.tsv", "dropped_shapes": 0}}
```

- **Ratio fields.** `max_check_ratio_state` is one of the following:
  - `none`: no checks were made, and the ratio is `null`;
  - `finite`;
  - `nonfinite`: the ratio is `null`, and the evidence fails closed.
- **What is never recorded:** operand values and memory addresses.
- **Unset values:** `cpu_library` is `null` until the first call resolves it, and
  `telemetry` is `null` when telemetry is off.
- **Single-process use is unchanged.** Without `VMB_REPORT_DIR`, the v0.2
  `VMB_REPORT` text report and `VMB_TELEMETRY` path still work.

## B. Attempt aggregation `vmb.attempt-evidence.v1`

`scripts/vmb_evidence.py` (standard library only) is the single acceptance model
for v0.3. `scripts/compare_runs.py` uses it for multi-rank runs. The VASP port
runs the pinned copy named in the binary manifest.

```
python3 -I -B scripts/vmb_evidence.py aggregate --dir REPORT_DIR --attempt-id ID \
    --world-size N [--mode M --check 0|1] [--executable PATH] [--not-before UNIX] \
    [--output NEW_FILE]
```

**Per-rank checks.** These run before any aggregation:
- schema, completeness and types;
- the rank is in range, the file name matches the rank, and the world size and
  attempt ID agree;
- the claim's pid equals the report's pid;
- the start time is not before `--not-before`, which rejects stale reports;
- for every counter: it is a non-negative integer, and

      calls = consumed + cpu_calls
      gpu_submitted >= gpu_completed >= consumed >= 0
      sum(statuses) = calls,   statuses.ok = consumed
      gpu_submitted - gpu_completed = statuses.gpu_error
      gpu_completed - consumed = statuses.gpu_nonfinite + statuses.check_failed
      check on:  checked = consumed + statuses.check_failed
      check off: checked = 0

- the ratio state is consistent with `checked`, and finite ratios are
  non-negative.

**Cross-rank checks.** All ranks must share:
- the backend version, source commit and digest;
- the mode and check setting, which must also equal `--mode`/`--check` if given;
- the hostname and executable identity, which must also equal `--executable` if
  given.

**Evidence.**
- **COMPLETE:** exactly one valid report for each rank 0…N-1, and no stray,
  temporary, duplicate or conflicting files.
- **INCOMPLETE:** some ranks have no report (killed, or never started), and
  nothing is invalid.
- **INVALID:** anything malformed, duplicated, stale, mixed-attempt,
  mixed-binary or inconsistent.

**Aggregation.** Counters and statuses are summed. `max_check_ratio` is the
maximum over the ranks, never a sum.

**Separate verdicts.** Each is reported on its own:

| Field | Values |
| --- | --- |
| `evidence` | COMPLETE / INCOMPLETE / INVALID |
| `execution` | `GPU_CONSUMED` (total consumed > 0) / `CPU_ONLY` (all calls on the CPU) |
| `shadow_check` | `PASSED` / `FAILED` / `NOT_ENABLED` / `NO_CHECKS` |
| `gpu_qualified` | `true` only if evidence is COMPLETE, total consumed > 0, and the shadow check is `PASSED` or `NOT_ENABLED` (only enabled checks are required to pass) |

GPU submission or completion alone never implies consumption, and no verdict
implies acceleration. Operational completion, scientific convergence and
numerical agreement with a reference are judged elsewhere: the runner, CMW
result review, and `compare_runs.py`.

## C. Pinned binary selection (VASP port runner)

```
run-vasp.sh --input … --output NEW --binary std --binary-manifest NAME --ranks 16 \
    --ncore N --kpar K --mpi-mode native --restart none --timeout S --stop-before 0 \
    --managed-foreground
```

**Manifest location.** `NAME` is a plain basename, `[A-Za-z0-9][A-Za-z0-9._-]{0,63}`.
The manifest is read only from `PORT_ROOT/private/binary-manifests/NAME.json`,
which must be:
- a regular file, not a symbolic link;
- owned by the user and not group- or world-writable;
- at most 64 KiB.

There is no command-line path to an executable and no free environment
override.

### Manifest schema `vasp-port.binary-manifest.v1`

```json
{"schema": "vasp-port.binary-manifest.v1", "name": "NAME", "status": "reviewed",
 "executable": {"path": "/abs/private/relink/vasp_std_vmb", "sha256": "…", "bytes": 0},
 "vasp": {"version": "6.6.1", "variant": "std"},
 "backend": {"name": "vasp-metal-backend", "version": "0.3.0", "source_commit": "…",
             "source_digest": "…", "source_dirty": true, "library_sha256": "…",
             "report_schema": "vmb.rank-report.v1",
             "aggregator": {"path": "/abs/…/vmb_evidence.py", "sha256": "…"}},
 "build": {"objects": "/abs/…/build/std", "object_count": 321,
           "relink_script_sha256": "…", "link_log_sha256": "…", "built_utc": "…"},
 "toolchain": {"PORT_GCC_VERSION": "…", "PORT_OPENMPI_VERSION": "…", "PORT_PRRTE_VERSION": "…",
               "PORT_OPENBLAS_VERSION": "…", "PORT_SCALAPACK_VERSION": "…", "PORT_FFTW_VERSION": "…"},
 "runtime_dependencies": [{"path": "/opt/homebrew/…/libopenblas.0.dylib", "sha256": "…"}],
 "selection": {"mode": "force", "shadow_check": true}}
```

### What the runner refuses

The runner refuses to start if any of these holds:
- the manifest is missing, unreadable, malformed or not `reviewed`;
- `variant` differs from `--binary`;
- the version is not 6.6.1;
- the executable is not a canonical regular arm64 Mach-O file, or its digest or
  size differs from the manifest;
- the toolchain differs from the runner's effective toolchain (defaults plus the
  host-local override);
- any runtime dependency or the aggregator digest differs;
- the mode is not `off`, `auto` or `force`;
- the check setting is not a boolean;
- `--controlled-suspension` is requested: the manifest route is not qualified
  for suspension.

The digests are checked again just before launch. Ordinary
`--binary std|gam|ncl` without a manifest is unchanged.

### Payload environment

The runner removes every inherited `VMB_*` variable, then sets:

| Variable | Value |
| --- | --- |
| `VMB_MODE` | the manifest mode |
| `VMB_CHECK` | `1` or `0`, from the manifest |
| `VMB_REPORT_DIR` | `OUTPUT/vmb-reports`, a new 0700 directory |
| `VMB_ATTEMPT_ID` | see below |
| `VMB_EXPECT_WORLD_SIZE` | `--ranks` |
| `VMB_TELEMETRY` | `rank` |

`VMB_ATTEMPT_ID` is the CMW attempt ID from `CMW_JOBS_ATTEMPT_ID` when it runs as
a managed CMW payload. Otherwise the runner generates a run ID and records it as
a run ID, not a CMW attempt ID.

After the payload ends, the runner runs the pinned aggregator. It writes
`OUTPUT/vmb-evidence.json` (exclusive create) and records the result in
`RUN_METADATA.txt`. The launcher status still comes from VASP. Backend evidence
problems are recorded and shown; they are never hidden by a successful exit.

### RUN_METADATA additions (manifest route only)

**Staging block:**
- `binary_selection: manifest`
- `binary_manifest: PATH; sha256=…; name=NAME`
- `backend: vasp-metal-backend 0.3.0; source_commit=…; source_digest=…; library_sha256=…`
- `backend_mode: force`
- `backend_check: True`
- `backend_attempt_id: ID; origin=cmw|runner`
- `backend_report_dir: OUTPUT/vmb-reports`

**Terminal block:**
- `backend_evidence: OUTPUT/vmb-evidence.json; sha256=…`
- `backend_evidence_status: COMPLETE|INCOMPLETE|INVALID|AGGREGATION_FAILED`
- `backend_summary: {compact JSON}`

Ordinary stock runs write exactly the older keys, so the older formats stay
valid.

## D. CMW integration

- **Argv grammar.** `--binary-manifest NAME` is accepted only together with
  `--binary std`. It is rejected with controlled suspension.
- **`cmw vasp check-runner`.** It reads the manifest and executable statically
  and never executes them. It records the binary and backend identity, and lists
  the manifest-derived files that must be pinned.
- **Node profiles.** A VASP profile may declare `binary_manifests: [NAME…]`. The
  manifest, executable, aggregator and runtime dependencies are then pinned like
  the toolchain sources.
- **Worker.** It exports `CMW_JOBS_ATTEMPT_ID` to the payload.
- **Result binding.** It accepts the new metadata keys only for manifest argv,
  requires `backend_attempt_id` to equal the job's attempt ID, and re-verifies
  the evidence digest and its summary invariants.
- **Display.** `jobs show` and the selected-job details show a one-line backend
  summary. Evidence problems stay visible even when the exit is successful.
