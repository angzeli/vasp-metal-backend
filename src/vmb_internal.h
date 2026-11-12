#ifndef VMB_INTERNAL_H
#define VMB_INTERNAL_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "vmb.h"

/*
 * One GPU GEMM. Matrices are pre-scaled by exact powers of two so that every
 * packed component lies in [-1, 1]; the kernel returns
 *   out = 2^so * (alpha_s * op(A') op(B') + beta_s * C')
 * with A' = 2^-sa A, B' = 2^-sb B, C' = 2^-sc C and alpha_s, beta_s chosen by
 * the caller so that out equals alpha op(A) op(B) + beta C. The caller has
 * already checked the exponent-range envelope (docs/PRECISION.md).
 */
struct vmb_gpu_job {
    int ta, tb;                  /* 0 = N, 1 = T, 2 = C */
    int m, n, k;
    const double *a; int lda, rows_a, cols_a, sa;
    const double *b; int ldb, rows_b, cols_b, sb;
    const double *c; int ldc, beta_zero, sc;
    double alpha_s[2], beta_s[2];
    int so;
    double *out;                 /* dense m x n, interleaved (re, im) */
    /* Filled by vmb_gpu_zgemm, seconds. */
    double t_pack, t_encode, t_gpu, t_wait, t_unpack;
    double t_queue, t_wake;      /* commit -> GPU start, GPU end -> wake-up */
};

/* Lazily creates device, kernel and queue; returns 1 when usable. */
int vmb_gpu_available(int simulate_no_device);
const char *vmb_gpu_device_name(void);
const char *vmb_gpu_init_error(void);

/* Returns 0 on completed execution. *submitted is set once committed. */
int vmb_gpu_zgemm(struct vmb_gpu_job *job, int *submitted);

/* Opt-in aggregated shape telemetry (VMB_TELEMETRY=path). No operand values
   or addresses are recorded. */
struct vmb_shape {
    int m, n, k, lda, ldb, ldc;
    char ta, tb;
    int beta_class;              /* 0: beta == 0, 1: beta == 1, 2: other */
};
void vmb_telemetry_record(const struct vmb_shape *shape, int status, double seconds);
void vmb_telemetry_write(const char *path);
/* Writes the table to an open stream; returns the dropped-shape count. */
uint64_t vmb_telemetry_write_stream(FILE *out);

double vmb_now(void);            /* monotonic seconds */

/* Consistent view of the process counters for the rank report. cpu_library is
   "" until the first call resolves OpenBLAS; nothing here initialises Metal. */
struct vmb_snapshot {
    int mode, check;
    uint64_t stats[VMB_NSTAT];
    double max_check_ratio;
    const char *cpu_library;
    /* Seconds inside zgemm_: GPU attempts (including declined checks) and the
       OpenBLAS path; calls by decade of m*n*k (index 0 also holds work < 10). */
    double try_s, cpu_s;
    uint64_t work_decades[16];
};
void vmb_snapshot_get(struct vmb_snapshot *out);
const char *vmb_mode_name(int mode);
const char *vmb_status_name(int status);

/* Per-rank report route (VMB_REPORT_DIR, docs/CONTRACT-v0.3.md). Configured in a
   library constructor; 1 when per-rank telemetry is to be recorded. */
int vmb_rank_telemetry(void);

/* Exported SYMBOL from the first loaded image whose basename starts with
   IMAGE_PREFIX, or NULL. SELF (our own definition) is never returned, so an
   interposer cannot resolve to itself. The image path is copied to IMAGE. */
void *vmb_find_symbol(const char *image_prefix, const char *symbol, const void *self,
                      char *image, size_t image_size);

/* v0.5 profiling of DGEMV, DGEMM and FFTW (rank-RRRRR.profile.json). Active only
   on the per-rank telemetry route. 0 off, 1 profile, 2 profile plus the private
   DGEMV reuse probe. */
enum { VMB_PROFILE_OFF, VMB_PROFILE_ON, VMB_PROFILE_REUSE };
int vmb_profile_level(void);
const char *vmb_profile_level_name(int level);
uint64_t vmb_ticks(void);        /* mach_absolute_time */
double vmb_ticks_to_s(uint64_t ticks);
/* JSON description of a return address: image basename, image-relative
   offset and the nearest exported symbol. The address itself is not written. */
void vmb_site_write_json(FILE *out, const void *addr);
void vmb_json_string(FILE *out, const char *s);
/* Profile sections; each writes one JSON object and returns its dropped count. */
uint64_t vmb_blas_write_json(FILE *out);
uint64_t vmb_fftw_write_json(FILE *out);
/* v0.5 opt-in FFTW plan cache (VMB_FFTW_PLAN_CACHE, build default
   PLAN_CACHE_DEFAULT): its report section, written whether it is on or off. */
void vmb_fftw_cache_write_json(FILE *out);
/* v0.5 opt-in DGEMV/DGEMM library (VMB_DBLAS_LIBRARY, build default
   DBLAS_ACCELERATE_DEFAULT): its report section, written whichever is used. */
void vmb_dblas_write_json(FILE *out);

#endif
