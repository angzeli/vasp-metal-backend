#ifndef VMB_INTERNAL_H
#define VMB_INTERNAL_H

#include <stddef.h>

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

double vmb_now(void);            /* monotonic seconds */

#endif
