#ifndef VMB_INTERNAL_H
#define VMB_INTERNAL_H

/*
 * One GPU GEMM. Matrices are pre-scaled by exact powers of two so that every
 * packed component lies in [-1, 1]; the kernel returns
 *   out = 2^so * (alpha_s * op(A') op(B') + beta_s * C')
 * with A' = 2^-sa A, B' = 2^-sb B, C' = 2^-sc C and alpha_s, beta_s chosen by
 * the caller so that out equals alpha op(A) op(B) + beta C.
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
};

/* Lazily creates device, kernel and queue; returns 1 when usable. */
int vmb_gpu_available(int simulate_no_device);
const char *vmb_gpu_device_name(void);
const char *vmb_gpu_init_error(void);

/* Returns 0 on completed execution. *submitted is set once committed. */
int vmb_gpu_zgemm(const struct vmb_gpu_job *job, int *submitted);

#endif
