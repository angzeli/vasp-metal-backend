/*
 * vasp-metal-backend: opt-in Apple Metal backend for complex GEMM (ZGEMM).
 *
 * The library defines the reference-BLAS symbol `zgemm_` (LP64, 32-bit
 * integers, gfortran hidden character lengths). When linked ahead of OpenBLAS
 * it receives every ZGEMM call of the program. With VMB_ENABLE=1, supported
 * calls are computed on the GPU in float-float ("df64") arithmetic; every other
 * call is forwarded unchanged to the original OpenBLAS `zgemm_`.
 *
 * Complex matrices are column-major arrays of interleaved (re, im) doubles.
 */
#ifndef VMB_H
#define VMB_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Outcome of vmb_zgemm_try. Only VMB_OK writes to C. */
enum vmb_status {
    VMB_OK = 0,
    VMB_DECLINED_DISABLED,     /* VMB_ENABLE is not 1 */
    VMB_DECLINED_TRIVIAL,      /* m or n or k is 0, or alpha == 0 */
    VMB_DECLINED_BADARG,       /* invalid trans flag or leading dimension */
    VMB_DECLINED_NONFINITE,    /* NaN/Inf in alpha, beta, A, B or read part of C */
    VMB_DECLINED_TOO_SMALL,    /* m*n*k below VMB_MIN_WORK */
    VMB_DECLINED_TOO_LARGE,    /* above VMB_MAX_WORK or VMB_MAX_BYTES */
    VMB_DECLINED_NO_DEVICE,    /* no Metal device or kernel build failed */
    VMB_FAILED_GPU,            /* command buffer did not complete */
    VMB_FAILED_GPU_NONFINITE,  /* GPU produced a non-finite element */
    VMB_FAILED_CHECK,          /* shadow check exceeded the tolerance */
    VMB_NSTATUS
};

/* Counter indices for vmb_stats_get (Fortran index = value + 1). */
enum vmb_stat {
    VMB_STAT_CALLS = 0,      /* zgemm_ entries */
    VMB_STAT_GPU_SUBMITTED,  /* command buffers committed */
    VMB_STAT_GPU_COMPLETED,  /* command buffers completed without error */
    VMB_STAT_CONSUMED,       /* GPU results written into the caller's C */
    VMB_STAT_CPU_CALLS,      /* calls served by the original CPU zgemm_ */
    VMB_STAT_CHECKED,        /* GPU results compared in shadow-check mode */
    VMB_STAT_REASON_BASE,    /* + vmb_status: count of calls per outcome */
    VMB_NSTAT = VMB_STAT_REASON_BASE + VMB_NSTATUS
};

/*
 * Precision policy (see docs/PRECISION.md). For each element of
 * C_new = alpha op(A) op(B) + beta C, with
 *   S_ij = |alpha| sum_l |op(A)_il| |op(B)_lj| + |beta| |C_ij|   (complex moduli)
 * the backend targets |C_gpu - C_exact|_ij <= VMB_ATOL(S_max) + VMB_RTOL(k) S_ij.
 */
#define VMB_RTOL(k) (8.0 * ((double)(k) + 2.0) * 0x1p-48)
#define VMB_ATOL(smax) (1e-18 * (smax))

/* Reference-BLAS ZGEMM entry point (interposed). */
void zgemm_(const char *transa, const char *transb, const int *m, const int *n,
            const int *k, const double *alpha, const double *a, const int *lda,
            const double *b, const int *ldb, const double *beta, double *c,
            const int *ldc, size_t len_transa, size_t len_transb);

/* Try the GPU path regardless of VMB_ENABLE; returns a vmb_status. */
int vmb_zgemm_try(char transa, char transb, int m, int n, int k,
                  const double alpha[2], const double *a, int lda,
                  const double *b, int ldb, const double beta[2], double *c,
                  int ldc);

/* The original CPU zgemm_ (OpenBLAS), never the GPU path. */
void vmb_cpu_zgemm(char transa, char transb, int m, int n, int k,
                   const double alpha[2], const double *a, int lda,
                   const double *b, int ldb, const double beta[2], double *c,
                   int ldc);

void vmb_stats_get(uint64_t out[VMB_NSTAT]);
void vmb_stats_reset(void);
double vmb_stats_max_check_ratio(void);
const char *vmb_device_name(void);   /* NULL when no device */
const char *vmb_cpu_library(void);   /* image providing the original zgemm_ */
void vmb_config_reload(void);        /* re-read VMB_* environment (tests) */
void vmb_report(FILE *out);

#ifdef __cplusplus
}
#endif

#endif
