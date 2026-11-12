/* v0.5 forwarding test: the interposed dgemv_, dgemm_ and dfftw_* entry points
 * must give results bit-identical to calling OpenBLAS and FFTW directly, with
 * profiling off or on (the profile level comes from the environment through
 * the per-rank route). Prints the calls it made, for the profile checks in
 * tests/test_profile_route.py. */
#include <fftw3.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void *vmb_find_symbol(const char *image_prefix, const char *symbol, const void *self,
                      char *image, size_t image_size);

typedef void (*dgemv_fn)(const char *, const int *, const int *, const double *,
                         const double *, const int *, const double *, const int *,
                         const double *, double *, const int *, size_t);
typedef void (*dgemm_fn)(const char *, const char *, const int *, const int *, const int *,
                         const double *, const double *, const int *, const double *,
                         const int *, const double *, double *, const int *, size_t, size_t);
void dgemv_(const char *, const int *, const int *, const double *, const double *,
            const int *, const double *, const int *, const double *, double *, const int *,
            size_t);
void dgemm_(const char *, const char *, const int *, const int *, const int *, const double *,
            const double *, const int *, const double *, const int *, const double *, double *,
            const int *, size_t, size_t);
void dfftw_plan_dft_3d_(void **, const int *, const int *, const int *, void *, void *,
                        const int *, const int *);
void dfftw_plan_many_dft_(void **, const int *, const int *, const int *, void *, const int *,
                          const int *, const int *, void *, const int *, const int *,
                          const int *, const int *, const int *);
void dfftw_plan_dft_r2c_3d_(void **, const int *, const int *, const int *, void *, void *,
                            const int *);
void dfftw_plan_dft_c2r_3d_(void **, const int *, const int *, const int *, void *, void *,
                            const int *);
void dfftw_execute_(void *const *);
void dfftw_execute_dft_(void *const *, void *, void *);
void dfftw_execute_dft_r2c_(void *const *, void *, void *);
void dfftw_execute_dft_c2r_(void *const *, void *, void *);
void dfftw_destroy_plan_(void **);

static int failures;

static void check(int ok, const char *what)
{
    if (!ok) {
        failures++;
        printf("FAIL: %s\n", what);
    }
}

static void fill(double *x, size_t n, unsigned seed)
{
    for (size_t i = 0; i < n; i++) {
        seed = seed * 1103515245u + 12345u;
        x[i] = ((seed >> 8) & 0xffff) / 32768.0 - 1.0;
    }
}

static int same_bits(const double *x, const double *y, size_t n)
{
    return memcmp(x, y, n * sizeof *x) == 0;
}

static int dgemv_calls, dgemm_calls;

/* One call site each (no inlining, unrolling or tail call moves it):
   the profile keys calls by return address. */
__attribute__((noinline, disable_tail_calls)) static void call_dgemv(char trans, int m, int n, double alpha,
                                                 const double *a, int lda, const double *x,
                                                 int incx, double beta, double *y, int incy)
{
    dgemv_(&trans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy, 1);
}

__attribute__((noinline, disable_tail_calls)) static void call_plan_3d(void **p, int *nx, int *ny, int *nz,
                                                   void *in, void *out, int *sign, int *flags)
{
    dfftw_plan_dft_3d_(p, nx, ny, nz, in, out, sign, flags);
}

static void gemv_case(dgemv_fn ref, char trans, int m, int n, int lda, int incx, int incy,
                      double alpha, double beta, int repeat)
{
    int tr = trans != 'N';
    int lx = tr ? m : n, ly = tr ? n : m;
    size_t nx = (size_t)(lx - 1) * abs(incx) + 1, ny = (size_t)(ly - 1) * abs(incy) + 1;
    double *a = malloc(sizeof(double) * (size_t)lda * n), *x = malloc(sizeof(double) * nx);
    double *y1 = malloc(sizeof(double) * ny), *y2 = malloc(sizeof(double) * ny);
    fill(a, (size_t)lda * n, 1);
    fill(x, nx, 2);
    fill(y1, ny, 3);
    if (beta == 0.0)
        for (size_t i = 0; i < ny; i++)
            y1[i] = NAN;  /* beta = 0 must not read y */
    memcpy(y2, y1, sizeof(double) * ny);
    for (int r = 0; r < repeat; r++) {
        call_dgemv(trans, m, n, alpha, a, lda, x, incx, beta, y1, incy);
        ref(&trans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y2, &incy, 1);
        dgemv_calls++;
    }
    char what[96];
    snprintf(what, sizeof what, "dgemv %c %dx%d lda %d inc %d,%d", trans, m, n, lda, incx, incy);
    check(same_bits(y1, y2, ny), what);
    free(a), free(x), free(y1), free(y2);
}

static void gemm_case(dgemm_fn ref, char ta, char tb, int m, int n, int k, double beta)
{
    int ra = ta == 'N' ? m : k, ca = ta == 'N' ? k : m, rb = tb == 'N' ? k : n,
        cb = tb == 'N' ? n : k, lda = ra + 3, ldb = rb + 1, ldc = m + 2;
    double alpha = 0.75;
    double *a = malloc(sizeof(double) * lda * ca), *b = malloc(sizeof(double) * ldb * cb);
    double *c1 = malloc(sizeof(double) * ldc * n), *c2 = malloc(sizeof(double) * ldc * n);
    fill(a, (size_t)lda * ca, 4);
    fill(b, (size_t)ldb * cb, 5);
    fill(c1, (size_t)ldc * n, 6);
    memcpy(c2, c1, sizeof(double) * ldc * n);
    dgemm_(&ta, &tb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c1, &ldc, 1, 1);
    ref(&ta, &tb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c2, &ldc, 1, 1);
    dgemm_calls++;
    char what[64];
    snprintf(what, sizeof what, "dgemm %c%c %dx%dx%d", ta, tb, m, n, k);
    check(same_bits(c1, c2, (size_t)ldc * n), what);
    free(a), free(b), free(c1), free(c2);
}

int main(void)
{
    dgemv_fn ref_gemv = (dgemv_fn)vmb_find_symbol("libopenblas", "dgemv_", (void *)dgemv_, NULL, 0);
    dgemm_fn ref_gemm = (dgemm_fn)vmb_find_symbol("libopenblas", "dgemm_", (void *)dgemm_, NULL, 0);
    check(ref_gemv && ref_gemm, "OpenBLAS symbols resolved");
    if (!ref_gemv || !ref_gemm)
        return 1;

    gemv_case(ref_gemv, 'N', 37, 11, 40, 1, 1, 1.0, 0.0, 3);
    gemv_case(ref_gemv, 'T', 37, 11, 37, 2, -1, -0.5, 1.0, 1);
    gemv_case(ref_gemv, 'N', 200, 64, 203, -3, 2, 2.0, 0.25, 2);
    gemv_case(ref_gemv, 'T', 1, 1, 1, 1, 1, 1.0, 0.0, 1);
    gemm_case(ref_gemm, 'N', 'N', 17, 9, 13, 0.0);
    gemm_case(ref_gemm, 'T', 'N', 64, 48, 32, 1.0);
    gemm_case(ref_gemm, 'N', 'T', 5, 70, 3, -2.0);

    /* FFTW: the same problem planned through the interposed Fortran interface
       and the C interface, both FFTW_ESTIMATE (deterministic plans). */
    int nx = 6, ny = 5, nz = 8, flags = FFTW_ESTIMATE, sign = FFTW_FORWARD;
    size_t n3 = (size_t)nx * ny * nz;
    fftw_complex *in = fftw_alloc_complex(n3), *o1 = fftw_alloc_complex(n3),
                 *o2 = fftw_alloc_complex(n3);
    fill((double *)in, 2 * n3, 7);
    void *p1;
    int fftw_plans = 0, fftw_execs = 0, fftw_destroys = 0;
    for (int r = 0; r < 3; r++) {  /* three plans of one shape: repeated planning */
        call_plan_3d(&p1, &nx, &ny, &nz, in, o1, &sign, &flags);
        fftw_plans++;
        dfftw_execute_(&p1);
        fftw_execs++;
        dfftw_execute_dft_(&p1, in, o1);
        fftw_execs++;
        dfftw_destroy_plan_(&p1);
        fftw_destroys++;
    }
    fftw_plan p2 = fftw_plan_dft_3d(nz, ny, nx, in, o2, sign, flags);  /* Fortran order */
    fftw_execute(p2);
    fftw_destroy_plan(p2);
    check(same_bits((double *)o1, (double *)o2, 2 * n3), "dfftw_plan_dft_3d_ forwarding");

    int rank = 1, len = 16, howmany = 4, stride = 1, dist = 16, back = FFTW_BACKWARD;
    fftw_complex *b_in = fftw_alloc_complex(64), *b1 = fftw_alloc_complex(64),
                 *b2 = fftw_alloc_complex(64);
    fill((double *)b_in, 128, 8);
    dfftw_plan_many_dft_(&p1, &rank, &len, &howmany, b_in, &len, &stride, &dist, b1, &len,
                         &stride, &dist, &back, &flags);
    fftw_plans++;
    dfftw_execute_dft_(&p1, b_in, b1);
    fftw_execs++;
    dfftw_destroy_plan_(&p1);
    fftw_destroys++;
    p2 = fftw_plan_many_dft(1, &len, howmany, b_in, &len, 1, dist, b2, &len, 1, dist, back, flags);
    fftw_execute(p2);
    fftw_destroy_plan(p2);
    check(same_bits((double *)b1, (double *)b2, 128), "dfftw_plan_many_dft_ forwarding");

    double *r_in = fftw_alloc_real(n3), *r_back1 = fftw_alloc_real(n3),
           *r_back2 = fftw_alloc_real(n3);
    size_t nh = (size_t)(nx / 2 + 1) * ny * nz;
    fftw_complex *h1 = fftw_alloc_complex(nh), *h2 = fftw_alloc_complex(nh);
    fill(r_in, n3, 9);
    void *pr, *pc;
    dfftw_plan_dft_r2c_3d_(&pr, &nx, &ny, &nz, r_in, h1, &flags);
    dfftw_plan_dft_c2r_3d_(&pc, &nx, &ny, &nz, h1, r_back1, &flags);
    fftw_plans += 2;
    dfftw_execute_dft_r2c_(&pr, r_in, h1);
    dfftw_execute_dft_c2r_(&pc, h1, r_back1);
    fftw_execs += 2;
    dfftw_destroy_plan_(&pr);
    dfftw_destroy_plan_(&pc);
    fftw_destroys += 2;
    fftw_plan q1 = fftw_plan_dft_r2c_3d(nz, ny, nx, r_in, h2, flags);
    fftw_plan q2 = fftw_plan_dft_c2r_3d(nz, ny, nx, h2, r_back2, flags);
    fftw_execute(q1);
    fftw_execute(q2);
    fftw_destroy_plan(q1);
    fftw_destroy_plan(q2);
    check(same_bits((double *)h1, (double *)h2, 2 * nh), "dfftw_plan_dft_r2c_3d_ forwarding");
    check(same_bits(r_back1, r_back2, n3), "dfftw_plan_dft_c2r_3d_ forwarding");

    printf("calls dgemv=%d dgemm=%d fftw_plans=%d fftw_executes=%d fftw_destroys=%d\n",
           dgemv_calls, dgemm_calls, fftw_plans, fftw_execs, fftw_destroys);
    printf("%s: %d failures\n", failures ? "FAIL" : "PASS", failures);
    return failures != 0;
}
