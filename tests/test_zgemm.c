/*
 * Standalone tests of the interposed zgemm_: GPU accuracy against the
 * independent oracle, dispatch accounting, and every fallback route.
 * Tolerances are fixed in vmb.h (VMB_RTOL, VMB_ATOL) before any measurement.
 */
#define _DARWIN_C_SOURCE
#include "oracle.h"
#include "vmb.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum kind { RANDOM, WIDE, TINY, HUGE_VALS };

struct mat {
    int rows, cols, ld;
    double *v;
};

static int failures, cases;
static double worst_ratio, worst_rel;
static uint64_t rng_state = 0x9E3779B97F4A7C15ull;

static double urand(void)  /* splitmix64 -> [-1, 1) */
{
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    return (double)(z >> 11) * 0x1p-52 - 1.0;
}

static double value(enum kind kind)
{
    switch (kind) {
    case WIDE: return urand() * pow(10.0, 30.0 * urand());
    case TINY: return urand() * 1e-150;
    case HUGE_VALS: return urand() * 1e120;
    default: return urand();
    }
}

static int is_trans(char t) { return t != 'N' && t != 'n'; }

/* Storage with ld = rows + pad; every stored slot starts as NaN so that a read
   of padding (or of C when beta == 0) would poison the result. */
static struct mat mat_new(int rows, int cols, int pad)
{
    struct mat x = {rows, cols, (rows > 0 ? rows : 1) + pad, NULL};
    size_t len = 2 * (size_t)x.ld * (size_t)(cols > 0 ? cols : 1);
    x.v = malloc(len * sizeof *x.v);
    for (size_t e = 0; e < len; e++)
        x.v[e] = NAN;
    return x;
}

static struct mat mat_copy(const struct mat *x)
{
    struct mat y = *x;
    size_t len = 2 * (size_t)x->ld * (size_t)(x->cols > 0 ? x->cols : 1);
    y.v = malloc(len * sizeof *y.v);
    memcpy(y.v, x->v, len * sizeof *y.v);
    return y;
}

static size_t mat_bytes(const struct mat *x)
{
    return 2 * (size_t)x->ld * (size_t)(x->cols > 0 ? x->cols : 1) * sizeof(double);
}

/* Store the value of op(M)(row, col) into M's storage. */
static void set_op(struct mat *x, char trans, int row, int col, double re, double im)
{
    double *z = is_trans(trans) ? x->v + 2 * ((size_t)col + (size_t)row * x->ld)
                                : x->v + 2 * ((size_t)row + (size_t)col * x->ld);
    z[0] = re;
    z[1] = (trans == 'C' || trans == 'c') ? -im : im;
}

static void fill(struct mat *x, enum kind kind)
{
    for (int j = 0; j < x->cols; j++)
        for (int i = 0; i < x->rows; i++)
            set_op(x, 'N', i, j, value(kind), value(kind));
}

static void call(char ta, char tb, int m, int n, int k, const double *alpha, struct mat *a,
                 struct mat *b, const double *beta, struct mat *c, int lda)
{
    zgemm_(&ta, &tb, &m, &n, &k, alpha, a->v, &lda, b->v, &b->ld, beta, c->v, &c->ld, 1, 1);
}

static void set_env(const char *name, const char *value)
{
    if (value)
        setenv(name, value, 1);
    else
        unsetenv(name);
    vmb_config_reload();
}

static void expect(int ok, const char *label, const char *what)
{
    if (!ok) {
        failures++;
        printf("FAIL %s: %s\n", label, what);
    }
}

/* Run one GPU case on given op(A), op(B) storage and compare to the oracle. */
static void gpu_compare(const char *label, char ta, char tb, int m, int n, int k,
                        const double alpha[2], struct mat *a, struct mat *b,
                        const double beta[2], struct mat *c0)
{
    struct mat cg = mat_copy(c0), cr = mat_copy(c0);
    double *s = malloc((size_t)m * n * sizeof *s);
    uint64_t before[VMB_NSTAT], after[VMB_NSTAT];
    vmb_stats_get(before);
    call(ta, tb, m, n, k, alpha, a, b, beta, &cg, a->ld);
    vmb_stats_get(after);
    oracle_zgemm(ta, tb, m, n, k, alpha, a->v, a->ld, b->v, b->ld, beta, cr.v, cr.ld, s);

    double smax = 0.0, ratio = 0.0, max_abs = 0.0, max_rel = 0.0;
    for (size_t e = 0; e < (size_t)m * n; e++)
        smax = fmax(smax, s[e]);
    int pad_ok = 1;
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            const double *g = cg.v + 2 * ((size_t)i + (size_t)j * cg.ld);
            const double *r = cr.v + 2 * ((size_t)i + (size_t)j * cr.ld);
            double err = hypot(g[0] - r[0], g[1] - r[1]), sij = s[i + (size_t)j * m];
            double tol = VMB_ATOL(smax) + VMB_RTOL(k) * sij;
            double q = err == 0.0 ? 0.0 : tol > 0.0 ? err / tol : INFINITY;
            if (isnan(q) || isnan(ratio))  /* sticky: NaN/inf are never cleared */
                ratio = NAN;
            else if (q > ratio)
                ratio = q;
            max_abs = fmax(max_abs, err);
            if (sij > 0)
                max_rel = fmax(max_rel, err / sij);
        }
        for (int i = m; i < cg.ld; i++)  /* C padding must be bit-identical */
            pad_ok &= memcmp(cg.v + 2 * ((size_t)i + (size_t)j * cg.ld),
                             c0->v + 2 * ((size_t)i + (size_t)j * c0->ld), 16) == 0;
    }
    int dispatched = after[VMB_STAT_CONSUMED] - before[VMB_STAT_CONSUMED] == 1 &&
                     after[VMB_STAT_GPU_COMPLETED] - before[VMB_STAT_GPU_COMPLETED] == 1 &&
                     after[VMB_STAT_CPU_CALLS] == before[VMB_STAT_CPU_CALLS];
    cases++;
    expect(dispatched, label, "GPU result not consumed exactly once");
    expect(ratio <= 1.0, label, "error exceeds tolerance");
    expect(pad_ok, label, "C padding modified");
    if (ratio > worst_ratio)
        worst_ratio = ratio;
    if (max_rel > worst_rel)
        worst_rel = max_rel;
    printf("%-28s m=%3d n=%3d k=%3d  err/tol=%.3e  max|err|=%.3e  max|err|/S=%.3e\n", label,
           m, n, k, ratio, max_abs, max_rel);
    free(cg.v);
    free(cr.v);
    free(s);
}

static void gpu_case(const char *label, char ta, char tb, int m, int n, int k, int pad,
                     const double alpha[2], const double beta[2], enum kind kind)
{
    struct mat a = is_trans(ta) ? mat_new(k, m, pad) : mat_new(m, k, pad);
    struct mat b = is_trans(tb) ? mat_new(n, k, pad + 1) : mat_new(k, n, pad + 1);
    struct mat c = mat_new(m, n, pad + 2);
    fill(&a, kind);
    fill(&b, kind);
    if (beta[0] != 0.0 || beta[1] != 0.0)
        fill(&c, kind);  /* else C stays NaN: it must not be read */
    gpu_compare(label, ta, tb, m, n, k, alpha, &a, &b, beta, &c);
    free(a.v);
    free(b.v);
    free(c.v);
}

/* Paired terms x*y and -x(1+d)*y cancel to about d of the scale S. */
static void cancel_case(const char *label, char ta, char tb, int m, int n, int k)
{
    const double alpha[2] = {1.0, 0.0}, beta[2] = {0.0, 0.0}, d = 0x1p-30;
    struct mat a = is_trans(ta) ? mat_new(k, m, 1) : mat_new(m, k, 1);
    struct mat b = is_trans(tb) ? mat_new(n, k, 2) : mat_new(k, n, 2);
    struct mat c = mat_new(m, n, 0);
    for (int p = 0; p + 1 < k; p += 2) {
        for (int i = 0; i < m; i++) {
            double xr = 1e8 * urand(), xi = 1e8 * urand();
            set_op(&a, ta, i, p, xr, xi);
            set_op(&a, ta, i, p + 1, xr * (1.0 + d), xi * (1.0 + d));
        }
        for (int j = 0; j < n; j++) {
            double yr = urand(), yi = urand();
            set_op(&b, tb, p, j, yr, yi);
            set_op(&b, tb, p + 1, j, -yr, -yi);
        }
    }
    gpu_compare(label, ta, tb, m, n, k, alpha, &a, &b, beta, &c);
    free(a.v);
    free(b.v);
    free(c.v);
}

/* A declined call must be served by the original CPU zgemm_, bit for bit. */
static void fallback_case(const char *label, int expect_status, char ta, char tb, int m,
                          int n, int k, const double alpha[2], const double beta[2],
                          int lda_shrink, int poison_a, int expect_submitted)
{
    struct mat a = is_trans(ta) ? mat_new(k, m, 1) : mat_new(m, k, 1);
    struct mat b = is_trans(tb) ? mat_new(n, k, 1) : mat_new(k, n, 1);
    struct mat c = mat_new(m, n, 1);
    fill(&a, RANDOM);
    fill(&b, RANDOM);
    fill(&c, RANDOM);
    if (poison_a)
        a.v[2] = NAN;
    struct mat cg = mat_copy(&c), cc = mat_copy(&c);
    int lda = a.ld - lda_shrink;
    uint64_t before[VMB_NSTAT], after[VMB_NSTAT];
    vmb_stats_get(before);
    call(ta, tb, m, n, k, alpha, &a, &b, beta, &cg, lda);
    vmb_stats_get(after);
    vmb_cpu_zgemm(ta, tb, m, n, k, alpha, a.v, lda, b.v, b.ld, beta, cc.v, cc.ld);
    cases++;
#define D(i) (after[i] - before[i])
    expect(D(VMB_STAT_REASON_BASE + expect_status) == 1, label, "unexpected outcome");
    expect(D(VMB_STAT_CONSUMED) == 0, label, "GPU result consumed");
    expect(D(VMB_STAT_CPU_CALLS) == 1, label, "CPU path not taken exactly once");
    expect(D(VMB_STAT_GPU_SUBMITTED) == (uint64_t)expect_submitted, label, "submission count");
    expect(memcmp(cg.v, cc.v, mat_bytes(&c)) == 0, label, "result differs from CPU path");
    printf("%-28s outcome=%d submitted=%llu completed=%llu consumed=%llu cpu=%llu  bitwise==CPU\n",
           label, expect_status, (unsigned long long)D(VMB_STAT_GPU_SUBMITTED),
           (unsigned long long)D(VMB_STAT_GPU_COMPLETED),
           (unsigned long long)D(VMB_STAT_CONSUMED), (unsigned long long)D(VMB_STAT_CPU_CALLS));
#undef D
    free(a.v);
    free(b.v);
    free(c.v);
    free(cg.v);
    free(cc.v);
}

static void determinism_case(void)
{
    const double alpha[2] = {0.5, -2.0}, beta[2] = {-0.3, 1.1};
    struct mat a = mat_new(37, 23, 2), b = mat_new(23, 19, 1), c = mat_new(37, 19, 3);
    fill(&a, RANDOM);
    fill(&b, RANDOM);
    fill(&c, RANDOM);
    struct mat c1 = mat_copy(&c), c2 = mat_copy(&c);
    call('N', 'N', 37, 19, 23, alpha, &a, &b, beta, &c1, a.ld);
    call('N', 'N', 37, 19, 23, alpha, &a, &b, beta, &c2, a.ld);
    cases++;
    expect(memcmp(c1.v, c2.v, mat_bytes(&c)) == 0, "determinism", "repeat differs");
    printf("%-28s repeated GPU result bitwise identical\n", "determinism");
    free(a.v);
    free(b.v);
    free(c.v);
    free(c1.v);
    free(c2.v);
}

static void latency_info(void)
{
    const double alpha[2] = {1.0, 0.0}, beta[2] = {0.0, 0.0};
    struct mat a = mat_new(32, 32, 0), b = mat_new(32, 32, 0), c = mat_new(32, 32, 0);
    fill(&a, RANDOM);
    fill(&b, RANDOM);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int r = 0; r < 50; r++)
        call('N', 'N', 32, 32, 32, alpha, &a, &b, beta, &c, a.ld);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double us = ((t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec)) / 50 / 1e3;
    printf("info: mean wall time per 32x32x32 GPU call (incl. packing, sync): %.1f us\n", us);
    free(a.v);
    free(b.v);
    free(c.v);
}

/* ---- v0.2: fail-closed tolerance, numerical-safety envelope, modes ---- */

void vmb_telemetry_write(const char *path);  /* internal; exercised below */

static void ratio_unit(const char *label, size_t count, const double *got, const double *ref,
                       const double *scale, int expect_pass, double expect_value)
{
    double r = vmb_tolerance_ratio(count, got, ref, scale, 4);
    int pass = r <= 1.0;
    cases++;
    expect(pass == expect_pass, label, "tolerance verdict");
    if (expect_value >= 0.0)
        expect(r == expect_value, label, "tolerance value");
    printf("%-28s ratio=%.3e -> %s\n", label, r, pass ? "pass" : "fail closed");
}

static void tolerance_units(void)
{
    const double z[4] = {0, 0, 0, 0};
    ratio_unit("tol: exact, zero tol", 2, z, z, (double[]){0, 0}, 1, 0.0);
    ratio_unit("tol: err>0, zero tol", 1, (double[]){1e-300, 0}, z, (double[]){0}, 0, -1);
    ratio_unit("tol: zero-tol fail sticky", 2, (double[]){1e-300, 0, 1, 0},
               (double[]){0, 0, 1, 0}, (double[]){0, 0}, 0, -1);
    ratio_unit("tol: NaN got sticky", 2, (double[]){NAN, 0, 1, 0}, (double[]){0, 0, 1, 0},
               (double[]){1, 1}, 0, -1);
    ratio_unit("tol: NaN ref", 1, (double[]){1, 0}, (double[]){NAN, 0}, (double[]){1}, 0, -1);
    ratio_unit("tol: inf scale", 1, (double[]){1, 0}, (double[]){1, 0}, (double[]){INFINITY}, 0, -1);
    ratio_unit("tol: NaN scale", 1, (double[]){1, 0}, (double[]){1, 0}, (double[]){NAN}, 0, -1);
    ratio_unit("tol: negative scale", 1, (double[]){1, 0}, (double[]){1, 0}, (double[]){-1}, 0, -1);
    ratio_unit("tol: difference overflow", 1, (double[]){1e308, 0}, (double[]){-1e308, 0},
               (double[]){1e308}, 0, -1);
    ratio_unit("tol: big then small err", 2, (double[]){2, 0, 1, 0}, (double[]){1, 0, 1, 0},
               (double[]){1, 1}, 0, -1);
    ratio_unit("tol: small err passes", 1, (double[]){1 + 0x1p-50, 0}, (double[]){1, 0},
               (double[]){1}, 1, -1);
}

enum { ACCURATE_OR_RANGE, MUST_RANGE, MUST_GPU };

/* Forced attempt with the shadow check off. Either the GPU result is consumed
   and accurate against the oracle, or a counted range fallback ran OpenBLAS
   exactly once with a bit-identical result. */
static void safety_case(const char *label, int want, char ta, char tb, int m, int n, int k,
                        const double alpha[2], struct mat *a, struct mat *b,
                        const double beta[2], struct mat *c0)
{
    struct mat cg = mat_copy(c0), cc = mat_copy(c0), cr = mat_copy(c0);
    double *s = malloc((size_t)m * n * sizeof *s), *got = malloc(16 * (size_t)m * n),
           *ref = malloc(16 * (size_t)m * n);
    uint64_t before[VMB_NSTAT], after[VMB_NSTAT];
    vmb_stats_get(before);
    call(ta, tb, m, n, k, alpha, a, b, beta, &cg, a->ld);
    vmb_stats_get(after);
    vmb_cpu_zgemm(ta, tb, m, n, k, alpha, a->v, a->ld, b->v, b->ld, beta, cc.v, cc.ld);
#define D(i) (after[i] - before[i])
    int consumed = D(VMB_STAT_CONSUMED) == 1;
    cases++;
    if (consumed) {
        oracle_zgemm(ta, tb, m, n, k, alpha, a->v, a->ld, b->v, b->ld, beta, cr.v, cr.ld, s);
        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++) {
                memcpy(got + 2 * (i + (size_t)j * m), cg.v + 2 * (i + (size_t)j * cg.ld), 16);
                memcpy(ref + 2 * (i + (size_t)j * m), cr.v + 2 * (i + (size_t)j * cr.ld), 16);
            }
        double r = vmb_tolerance_ratio((size_t)m * n, got, ref, s, k);
        expect(want != MUST_RANGE, label, "GPU result consumed outside the envelope");
        expect(r <= 1.0, label, "consumed GPU result inaccurate");
        printf("%-28s GPU consumed, err/tol=%.3e\n", label, r);
    } else {
        expect(want != MUST_GPU, label, "envelope declined an ordinary-range call");
        expect(D(VMB_STAT_REASON_BASE + VMB_DECLINED_RANGE) == 1, label, "not a range fallback");
        expect(D(VMB_STAT_GPU_SUBMITTED) == 0, label, "GPU work submitted before declining");
        expect(D(VMB_STAT_CPU_CALLS) == 1, label, "CPU path not taken exactly once");
        expect(memcmp(cg.v, cc.v, mat_bytes(c0)) == 0, label, "result differs from OpenBLAS");
        printf("%-28s range fallback, bitwise==OpenBLAS\n", label);
    }
#undef D
    free(cg.v), free(cc.v), free(cr.v), free(s), free(got), free(ref);
}

/* Random sign and mantissa in [0.5, 1) times 2^e, e uniform in [lo, hi]. */
static void fill_exp(struct mat *x, int lo, int hi)
{
    for (int j = 0; j < x->cols; j++)
        for (int i = 0; i < x->rows; i++) {
            double v[2];
            for (int p = 0; p < 2; p++) {
                int e = lo + (int)((urand() + 1.0) * 0.5 * (hi - lo + 1));
                v[p] = ldexp(urand() < 0 ? -0.5 - 0.25 * (urand() + 1) : 0.5 + 0.25 * (urand() + 1),
                             e > hi ? hi : e);
            }
            set_op(x, 'N', i, j, v[0], v[1]);
        }
}

static void fill_zero(struct mat *x)
{
    for (int j = 0; j < x->cols; j++)
        for (int i = 0; i < x->rows; i++)
            set_op(x, 'N', i, j, 0.0, 0.0);
}

static void safety_cases(void)
{
    const double one[2] = {1, 0}, zero[2] = {0, 0}, some[2] = {0.5, -2};
    struct mat a, b, c;

    /* The d3be811 defect: exact result I, both diagonal products lost. */
    a = mat_new(2, 2, 0), b = mat_new(2, 2, 0), c = mat_new(2, 2, 0);
    fill_zero(&a), fill_zero(&b);
    set_op(&a, 'N', 0, 0, 0x1p90, 0), set_op(&a, 'N', 1, 1, 0x1p-90, 0);
    set_op(&b, 'N', 0, 0, 0x1p-90, 0), set_op(&b, 'N', 1, 1, 0x1p90, 0);
    safety_case("safety: diag 2^+-90 2x2", ACCURATE_OR_RANGE, 'N', 'N', 2, 2, 2, one, &a, &b, zero, &c);
    {   /* caller-visible result is exactly I; a direct try leaves C untouched */
        struct mat cg = mat_copy(&c), ct = mat_copy(&c);
        call('N', 'N', 2, 2, 2, one, &a, &b, zero, &cg, a.ld);
        int st = vmb_zgemm_try('N', 'N', 2, 2, 2, one, a.v, a.ld, b.v, b.ld, zero, ct.v, ct.ld);
        cases++;
        expect(cg.v[0] == 1 && cg.v[6] == 1 && cg.v[1] == 0 && cg.v[2] == 0 && cg.v[4] == 0 &&
               cg.v[7] == 0, "safety: diag result", "C != I");
        expect(st == VMB_DECLINED_RANGE || st == VMB_OK, "safety: diag try", "unexpected status");
        if (st != VMB_OK)
            expect(memcmp(ct.v, c.v, mat_bytes(&c)) == 0, "safety: diag try", "partial write");
        printf("%-28s zgemm_ C == I exactly; direct try status=%d, C untouched\n",
               "safety: diag result", st);
        free(cg.v), free(ct.v);
    }
    free(a.v), free(b.v), free(c.v);

    /* The same pair embedded in an ordinary eligible 64x64x64 problem. */
    a = mat_new(64, 64, 3), b = mat_new(64, 64, 1), c = mat_new(64, 64, 2);
    fill(&a, RANDOM), fill(&b, RANDOM), fill(&c, RANDOM);
    set_op(&a, 'N', 0, 0, 0x1p90, 0), set_op(&a, 'N', 1, 1, 0x1p-90, 0);
    set_op(&b, 'N', 0, 0, 0x1p-90, 0), set_op(&b, 'N', 1, 1, 0x1p90, 0);
    safety_case("safety: diag in 64^3", ACCURATE_OR_RANGE, 'N', 'N', 64, 64, 64, one, &a, &b, some, &c);
    free(a.v), free(b.v), free(c.v);

    /* Mixed scales: mild ranges must still reach the GPU; wide ones may not. */
    a = mat_new(48, 40, 1), b = mat_new(40, 36, 2), c = mat_new(48, 36, 1);
    fill_exp(&a, -10, 10), fill_exp(&b, -10, 10), fill_exp(&c, -10, 10);
    safety_case("safety: mixed 2^+-10", MUST_GPU, 'N', 'N', 48, 36, 40, some, &a, &b, some, &c);
    fill_exp(&a, -40, 40), fill_exp(&b, -40, 40);
    safety_case("safety: mixed 2^+-40", ACCURATE_OR_RANGE, 'N', 'N', 48, 36, 40, some, &a, &b, zero, &c);
    fill_exp(&a, -200, 200), fill_exp(&b, -200, 200);
    safety_case("safety: mixed 2^+-200", ACCURATE_OR_RANGE, 'N', 'N', 48, 36, 40, some, &a, &b, zero, &c);
    free(a.v), free(b.v), free(c.v);

    /* Risky intermediate products: FP64 underflow, overflow, inf - inf. */
    a = mat_new(8, 6, 0), b = mat_new(6, 5, 0), c = mat_new(8, 5, 0);
    fill_exp(&a, -521, -519), fill_exp(&b, -521, -519);
    safety_case("safety: products ~2^-1040", MUST_RANGE, 'N', 'N', 8, 5, 6, one, &a, &b, zero, &c);
    fill_exp(&a, 519, 521), fill_exp(&b, 519, 521);
    safety_case("safety: products overflow", MUST_RANGE, 'N', 'N', 8, 5, 6, one, &a, &b, zero, &c);
    fill_zero(&a), fill_zero(&b);
    set_op(&a, 'N', 0, 0, 0x1p600, 0), set_op(&a, 'N', 0, 1, 0x1p600, 0);
    set_op(&b, 'N', 0, 0, 0x1p600, 0), set_op(&b, 'N', 1, 0, -0x1p600, 0);
    safety_case("safety: inf - inf on CPU", MUST_RANGE, 'N', 'N', 8, 5, 6, one, &a, &b, zero, &c);
    free(a.v), free(b.v), free(c.v);

    /* beta*C scale imbalance in both directions. */
    a = mat_new(24, 20, 1), b = mat_new(20, 16, 0), c = mat_new(24, 16, 1);
    fill(&a, RANDOM), fill(&b, RANDOM);
    fill_exp(&c, 299, 301);
    safety_case("safety: beta*C ~2^300", ACCURATE_OR_RANGE, 'C', 'N', 16, 16, 20, one, &a, &b, some, &c);
    fill_exp(&c, -301, -299);
    safety_case("safety: beta*C ~2^-300", MUST_GPU, 'N', 'N', 24, 16, 20, one, &a, &b, some, &c);
    fill_zero(&c);
    set_op(&c, 'N', 0, 0, 0x1p200, 0);
    safety_case("safety: one C elem 2^200", ACCURATE_OR_RANGE, 'N', 'N', 24, 16, 20, one, &a, &b, one, &c);
    free(a.v), free(b.v), free(c.v);

    /* Zero denominators: every S_ij is zero. */
    a = mat_new(7, 5, 1), b = mat_new(5, 6, 1), c = mat_new(7, 6, 1);
    fill_zero(&a), fill(&b, RANDOM);
    safety_case("safety: A == 0 (S == 0)", ACCURATE_OR_RANGE, 'N', 'N', 7, 6, 5, one, &a, &b, zero, &c);
    free(a.v), free(b.v), free(c.v);

    /* In-process shadow check fails closed on a perturbed GPU result. */
    set_env("VMB_CHECK", "1");
    set_env("VMB_TEST_FAULT", "gpu_perturb");
    fallback_case("shadow: perturbed GPU", VMB_FAILED_CHECK, 'N', 'C', 6, 5, 4, one, some, 0, 0, 1);
    set_env("VMB_TEST_FAULT", NULL);
    set_env("VMB_CHECK", NULL);
}

/* Unscaled-overflow regressions on real entries: N,N, m = n = 1, k = 2,
   beta = 0, VMB_MODE=force, VMB_CHECK=0. A direct try must decline with range
   and leave a sentinel C untouched; through zgemm_ the call must be one range
   fallback that is bit-identical to calling OpenBLAS directly. With
   want_range == 0 it is a positive control that must reach the GPU. */
static void overflow_case(const char *label, int want_range, double alpha_re,
                          const double a[2], const double b[2])
{
    const double alpha[2] = {alpha_re, 0}, zero[2] = {0, 0};
    const double av[4] = {a[0], 0, a[1], 0}, bv[4] = {b[0], 0, b[1], 0};
    const double sentinel[2] = {12345.0, -678.0};
    double ct[2], cg[2], cc[2];
    uint64_t before[VMB_NSTAT], after[VMB_NSTAT];
    memcpy(ct, sentinel, sizeof ct), memcpy(cg, sentinel, sizeof cg), memcpy(cc, sentinel, sizeof cc);

    int st = vmb_zgemm_try('N', 'N', 1, 1, 2, alpha, av, 1, bv, 2, zero, ct, 1);
    vmb_stats_get(before);
    char t = 'N';
    int one = 1, two = 2;
    zgemm_(&t, &t, &one, &one, &two, alpha, av, &one, bv, &two, zero, cg, &one, 1, 1);
    vmb_stats_get(after);
    vmb_cpu_zgemm('N', 'N', 1, 1, 2, alpha, av, 1, bv, 2, zero, cc, 1);
#define D(i) (after[i] - before[i])
    unsigned long long range = D(VMB_STAT_REASON_BASE + VMB_DECLINED_RANGE),
                       sub = D(VMB_STAT_GPU_SUBMITTED), used = D(VMB_STAT_CONSUMED),
                       cpu = D(VMB_STAT_CPU_CALLS);
#undef D
    cases++;
    if (want_range) {
        expect(st == VMB_DECLINED_RANGE, label, "direct try did not return range");
        expect(memcmp(ct, sentinel, sizeof ct) == 0, label, "direct try wrote to C");
        expect(range == 1 && sub == 0 && used == 0 && cpu == 1, label,
               "zgemm_ counter deltas are not range=1 submitted=0 consumed=0 cpu=1");
    } else {
        expect(st == VMB_OK, label, "direct try did not consume a GPU result");
        expect(range == 0 && sub == 1 && used == 1 && cpu == 0, label,
               "zgemm_ counter deltas are not range=0 submitted=1 consumed=1 cpu=0");
    }
    expect(memcmp(cg, cc, sizeof cg) == 0, label, "zgemm_ result differs from OpenBLAS");
    printf("%-28s try=%d; zgemm_ deltas range=%llu submitted=%llu consumed=%llu cpu=%llu; "
           "C=(%a, %a) %s OpenBLAS (%a, %a)\n", label, st, range, sub, used, cpu, cg[0], cg[1],
           memcmp(cg, cc, sizeof cg) == 0 ? "==" : "!=", cc[0], cc[1]);
}

static void overflow_cases(void)
{
    set_env("VMB_MODE", "force");
    set_env("VMB_CHECK", "0");
    /* Each unscaled product is 2^1040; alpha * (A B) is exactly 2^541. */
    overflow_case("overflow: products 2^1040", 1, 0x1p-500,
                  (const double[2]){0x1p600, 0x1p440}, (const double[2]){0x1p440, 0x1p600});
    /* Each product is 2^1023 (finite); their sum is 2^1024. Exact result 2^524. */
    overflow_case("overflow: partial sum 2^1024", 1, 0x1p-500,
                  (const double[2]){0x1p511, 0x1p511}, (const double[2]){0x1p512, 0x1p512});
    /* Positive controls: the same shape and alpha with representable sums. */
    overflow_case("overflow ctl: alpha 2^-500", 0, 0x1p-500,
                  (const double[2]){0x1p10, 0x1p-3}, (const double[2]){0x1p5, 0x1p7});
    overflow_case("overflow ctl: alpha 1", 0, 1.0,
                  (const double[2]){1.5, -2.0}, (const double[2]){0.25, 3.0});
    set_env("VMB_CHECK", NULL);
    set_env("VMB_MODE", NULL);
}

/* Route check through zgemm_: expected outcome counted once, result equal to
   OpenBLAS whenever the CPU served the call. */
static void mode_case(const char *label, int expect_status, int use_diag)
{
    const double one[2] = {1, 0}, zero[2] = {0, 0};
    struct mat a = mat_new(9, 7, 1), b = mat_new(7, 8, 0), c = mat_new(9, 8, 1);
    fill(&a, RANDOM), fill(&b, RANDOM), fill(&c, RANDOM);
    if (use_diag) {
        set_op(&a, 'N', 0, 0, 0x1p90, 0), set_op(&a, 'N', 1, 1, 0x1p-90, 0);
        set_op(&b, 'N', 0, 0, 0x1p-90, 0), set_op(&b, 'N', 1, 1, 0x1p90, 0);
    }
    struct mat cg = mat_copy(&c), cc = mat_copy(&c);
    uint64_t before[VMB_NSTAT], after[VMB_NSTAT];
    vmb_stats_get(before);
    call('N', 'N', 9, 8, 7, one, &a, &b, zero, &cg, a.ld);
    vmb_stats_get(after);
    vmb_cpu_zgemm('N', 'N', 9, 8, 7, one, a.v, a.ld, b.v, b.ld, zero, cc.v, cc.ld);
    cases++;
    expect(after[VMB_STAT_REASON_BASE + expect_status] - before[VMB_STAT_REASON_BASE + expect_status] == 1,
           label, "unexpected outcome");
    if (expect_status != VMB_OK)
        expect(memcmp(cg.v, cc.v, mat_bytes(&c)) == 0, label, "CPU-served result differs");
    printf("%-28s mode=%d outcome=%d as expected\n", label, vmb_mode(), expect_status);
    free(a.v), free(b.v), free(c.v), free(cg.v), free(cc.v);
}

static void mode_cases(void)
{
    set_env("VMB_ENABLE", NULL);
    set_env("VMB_MODE", "auto");
    mode_case("mode: auto, no region", VMB_DECLINED_AUTO, 0);
    set_env("VMB_TEST_AUTO_MIN_WORK", "0");
    mode_case("mode: auto, test region", VMB_OK, 0);
    mode_case("mode: auto cannot bypass", VMB_DECLINED_RANGE, 1);
    set_env("VMB_TEST_AUTO_MIN_WORK", NULL);
    set_env("VMB_MODE", "force");
    mode_case("mode: force", VMB_OK, 0);
    mode_case("mode: force cannot bypass", VMB_DECLINED_RANGE, 1);
    set_env("VMB_MODE", "bogus");
    mode_case("mode: unknown -> off", VMB_DECLINED_DISABLED, 0);
    set_env("VMB_MODE", NULL);
    mode_case("mode: default off", VMB_DECLINED_DISABLED, 0);
    set_env("VMB_ENABLE", "1");
    mode_case("mode: VMB_ENABLE=1 -> force", VMB_OK, 0);
}

static void telemetry_case(const char *path)
{
    vmb_telemetry_write(path);
    FILE *in = fopen(path, "r");
    char line[512];
    int rows = 0, found = 0;
    while (in && fgets(line, sizeof line, in)) {
        if (line[0] == '#' || strncmp(line, "m\t", 2) == 0)
            continue;
        rows++;
        /* mode_case shape: 9x8x7 N N lda=10 ldb=7 ldc=10 beta zero, status ok */
        static const char key[] = "9\t8\t7\tN\tN\t10\t7\t10\tzero\t0\t";
        if (strncmp(line, key, sizeof key - 1) == 0)
            found = 1;
    }
    if (in)
        fclose(in);
    cases++;
    expect(rows > 0 && found, "telemetry", "aggregated shape row missing");
    printf("%-28s %d aggregated rows, known shape %s\n", "telemetry", rows,
           found ? "present" : "MISSING");
}

int main(void)
{
    static const double alphas[4][2] = {{1, 0}, {0.5, -2}, {-1.25, 0.75}, {0, 1}};
    static const double betas[4][2] = {{0, 0}, {1, 0}, {-0.3, 1.1}, {0, -1}};
    static const int sizes[][3] = {{1, 1, 1},    {2, 3, 4},     {5, 7, 3},
                                   {17, 13, 29}, {31, 33, 1},   {64, 48, 31},
                                   {127, 129, 5}, {9, 1, 257},  {1, 40, 64}};
    static const char upper[] = "NTC", lower[] = "ntc";
    char label[64];

    set_env("VMB_MODE", NULL);
    set_env("VMB_TELEMETRY", "build/test-telemetry.tsv");
    set_env("VMB_ENABLE", "1");
    set_env("VMB_MIN_WORK", NULL);
    set_env("VMB_CHECK", NULL);
    set_env("VMB_TEST_NO_DEVICE", NULL);
    set_env("VMB_TEST_FAULT", NULL);
    printf("cpu library: %s\n", vmb_cpu_library());
    printf("tolerance: |err| <= %.1e*S_max + 8(k+2)2^-48*S_ij\n", VMB_ATOL(1.0));

    for (size_t s = 0; s < sizeof sizes / sizeof sizes[0]; s++)
        for (int x = 0; x < 3; x++)
            for (int y = 0; y < 3; y++) {
                int combo = (int)(s + x + y) % 4;
                char ta = (s + x) % 2 ? lower[x] : upper[x], tb = (s + y) % 2 ? upper[y] : lower[y];
                snprintf(label, sizeof label, "sweep %c%c ab%d", ta, tb, combo);
                gpu_case(label, ta, tb, sizes[s][0], sizes[s][1], sizes[s][2], (int)(s % 3),
                         alphas[combo], betas[combo], RANDOM);
            }
    if (vmb_device_name())
        printf("device: %s\n", vmb_device_name());

    gpu_case("wide-range 1e+-30 NN", 'N', 'N', 33, 17, 41, 1, alphas[2], betas[2], WIDE);
    gpu_case("wide-range 1e+-30 CT", 'C', 'T', 33, 17, 41, 2, alphas[1], betas[3], WIDE);
    gpu_case("tiny 1e-150", 'T', 'N', 20, 21, 22, 1, alphas[1], betas[2], TINY);
    gpu_case("huge 1e+120", 'N', 'C', 20, 21, 22, 1, alphas[2], betas[1], HUGE_VALS);
    cancel_case("cancellation NN", 'N', 'N', 19, 23, 64);
    cancel_case("cancellation CT", 'C', 'T', 19, 23, 64);
    cancel_case("cancellation TC", 'T', 'C', 7, 5, 200);
    determinism_case();

    set_env("VMB_CHECK", "1");
    gpu_case("shadow check mode", 'C', 'N', 24, 16, 40, 1, alphas[1], betas[2], RANDOM);
    printf("%-28s max shadow-check err/tol=%.3e\n", "shadow check mode", vmb_stats_max_check_ratio());
    set_env("VMB_CHECK", NULL);

    const double one[2] = {1, 0}, zero[2] = {0, 0}, some[2] = {0.5, -2};
    fallback_case("fallback: nan in A", VMB_DECLINED_NONFINITE, 'N', 'T', 6, 5, 4, one, some, 0, 1, 0);
    fallback_case("fallback: k == 0", VMB_DECLINED_TRIVIAL, 'N', 'N', 6, 5, 0, one, some, 0, 0, 0);
    fallback_case("fallback: m == 0", VMB_DECLINED_TRIVIAL, 'N', 'N', 0, 5, 3, one, some, 0, 0, 0);
    fallback_case("fallback: alpha == 0", VMB_DECLINED_TRIVIAL, 'C', 'N', 6, 5, 3, zero, some, 0, 0, 0);
    fallback_case("fallback: lda < m (badarg)", VMB_DECLINED_BADARG, 'N', 'N', 6, 5, 3, one, some, 2, 0, 0);
    set_env("VMB_MIN_WORK", "1e9");
    fallback_case("fallback: below min work", VMB_DECLINED_TOO_SMALL, 'N', 'N', 6, 5, 4, one, some, 0, 0, 0);
    set_env("VMB_MIN_WORK", NULL);
    set_env("VMB_TEST_NO_DEVICE", "1");
    fallback_case("fallback: no device", VMB_DECLINED_NO_DEVICE, 'T', 'C', 6, 5, 4, one, some, 0, 0, 0);
    set_env("VMB_TEST_NO_DEVICE", NULL);
    set_env("VMB_TEST_FAULT", "gpu_error");
    fallback_case("fallback: gpu error", VMB_FAILED_GPU, 'N', 'C', 6, 5, 4, one, some, 0, 0, 1);
    set_env("VMB_TEST_FAULT", "gpu_nonfinite");
    fallback_case("fallback: gpu nonfinite", VMB_FAILED_GPU_NONFINITE, 'C', 'C', 6, 5, 4, one, some, 0, 0, 1);
    set_env("VMB_TEST_FAULT", NULL);
    set_env("VMB_ENABLE", NULL);
    fallback_case("fallback: disabled", VMB_DECLINED_DISABLED, 'N', 'N', 6, 5, 4, one, some, 0, 0, 0);
    set_env("VMB_ENABLE", "1");

    tolerance_units();
    safety_cases();
    overflow_cases();
    mode_cases();
    telemetry_case("build/test-telemetry.tsv");
    latency_info();
    printf("\nworst err/tol over GPU cases: %.3e   worst max|err|/S: %.3e\n", worst_ratio, worst_rel);
    vmb_report(stdout);
    printf("%s: %d cases, %d failures\n", failures ? "FAIL" : "PASS", cases,
           failures);
    return failures ? 1 : 0;
}
