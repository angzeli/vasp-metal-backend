/* Policy, safety envelope, counters, fallback and the interposed zgemm_. */
#define _DARWIN_C_SOURCE
#include "vmb.h"
#include "vmb_internal.h"

#include <dlfcn.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef void (*zgemm_fn)(const char *, const char *, const int *, const int *,
                         const int *, const double *, const double *, const int *,
                         const double *, const int *, const double *, double *,
                         const int *, size_t, size_t);

enum { FAULT_NONE, FAULT_GPU_ERROR, FAULT_GPU_NONFINITE, FAULT_GPU_PERTURB };

/*
 * Exponent-range safety envelope (docs/PRECISION.md). In scaled units every
 * packed component and every partial sum is bounded, and each float operation
 * that underflows contributes an absolute error of at most 2^-126 even with
 * flush-to-zero. A generous bound of 2^ENV_TERM_EXP per k-term (plus two for
 * the epilogue) covers all such errors; scaled back by 2^so it must stay below
 * half of the absolute tolerance computed from a cheap lower bound of S_max.
 * ENV_SO_MAX bounds the exponent of the scaled output and, through cpu_exp(),
 * of every unscaled intermediate the CPU path may form, so both paths stay far
 * from FP64 overflow and agree on its semantics. ENV_LOWER_MIN keeps results
 * out of the FP64 subnormal range.
 */
#define ENV_TERM_EXP (-110)
#define ENV_SO_MAX 1000
#define ENV_LOWER_MIN 0x1p-900

/* Shape regions where forced Metal showed a reproducible end-to-end win over
   single-thread OpenBLAS. None has been established (docs/RESULTS-v0.2.md), so
   auto mode currently serves every call on the CPU. */
struct region {
    int min_m, min_n, min_k;
};
static const struct region auto_regions[1] = {{0, 0, 0}};
static const int n_auto_regions = 0;

static struct {
    int mode, check, no_device, fault, telemetry_rank;
    double min_work, max_work, max_bytes, test_auto_min_work;
    char report[1024], telemetry[1024];
} cfg;

static _Atomic uint64_t stats[VMB_NSTAT];
/* zgemm_ calls by decade of m*n*k, and time in the GPU attempt / CPU path. */
static _Atomic uint64_t work_hist[16], try_ns, cpu_ns;
static double max_check_ratio;        /* guarded by gpu_lock */
static struct vmb_timing last_timing; /* guarded by gpu_lock */
static pthread_mutex_t gpu_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_once_t init_once = PTHREAD_ONCE_INIT;
static zgemm_fn cpu_zgemm;
static char cpu_library[1024];

static const char *const status_names[VMB_NSTATUS] = {
    "ok", "disabled", "trivial", "badarg", "nonfinite", "too_small", "too_large",
    "no_device", "gpu_error", "gpu_nonfinite", "check_failed", "range", "auto_cpu",
};
static const char *const mode_names[3] = {"off", "auto", "force"};

double vmb_now(void) { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW) * 1e-9; }

static double env_double(const char *name, double fallback)
{
    const char *s = getenv(name);
    if (!s || !*s)
        return fallback;
    char *end;
    double v = strtod(s, &end);
    if (*end != '\0' || !(v >= 0)) {
        fprintf(stderr, "vmb: ignoring invalid %s=%s\n", name, s);
        return fallback;
    }
    return v;
}

static int env_is(const char *name, const char *value)
{
    const char *s = getenv(name);
    return s && strcmp(s, value) == 0;
}

static int env_mode(void)
{
    const char *s = getenv("VMB_MODE");
    if (!s || !*s)  /* v0.1 compatibility: VMB_ENABLE=1 meant every eligible call */
        return env_is("VMB_ENABLE", "1") ? VMB_MODE_FORCE : VMB_MODE_OFF;
    for (int m = 0; m < 3; m++)
        if (strcmp(s, mode_names[m]) == 0)
            return m;
    fprintf(stderr, "vmb: unknown VMB_MODE=%s; using off\n", s);
    return VMB_MODE_OFF;
}

static void write_at_exit(void)
{
    if (cfg.telemetry[0])
        vmb_telemetry_write(cfg.telemetry);
    if (!cfg.report[0])
        return;
    FILE *out = strcmp(cfg.report, "stderr") == 0 ? stderr : fopen(cfg.report, "w");
    if (!out) {
        fprintf(stderr, "vmb: cannot write report to %s\n", cfg.report);
        return;
    }
    vmb_report(out);
    if (out != stderr)
        fclose(out);
}

void vmb_config_reload(void)
{
    cfg.mode = env_mode();
    cfg.check = env_is("VMB_CHECK", "1");
    cfg.no_device = env_is("VMB_TEST_NO_DEVICE", "1");
    cfg.fault = env_is("VMB_TEST_FAULT", "gpu_error")       ? FAULT_GPU_ERROR
                : env_is("VMB_TEST_FAULT", "gpu_nonfinite") ? FAULT_GPU_NONFINITE
                : env_is("VMB_TEST_FAULT", "gpu_perturb")   ? FAULT_GPU_PERTURB
                                                            : FAULT_NONE;
    cfg.min_work = env_double("VMB_MIN_WORK", 0.0);
    cfg.max_work = env_double("VMB_MAX_WORK", 0x1p34);
    cfg.max_bytes = env_double("VMB_MAX_BYTES", 0x1p30);
    cfg.test_auto_min_work = env_double("VMB_TEST_AUTO_MIN_WORK", -1.0);
    const char *report = getenv("VMB_REPORT"), *telemetry = getenv("VMB_TELEMETRY");
    /* "rank" selects per-rank telemetry in VMB_REPORT_DIR, never a file name;
       the report constructor rejects it without that directory. */
    cfg.telemetry_rank = telemetry && strcmp(telemetry, "rank") == 0;
    snprintf(cfg.report, sizeof cfg.report, "%s", report ? report : "");
    snprintf(cfg.telemetry, sizeof cfg.telemetry, "%s",
             telemetry && !cfg.telemetry_rank ? telemetry : "");
}

/* zgemm_ exported by the loaded OpenBLAS image. RTLD_NEXT is not used: the
   Metal and Foundation frameworks load Accelerate, whose BLAS would win. */
static zgemm_fn find_openblas_zgemm(void)
{
    for (uint32_t i = 0; i < _dyld_image_count(); i++) {
        const char *name = _dyld_get_image_name(i);
        const char *base = strrchr(name, '/');
        if (strncmp(base ? base + 1 : name, "libopenblas", 11) != 0)
            continue;
        void *handle = dlopen(name, RTLD_LAZY | RTLD_NOLOAD);
        void *fn = handle ? dlsym(handle, "zgemm_") : NULL;
        Dl_info info;
        if (fn && fn != (void *)zgemm_ && dladdr(fn, &info) && strcmp(info.dli_fname, name) == 0) {
            snprintf(cpu_library, sizeof cpu_library, "%s", name);
            return (zgemm_fn)fn;
        }
    }
    return NULL;
}

static void init(void)
{
    /* Without the original routine there is no safe fallback; resolving our
       own symbol would recurse. Both are fatal. */
    cpu_zgemm = find_openblas_zgemm();
    if (!cpu_zgemm) {
        fprintf(stderr, "vmb: OpenBLAS zgemm_ not found among loaded images; "
                        "link libvmb.a ahead of -lopenblas\n");
        abort();
    }
    vmb_config_reload();
    if (cfg.report[0] || cfg.telemetry[0])
        atexit(write_at_exit);
}

static int trans_code(char t)
{
    switch (t) {
    case 'N': case 'n': return 0;
    case 'T': case 't': return 1;
    case 'C': case 'c': return 2;
    default: return -1;
    }
}

static int imax(int a, int b) { return a > b ? a : b; }

/* Returns 0 if a stored element is non-finite. *maxabs, and the optional
   per-column / per-row maxima, use max(|re|, |im|). */
static int scan(const double *x, int rows, int cols, int ld, double *colmax, double *rowmax,
                double *maxabs)
{
    double mx = 0.0;
    for (int j = 0; j < cols; j++) {
        const double *col = x + 2 * (size_t)j * ld;
        double cmx = 0.0;
        for (int i = 0; i < rows; i++) {
            double v = fmax(fabs(col[2 * i]), fabs(col[2 * i + 1]));
            if (!isfinite(col[2 * i]) || !isfinite(col[2 * i + 1]))
                return 0;
            if (v > cmx)
                cmx = v;
            if (rowmax && v > rowmax[i])
                rowmax[i] = v;
        }
        if (colmax)
            colmax[j] = cmx;
        if (cmx > mx)
            mx = cmx;
    }
    *maxabs = mx;
    return 1;
}

/* Exponent e with v * 2^-e in [0.5, 1); 0 for v == 0. */
static int scale_exp(double v) { return v > 0 ? ilogb(v) + 1 : 0; }

/*
 * Overflow-safe exponent bound, from integers only, on every unscaled value
 * the CPU path may form; each magnitude is < 2^e for the e listed:
 *   real product x*y of op(A)_il, op(B)_lj components    ep = max_l(ea_l + eb_l)
 *   any partial sum of the 2k real products making up
 *   a real or imaginary part, in any order (< 2k 2^ep)    es = ep + kbits + 1 (margin)
 *   alpha applied to that sum (complex multiply), or
 *   first to an operand as reference BLAS does            ealpha + max(es, sa, sb) + 1
 *   beta*C                                                ebeta + sc + 1
 *   the final sum of the alpha and beta terms             max of both + 1
 * ea_l, eb_l, sa, sb, sc, ealpha, ebeta are scale_exp() of the corresponding
 * component maxima, so each value is strictly below 2^e. The unscaled sum
 * itself is bounded independently of alpha: a tiny alpha cannot hide it.
 */
static int cpu_exp(int ep, int kbits, int ealpha, int sa, int sb, int beta_term, int ebeta,
                   int sc)
{
    int es = ep + kbits + 1;
    int e_alpha = ealpha + imax(es, imax(sa, sb)) + 1;
    int e = imax(es, e_alpha);
    if (beta_term)
        e = imax(e, imax(e_alpha, ebeta + sc + 1) + 1);
    return e;
}

static int range_safe(int k, int so, int cpu_e, double lower)
{
    /* lower is +inf when it could not be formed safely: never permissive. */
    if (so > ENV_SO_MAX || cpu_e > ENV_SO_MAX || !isfinite(lower) || !(lower >= ENV_LOWER_MIN))
        return 0;
    return ldexp(k + 2.0, so + ENV_TERM_EXP) <= 0.5 * VMB_ATOL(lower);
}

static void op_elem(const double *x, int ld, int trans, int row, int col, double *re, double *im)
{
    const double *z = trans ? x + 2 * ((size_t)col + (size_t)row * ld)
                            : x + 2 * ((size_t)row + (size_t)col * ld);
    *re = z[0];
    *im = trans == 2 ? -z[1] : z[1];
}

double vmb_tolerance_ratio(size_t count, const double *got, const double *ref,
                           const double *scale, int k)
{
    double smax = 0.0;
    for (size_t e = 0; e < count; e++) {
        if (!isfinite(scale[e]) || scale[e] < 0.0)
            return INFINITY;
        smax = fmax(smax, scale[e]);
    }
    double atol = VMB_ATOL(smax), rtol = VMB_RTOL(k), worst = 0.0;
    for (size_t e = 0; e < count; e++) {
        const double *g = got + 2 * e, *r = ref + 2 * e;
        if (!isfinite(g[0]) || !isfinite(g[1]) || !isfinite(r[0]) || !isfinite(r[1]))
            return INFINITY;
        double err = hypot(g[0] - r[0], g[1] - r[1]);
        if (!isfinite(err))
            return INFINITY;
        if (err == 0.0)
            continue;  /* exact agreement passes even with zero tolerance */
        double tol = atol + rtol * scale[e];
        if (!(tol > 0.0))
            return INFINITY;
        double q = err / tol;
        if (!(q <= worst))
            worst = q;  /* q is never NaN here; inf propagates */
    }
    return worst;
}

/* err/tol of the GPU result against OpenBLAS on the same inputs. */
static double shadow_check(int ta, int tb, int m, int n, int k, const double *alpha,
                           const double *a, int lda, const double *b, int ldb,
                           const double *beta, const double *c, int ldc, int beta_zero,
                           const double *out)
{
    static const char flags[3] = {'N', 'T', 'C'};
    size_t mn = (size_t)m * n;
    double *ref = calloc(2 * mn, sizeof *ref), *s = malloc(mn * sizeof *s);
    if (!ref || !s) {
        free(ref);
        free(s);
        return INFINITY;
    }
    for (int j = 0; j < n && !beta_zero; j++)
        memcpy(ref + 2 * (size_t)j * m, c + 2 * (size_t)j * ldc, 2 * (size_t)m * sizeof *ref);
    cpu_zgemm(&flags[ta], &flags[tb], &m, &n, &k, alpha, a, &lda, b, &ldb, beta, ref, &m, 1, 1);

    double aa = hypot(alpha[0], alpha[1]), ab = hypot(beta[0], beta[1]);
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++) {
            double acc = 0.0, xr, xi, yr, yi;
            for (int l = 0; l < k; l++) {
                op_elem(a, lda, ta, i, l, &xr, &xi);
                op_elem(b, ldb, tb, l, j, &yr, &yi);
                acc += hypot(xr, xi) * hypot(yr, yi);
            }
            double sij = aa * acc;
            if (!beta_zero)
                sij += ab * hypot(c[2 * (i + (size_t)j * ldc)], c[2 * (i + (size_t)j * ldc) + 1]);
            s[i + (size_t)j * m] = sij;
        }
    double ratio = vmb_tolerance_ratio(mn, out, ref, s, k);
    free(ref);
    free(s);
    return ratio;
}

static int try_gpu(char transa, char transb, int m, int n, int k, const double alpha[2],
                   const double *a, int lda, const double *b, int ldb,
                   const double beta[2], double *c, int ldc)
{
    double t_start = vmb_now();
    int ta = trans_code(transa), tb = trans_code(transb);
    int rows_a = ta ? k : m, cols_a = ta ? m : k;
    int rows_b = tb ? n : k, cols_b = tb ? k : n;
    if (ta < 0 || tb < 0 || m < 0 || n < 0 || k < 0 || lda < imax(1, rows_a) ||
        ldb < imax(1, rows_b) || ldc < imax(1, m))
        return VMB_DECLINED_BADARG;
    if (m == 0 || n == 0 || k == 0 || (alpha[0] == 0.0 && alpha[1] == 0.0))
        return VMB_DECLINED_TRIVIAL;
    if (!isfinite(alpha[0]) || !isfinite(alpha[1]) || !isfinite(beta[0]) || !isfinite(beta[1]))
        return VMB_DECLINED_NONFINITE;
    double work = (double)m * n * k;
    double bytes = 16.0 * ((double)m * k + (double)k * n + 2.0 * m * n) + 16.0 * m * n;
    if (work < cfg.min_work)
        return VMB_DECLINED_TOO_SMALL;
    if (work > cfg.max_work || bytes > cfg.max_bytes)
        return VMB_DECLINED_TOO_LARGE;

    /* lmax[l]: max over i of op(A)_il; lmax[k + l]: max over j of op(B)_lj. */
    double *lmax = calloc(2 * (size_t)k, sizeof *lmax);
    if (!lmax)
        return VMB_DECLINED_TOO_LARGE;
    int beta_zero = beta[0] == 0.0 && beta[1] == 0.0;  /* BLAS: C is not read */
    double ma, mb, mc = 0.0;
    int finite = scan(a, rows_a, cols_a, lda, ta ? NULL : lmax, ta ? lmax : NULL, &ma) &&
                 scan(b, rows_b, cols_b, ldb, tb ? lmax + k : NULL, tb ? NULL : lmax + k, &mb) &&
                 (beta_zero || scan(c, m, n, ldc, NULL, NULL, &mc));
    if (!finite) {
        free(lmax);
        return VMB_DECLINED_NONFINITE;
    }
    int ep = INT_MIN / 4;  /* no nonzero product */
    for (int l = 0; l < k; l++)
        if (lmax[l] > 0.0 && lmax[k + l] > 0.0)
            ep = imax(ep, scale_exp(lmax[l]) + scale_exp(lmax[k + l]));

    /* Power-of-two scaling keeps packed values in [-1, 1] and the output near
       unit magnitude; the envelope then bounds what scaling cannot fix. */
    struct vmb_gpu_job job = {
        .ta = ta, .tb = tb, .m = m, .n = n, .k = k,
        .a = a, .lda = lda, .rows_a = rows_a, .cols_a = cols_a, .sa = scale_exp(ma),
        .b = b, .ldb = ldb, .rows_b = rows_b, .cols_b = cols_b, .sb = scale_exp(mb),
        .c = c, .ldc = ldc, .beta_zero = beta_zero, .sc = scale_exp(mc),
    };
    double amax = fmax(fabs(alpha[0]), fabs(alpha[1])), bmax = fmax(fabs(beta[0]), fabs(beta[1]));
    int kbits = ilogb((double)k) + 2;  /* 2k <= 2^kbits bounds the sum */
    int beta_term = !beta_zero && mc > 0.0;
    job.so = scale_exp(amax) + 1 + job.sa + job.sb + kbits;
    if (beta_term)
        job.so = imax(job.so, scale_exp(bmax) + 1 + job.sc);
    int cpu_e = cpu_exp(ep, kbits, scale_exp(amax), job.sa, job.sb, beta_term, scale_exp(bmax),
                        job.sc);
    /* lower bound of max_ij S_ij; formed only once cpu_e shows that every
       product below is < 2^ENV_SO_MAX, so nothing here can overflow. */
    double lower = INFINITY;
    if (cpu_e <= ENV_SO_MAX) {
        double prod = 0.0;
        for (int l = 0; l < k; l++)
            prod = fmax(prod, lmax[l] * lmax[k + l]);
        lower = amax * prod;
        if (beta_term)
            lower = fmax(lower, bmax * mc);
    }
    free(lmax);
    if (!range_safe(k, job.so, cpu_e, lower))
        return VMB_DECLINED_RANGE;
    if (!vmb_gpu_available(cfg.no_device))
        return VMB_DECLINED_NO_DEVICE;
    for (int p = 0; p < 2; p++) {
        job.alpha_s[p] = ldexp(alpha[p], job.sa + job.sb - job.so);
        job.beta_s[p] = beta_zero ? 0.0 : ldexp(beta[p], job.sc - job.so);
    }
    size_t mn = (size_t)m * n;
    job.out = malloc(2 * mn * sizeof *job.out);
    if (!job.out)
        return VMB_DECLINED_TOO_LARGE;

    pthread_mutex_lock(&gpu_lock);
    int submitted = 0, status = VMB_OK;
    int rc = vmb_gpu_zgemm(&job, &submitted);
    if (submitted)
        stats[VMB_STAT_GPU_SUBMITTED]++;
    if (rc == 0 && cfg.fault == FAULT_GPU_ERROR)
        rc = -1;
    double t_finish = vmb_now();
    if (rc != 0) {
        status = VMB_FAILED_GPU;
        goto done;
    }
    stats[VMB_STAT_GPU_COMPLETED]++;
    if (cfg.fault == FAULT_GPU_NONFINITE)
        job.out[2 * mn - 1] = NAN;
    /* Validate everything before the first write: C is either fully updated
       from the GPU result or left untouched for the CPU path. */
    for (size_t e = 0; e < 2 * mn; e++)
        if (!isfinite(job.out[e])) {
            status = VMB_FAILED_GPU_NONFINITE;
            goto done;
        }
    if (cfg.fault == FAULT_GPU_PERTURB)
        job.out[0] += 1.0 + fabs(job.out[0]);
    if (cfg.check) {
        double r = shadow_check(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                                beta_zero, job.out);
        stats[VMB_STAT_CHECKED]++;
        if (!(r <= max_check_ratio))
            max_check_ratio = r;
        if (!(r <= 1.0)) {
            status = VMB_FAILED_CHECK;
            goto done;
        }
    }
    for (int j = 0; j < n; j++)
        memcpy(c + 2 * (size_t)j * ldc, job.out + 2 * (size_t)j * m, 2 * (size_t)m * sizeof *c);
    stats[VMB_STAT_CONSUMED]++;
done:
    last_timing = (struct vmb_timing){
        .pack = job.t_pack, .encode = job.t_encode, .gpu = job.t_gpu, .wait = job.t_wait,
        .unpack = job.t_unpack, .finish = vmb_now() - t_finish, .total = vmb_now() - t_start,
    };
    pthread_mutex_unlock(&gpu_lock);
    free(job.out);
    return status;
}

int vmb_zgemm_try(char transa, char transb, int m, int n, int k, const double alpha[2],
                  const double *a, int lda, const double *b, int ldb,
                  const double beta[2], double *c, int ldc)
{
    pthread_once(&init_once, init);
    int status = try_gpu(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    stats[VMB_STAT_REASON_BASE + status]++;
    return status;
}

static int auto_selects(int m, int n, int k)
{
    if (cfg.test_auto_min_work >= 0.0)  /* test hook: exercises the auto route */
        return (double)m * n * k >= cfg.test_auto_min_work;
    for (int r = 0; r < n_auto_regions; r++)
        if (m >= auto_regions[r].min_m && n >= auto_regions[r].min_n && k >= auto_regions[r].min_k)
            return 1;
    return 0;
}

void zgemm_(const char *transa, const char *transb, const int *m, const int *n,
            const int *k, const double *alpha, const double *a, const int *lda,
            const double *b, const int *ldb, const double *beta, double *c,
            const int *ldc, size_t len_transa, size_t len_transb)
{
    pthread_once(&init_once, init);
    stats[VMB_STAT_CALLS]++;
    double work = (double)*m * *n * *k;
    work_hist[work < 10.0 ? 0 : (int)fmin(15.0, floor(log10(work)))]++;
    uint64_t t0 = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
    int status;
    if (cfg.mode == VMB_MODE_FORCE ||
        (cfg.mode == VMB_MODE_AUTO && auto_selects(*m, *n, *k))) {
        status = vmb_zgemm_try(*transa, *transb, *m, *n, *k, alpha, a, *lda, b, *ldb,
                               beta, c, *ldc);
    } else {
        status = cfg.mode == VMB_MODE_AUTO ? VMB_DECLINED_AUTO : VMB_DECLINED_DISABLED;
        stats[VMB_STAT_REASON_BASE + status]++;
    }
    uint64_t t1 = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
    try_ns += t1 - t0;
    if (status != VMB_OK) {
        stats[VMB_STAT_CPU_CALLS]++;
        cpu_zgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                  len_transa, len_transb);
    }
    uint64_t t2 = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
    if (status != VMB_OK)
        cpu_ns += t2 - t1;
    if (cfg.telemetry[0] || (cfg.telemetry_rank && vmb_rank_telemetry())) {
        struct vmb_shape shape = {
            .m = *m, .n = *n, .k = *k, .lda = *lda, .ldb = *ldb, .ldc = *ldc,
            .ta = *transa, .tb = *transb,
            .beta_class = beta[0] == 0.0 && beta[1] == 0.0   ? 0
                          : beta[0] == 1.0 && beta[1] == 0.0 ? 1
                                                             : 2,
        };
        vmb_telemetry_record(&shape, status, (t2 - t0) * 1e-9);
    }
}

void vmb_cpu_zgemm(char transa, char transb, int m, int n, int k, const double alpha[2],
                   const double *a, int lda, const double *b, int ldb,
                   const double beta[2], double *c, int ldc)
{
    pthread_once(&init_once, init);
    cpu_zgemm(&transa, &transb, &m, &n, &k, alpha, a, &lda, b, &ldb, beta, c, &ldc, 1, 1);
}

int vmb_device_init(double *seconds)
{
    pthread_once(&init_once, init);
    double t0 = vmb_now();
    int ok = vmb_gpu_available(cfg.no_device);
    if (seconds)
        *seconds = vmb_now() - t0;
    return ok;
}

int vmb_mode(void)
{
    pthread_once(&init_once, init);
    return cfg.mode;
}

void vmb_last_timing(struct vmb_timing *out)
{
    pthread_mutex_lock(&gpu_lock);
    *out = last_timing;
    pthread_mutex_unlock(&gpu_lock);
}

void vmb_stats_get(uint64_t out[VMB_NSTAT])
{
    for (int i = 0; i < VMB_NSTAT; i++)
        out[i] = stats[i];
}

void vmb_stats_reset(void)
{
    pthread_mutex_lock(&gpu_lock);
    for (int i = 0; i < VMB_NSTAT; i++)
        stats[i] = 0;
    for (int i = 0; i < 16; i++)
        work_hist[i] = 0;
    try_ns = cpu_ns = 0;
    max_check_ratio = 0.0;
    pthread_mutex_unlock(&gpu_lock);
}

double vmb_stats_max_check_ratio(void) { return max_check_ratio; }

const char *vmb_device_name(void) { return vmb_gpu_device_name(); }

const char *vmb_mode_name(int mode) { return mode_names[mode]; }

const char *vmb_status_name(int status) { return status_names[status]; }

/* Counters are read under gpu_lock so that a GPU attempt is seen whole. */
void vmb_snapshot_get(struct vmb_snapshot *out)
{
    pthread_mutex_lock(&gpu_lock);
    out->mode = cfg.mode;
    out->check = cfg.check;
    for (int i = 0; i < VMB_NSTAT; i++)
        out->stats[i] = stats[i];
    out->max_check_ratio = max_check_ratio;
    out->cpu_library = cpu_library;
    pthread_mutex_unlock(&gpu_lock);
}

const char *vmb_cpu_library(void)
{
    pthread_once(&init_once, init);
    return cpu_library;
}

void vmb_report(FILE *out)
{
    static const char *const names[VMB_STAT_REASON_BASE] = {
        "calls", "gpu_submitted", "gpu_completed", "consumed", "cpu_calls", "checked",
    };
    const char *dev = vmb_gpu_device_name();
    const char *err = vmb_gpu_init_error();
    fprintf(out, "vmb.version=0.3\nvmb.mode=%s\nvmb.enabled=%d\nvmb.check=%d\nvmb.min_work=%.17g\n",
            mode_names[cfg.mode], cfg.mode != VMB_MODE_OFF, cfg.check, cfg.min_work);
    fprintf(out, "vmb.device=%s\n", dev ? dev : "none");
    if (err && *err)
        fprintf(out, "vmb.device_error=%s\n", err);
    fprintf(out, "vmb.cpu_library=%s\n", cpu_library);
    for (int i = 0; i < VMB_STAT_REASON_BASE; i++)
        fprintf(out, "vmb.%s=%llu\n", names[i], (unsigned long long)stats[i]);
    for (int s = 0; s < VMB_NSTATUS; s++)
        fprintf(out, "vmb.status.%s=%llu\n", status_names[s],
                (unsigned long long)stats[VMB_STAT_REASON_BASE + s]);
    fprintf(out, "vmb.max_check_ratio=%.6e\n", max_check_ratio);
    fprintf(out, "vmb.try_seconds=%.6f\nvmb.cpu_seconds=%.6f\n", try_ns * 1e-9, cpu_ns * 1e-9);
    for (int b = 0; b < 16; b++)
        if (work_hist[b])
            fprintf(out, "vmb.calls_work_1e%d=%llu\n", b, (unsigned long long)work_hist[b]);
}
