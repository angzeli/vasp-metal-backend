/* Policy, counters, fallback and the interposed zgemm_ entry point. */
#define _DARWIN_C_SOURCE
#include "vmb.h"
#include "vmb_internal.h"

#include <dlfcn.h>
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

enum { FAULT_NONE, FAULT_GPU_ERROR, FAULT_GPU_NONFINITE };

static struct {
    int enabled, check, no_device, fault;
    double min_work, max_work, max_bytes;
    char report[1024];
} cfg;

static _Atomic uint64_t stats[VMB_NSTAT];
/* zgemm_ calls by decade of m*n*k, and time in the GPU attempt / CPU path. */
static _Atomic uint64_t work_hist[16], try_ns, cpu_ns;
static double max_check_ratio;  /* guarded by gpu_lock */
static pthread_mutex_t gpu_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_once_t init_once = PTHREAD_ONCE_INIT;
static zgemm_fn cpu_zgemm;
static char cpu_library[1024];

static const char *const status_names[VMB_NSTATUS] = {
    "ok", "disabled", "trivial", "badarg", "nonfinite", "too_small",
    "too_large", "no_device", "gpu_error", "gpu_nonfinite", "check_failed",
};

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

static void report_at_exit(void)
{
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
    cfg.enabled = env_is("VMB_ENABLE", "1");
    cfg.check = env_is("VMB_CHECK", "1");
    cfg.no_device = env_is("VMB_TEST_NO_DEVICE", "1");
    cfg.fault = env_is("VMB_TEST_FAULT", "gpu_error")       ? FAULT_GPU_ERROR
                : env_is("VMB_TEST_FAULT", "gpu_nonfinite") ? FAULT_GPU_NONFINITE
                                                            : FAULT_NONE;
    cfg.min_work = env_double("VMB_MIN_WORK", 0.0);
    cfg.max_work = env_double("VMB_MAX_WORK", 0x1p34);
    cfg.max_bytes = env_double("VMB_MAX_BYTES", 0x1p30);
    const char *report = getenv("VMB_REPORT");
    snprintf(cfg.report, sizeof cfg.report, "%s", report ? report : "");
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
    if (cfg.report[0])
        atexit(report_at_exit);
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

/* Returns 0 if a stored element is non-finite; *maxabs is max(|re|, |im|). */
static int scan(const double *x, int rows, int cols, int ld, double *maxabs)
{
    double mx = 0.0;
    for (int j = 0; j < cols; j++) {
        const double *col = x + 2 * (size_t)j * ld;
        for (int i = 0; i < 2 * rows; i++) {
            double v = fabs(col[i]);
            if (!isfinite(v))
                return 0;
            if (v > mx)
                mx = v;
        }
    }
    *maxabs = mx;
    return 1;
}

/* Exponent e with v * 2^-e in [0.5, 1); 0 for v == 0. */
static int scale_exp(double v) { return v > 0 ? ilogb(v) + 1 : 0; }

static void op_elem(const double *x, int ld, int trans, int row, int col, double *re, double *im)
{
    const double *z = trans ? x + 2 * ((size_t)col + (size_t)row * ld)
                            : x + 2 * ((size_t)row + (size_t)col * ld);
    *re = z[0];
    *im = trans == 2 ? -z[1] : z[1];
}

/* Max over elements of |gpu - cpu| / (atol + rtol S_ij) against OpenBLAS. */
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

    double smax = 0.0, aa = hypot(alpha[0], alpha[1]), ab = hypot(beta[0], beta[1]);
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
            if (sij > smax)
                smax = sij;
        }
    double ratio = 0.0, atol = VMB_ATOL(smax), rtol = VMB_RTOL(k);
    for (size_t e = 0; e < mn; e++) {
        double r = hypot(out[2 * e] - ref[2 * e], out[2 * e + 1] - ref[2 * e + 1]) /
                   (atol + rtol * s[e]);
        if (!(r <= ratio))
            ratio = r;  /* also propagates NaN */
    }
    free(ref);
    free(s);
    return ratio;
}

static int try_gpu(char transa, char transb, int m, int n, int k, const double alpha[2],
                   const double *a, int lda, const double *b, int ldb,
                   const double beta[2], double *c, int ldc)
{
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

    int beta_zero = beta[0] == 0.0 && beta[1] == 0.0;  /* BLAS: C is not read */
    double ma, mb, mc = 0.0;
    if (!scan(a, rows_a, cols_a, lda, &ma) || !scan(b, rows_b, cols_b, ldb, &mb) ||
        (!beta_zero && !scan(c, m, n, ldc, &mc)))
        return VMB_DECLINED_NONFINITE;
    if (!vmb_gpu_available(cfg.no_device))
        return VMB_DECLINED_NO_DEVICE;

    /* Power-of-two scaling keeps packed values in [-1, 1] and the output near
       unit magnitude, so float-float exponent range is never the limit. */
    struct vmb_gpu_job job = {
        .ta = ta, .tb = tb, .m = m, .n = n, .k = k,
        .a = a, .lda = lda, .rows_a = rows_a, .cols_a = cols_a, .sa = scale_exp(ma),
        .b = b, .ldb = ldb, .rows_b = rows_b, .cols_b = cols_b, .sb = scale_exp(mb),
        .c = c, .ldc = ldc, .beta_zero = beta_zero, .sc = scale_exp(mc),
    };
    int kbits = ilogb((double)k) + 2;  /* 2k <= 2^kbits bounds the sum */
    job.so = scale_exp(fmax(fabs(alpha[0]), fabs(alpha[1]))) + 1 + job.sa + job.sb + kbits;
    if (!beta_zero && mc > 0.0) {
        int so_c = scale_exp(fmax(fabs(beta[0]), fabs(beta[1]))) + 1 + job.sc;
        if (so_c > job.so)
            job.so = so_c;
    }
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
    if (cfg.enabled) {
        status = vmb_zgemm_try(*transa, *transb, *m, *n, *k, alpha, a, *lda, b, *ldb,
                               beta, c, *ldc);
    } else {
        status = VMB_DECLINED_DISABLED;
        stats[VMB_STAT_REASON_BASE + status]++;
    }
    uint64_t t1 = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
    try_ns += t1 - t0;
    if (status != VMB_OK) {
        stats[VMB_STAT_CPU_CALLS]++;
        cpu_zgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                  len_transa, len_transb);
        cpu_ns += clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - t1;
    }
}

void vmb_cpu_zgemm(char transa, char transb, int m, int n, int k, const double alpha[2],
                   const double *a, int lda, const double *b, int ldb,
                   const double beta[2], double *c, int ldc)
{
    pthread_once(&init_once, init);
    cpu_zgemm(&transa, &transb, &m, &n, &k, alpha, a, &lda, b, &ldb, beta, c, &ldc, 1, 1);
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
    fprintf(out, "vmb.enabled=%d\nvmb.check=%d\nvmb.min_work=%.17g\n", cfg.enabled, cfg.check,
            cfg.min_work);
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
