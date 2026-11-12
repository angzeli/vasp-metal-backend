/* v0.5 standalone DGEMV/DGEMM benchmark on observed VASP shapes (no VASP).
 *
 *   blas_replay replay SHAPES.tsv --lib openblas|accelerate [--reps R] [--calls N]
 *                      [--cold-mb MB] [--start-at UNIX_SECONDS] --output NEW.tsv
 *   blas_replay batch  SHAPES.tsv --lib openblas|accelerate [--batches 1,2,4,...]
 *                      [--reps R] --output NEW.tsv
 *
 * replay times N consecutive calls of each shape, R times, and reports the median
 * and quartiles per call. With --cold-mb the matrix rotates through a pool at
 * least that large, so it is not in cache (VASP's projector arrays are large).
 * batch compares b DGEMV calls, y_i = alpha op(A) x_i + beta y_i, with one DGEMM on
 * the packed X = [x_1 .. x_b], counting the packing of X and the gather and
 * scatter of Y, for each b. Every library result is checked against a
 * reference sum. Inputs are multiples of 2^-15 in [-1, 1), so every product
 * and partial sum is exact in double (long double is double on arm64) and any
 * correct result matches the reference exactly, whatever the summation order.
 *
 * SHAPES.tsv: id op trans transb m n k lda ldb_or_incx ldc_or_incy beta
 * (op dgemv|dgemm; transb and k are '-'/0 for dgemv; beta zero|one|other).
 * OpenBLAS is the linked library; Accelerate is loaded explicitly. One thread
 * each (OPENBLAS_NUM_THREADS / VECLIB_MAXIMUM_THREADS = 1) unless set. */
#define _DARWIN_C_SOURCE
#include <dlfcn.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

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
void openblas_set_num_threads(int);
int openblas_get_num_threads(void);

static dgemv_fn gemv;
static dgemm_fn gemm;
static char lib_image[1024];

struct shape {
    char id[64], op[8], ta, tb, beta_class[8];
    int m, n, k, lda, ldb, ldc;
};

static double now(void) { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW) * 1e-9; }

__attribute__((noreturn)) static void die(const char *msg)
{
    fprintf(stderr, "blas_replay: %s\n", msg);
    exit(2);
}

static void load_library(const char *name)
{
    void *g, *m;
    if (strcmp(name, "openblas") == 0) {
        g = (void *)dgemv_;
        m = (void *)dgemm_;
    } else if (strcmp(name, "accelerate") == 0) {
        void *h = dlopen("/System/Library/Frameworks/Accelerate.framework/Accelerate",
                         RTLD_NOW | RTLD_LOCAL);
        if (!h)
            die("cannot load Accelerate");
        g = dlsym(h, "dgemv_");
        m = dlsym(h, "dgemm_");
    } else {
        die("--lib must be openblas or accelerate");
    }
    Dl_info info;
    if (!g || !m || !dladdr(g, &info))
        die("dgemv_/dgemm_ not found");
    snprintf(lib_image, sizeof lib_image, "%s", info.dli_fname);
    if ((strcmp(name, "openblas") == 0) != (strstr(lib_image, "openblas") != NULL))
        die("resolved dgemv_ is not from the requested library");
    gemv = (dgemv_fn)g;
    gemm = (dgemm_fn)m;
}

static int read_shapes(const char *path, struct shape **out)
{
    FILE *f = fopen(path, "r");
    if (!f)
        die("cannot open shapes file");
    char line[512];
    int n = 0, cap = 64;
    struct shape *s = malloc(sizeof *s * cap);
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#' || strncmp(line, "id\t", 3) == 0 || line[0] == '\n')
            continue;
        if (n == cap)
            s = realloc(s, sizeof *s * (cap *= 2));
        struct shape *x = &s[n];
        char ta[4], tb[4];
        if (sscanf(line, "%63s %7s %3s %3s %d %d %d %d %d %d %7s", x->id, x->op, ta, tb, &x->m,
                   &x->n, &x->k, &x->lda, &x->ldb, &x->ldc, x->beta_class) != 11)
            die("malformed shapes line");
        x->ta = ta[0];
        x->tb = tb[0];
        int is_gemv = strcmp(x->op, "dgemv") == 0;
        if (!is_gemv && strcmp(x->op, "dgemm") != 0)
            die("op must be dgemv or dgemm");
        if (x->m < 1 || x->n < 1 || x->lda < 1 || (!is_gemv && x->k < 1))
            die("non-positive dimension");
        n++;
    }
    fclose(f);
    *out = s;
    return n;
}

static double beta_value(const struct shape *s)
{
    return strcmp(s->beta_class, "zero") == 0 ? 0.0 : strcmp(s->beta_class, "one") == 0 ? 1.0 : 0.5;
}

static void fill(double *x, size_t n, unsigned seed)
{
    for (size_t i = 0; i < n; i++) {
        seed = seed * 1103515245u + 12345u;
        x[i] = ((seed >> 8) & 0xffff) / 32768.0 - 1.0;
    }
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static void quartiles(double *v, int n, double q[3])
{
    qsort(v, n, sizeof *v, cmp_double);
    q[0] = v[n / 4];
    q[1] = n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
    q[2] = v[(3 * n) / 4];
}

/* Matrix element (i, j) of op(A) for an m x n stored matrix. */
static double op_a(const double *a, int lda, char t, int i, int j)
{
    return (t == 'N' || t == 'n') ? a[i + (size_t)j * lda] : a[j + (size_t)i * lda];
}

/* Largest |lib - ref| / (sum |terms| + |beta y|), over a DGEMV result. */
static double gemv_error(const struct shape *s, const double *a, const double *x, const double *y0,
                         const double *y, double alpha, double beta)
{
    int tr = s->ta != 'N' && s->ta != 'n';
    int rows = tr ? s->n : s->m, cols = tr ? s->m : s->n, ix = abs(s->ldb), iy = abs(s->ldc);
    double worst = 0;
    for (int i = 0; i < rows; i++) {
        long double acc = 0, mag = 0;
        for (int j = 0; j < cols; j++) {
            long double t = (long double)op_a(a, s->lda, s->ta, i, j) *
                            x[(size_t)(s->ldb > 0 ? j : cols - 1 - j) * ix];
            acc += t;
            mag += fabsl(t);
        }
        size_t yi = (size_t)(s->ldc > 0 ? i : rows - 1 - i) * iy;
        long double ref = alpha * acc + (beta != 0 ? beta * (long double)y0[yi] : 0);
        long double scale = fabsl(alpha) * mag + (beta != 0 ? fabsl(beta * (long double)y0[yi]) : 0);
        double e = scale > 0 ? (double)(fabsl(ref - y[yi]) / scale) : (ref == y[yi] ? 0 : INFINITY);
        if (e > worst)
            worst = e;
    }
    return worst;
}

static double gemm_error(const struct shape *s, const double *a, const double *b, const double *c0,
                         const double *c, double alpha, double beta)
{
    double worst = 0;
    for (int j = 0; j < s->n; j++)
        for (int i = 0; i < s->m; i++) {
            long double acc = 0, mag = 0;
            for (int l = 0; l < s->k; l++) {
                long double t = (long double)op_a(a, s->lda, s->ta, i, l) *
                                op_a(b, s->ldb, s->tb, l, j);
                acc += t;
                mag += fabsl(t);
            }
            size_t ci = i + (size_t)j * s->ldc;
            long double ref = alpha * acc + (beta != 0 ? beta * (long double)c0[ci] : 0);
            long double scale = fabsl(alpha) * mag + (beta != 0 ? fabsl(beta * (long double)c0[ci]) : 0);
            double e = scale > 0 ? (double)(fabsl(ref - c[ci]) / scale) : (ref == c[ci] ? 0 : INFINITY);
            if (e > worst)
                worst = e;
        }
    return worst;
}

static void wait_until(double start_at)
{
    if (start_at <= 0)
        return;
    for (;;) {
        struct timeval tv;
        gettimeofday(&tv, NULL);
        double left = start_at - (tv.tv_sec + tv.tv_usec * 1e-6);
        if (left <= 0)
            return;
        usleep(left > 0.01 ? 5000 : 100);
    }
}

static void replay(const struct shape *s, int ns, int reps, int calls, double cold_mb, FILE *out)
{
    fprintf(out, "id\top\ttrans\ttransb\tm\tn\tk\tlda\tldb_or_incx\tldc_or_incy\tbeta\t"
                 "pool_copies\tcalls\treps\tq1_us\tmedian_us\tq3_us\tgflops\tmax_rel_err\n");
    for (int si = 0; si < ns; si++) {
        const struct shape *x = &s[si];
        int is_gemv = strcmp(x->op, "dgemv") == 0, tr = x->ta != 'N' && x->ta != 'n';
        double alpha = 1.0, beta = beta_value(x);
        size_t a_elems = (size_t)x->lda * (is_gemv ? x->n : ((x->ta == 'N' || x->ta == 'n') ? x->k : x->m));
        size_t b_elems, c_elems;
        if (is_gemv) {
            b_elems = (size_t)((tr ? x->m : x->n) - 1) * abs(x->ldb) + 1;
            c_elems = (size_t)((tr ? x->n : x->m) - 1) * abs(x->ldc) + 1;
        } else {
            b_elems = (size_t)x->ldb * ((x->tb == 'N' || x->tb == 'n') ? x->n : x->k);
            c_elems = (size_t)x->ldc * x->n;
        }
        int copies = 1;
        if (cold_mb > 0) {
            double bytes = 8.0 * (double)a_elems;
            copies = (int)fmin(4096.0, ceil(cold_mb * 1048576.0 / fmax(bytes, 1.0)));
        }
        double *a = malloc(sizeof(double) * a_elems * copies), *b = malloc(sizeof(double) * b_elems);
        double *c = malloc(sizeof(double) * c_elems), *c0 = malloc(sizeof(double) * c_elems);
        fill(a, a_elems * copies, 11 + si);
        fill(b, b_elems, 13 + si);
        fill(c0, c_elems, 17 + si);
        /* Correctness of one call against the long-double reference. */
        memcpy(c, c0, sizeof(double) * c_elems);
        double err;
        if (is_gemv) {
            gemv(&x->ta, &x->m, &x->n, &alpha, a, &x->lda, b, &x->ldb, &beta, c, &x->ldc, 1);
            err = gemv_error(x, a, b, c0, c, alpha, beta);
        } else {
            gemm(&x->ta, &x->tb, &x->m, &x->n, &x->k, &alpha, a, &x->lda, b, &x->ldb, &beta, c,
                 &x->ldc, 1, 1);
            err = gemm_error(x, a, b, c0, c, alpha, beta);
        }
        double *t = malloc(sizeof(double) * reps);
        for (int r = 0; r < reps; r++) {
            memcpy(c, c0, sizeof(double) * c_elems);
            double t0 = now();
            for (int i = 0; i < calls; i++) {
                const double *ai = a + a_elems * (size_t)(i % copies);
                if (is_gemv)
                    gemv(&x->ta, &x->m, &x->n, &alpha, ai, &x->lda, b, &x->ldb, &beta, c, &x->ldc, 1);
                else
                    gemm(&x->ta, &x->tb, &x->m, &x->n, &x->k, &alpha, ai, &x->lda, b, &x->ldb,
                         &beta, c, &x->ldc, 1, 1);
            }
            t[r] = (now() - t0) / calls * 1e6;
        }
        double q[3];
        quartiles(t, reps, q);
        double flops = is_gemv ? 2.0 * x->m * x->n : 2.0 * x->m * x->n * x->k;
        fprintf(out, "%s\t%s\t%c\t%c\t%d\t%d\t%d\t%d\t%d\t%d\t%s\t%d\t%d\t%d\t%.4f\t%.4f\t%.4f\t%.3f\t%.3e\n",
                x->id, x->op, x->ta, is_gemv ? '-' : x->tb, x->m, x->n, is_gemv ? 0 : x->k, x->lda,
                x->ldb, x->ldc, x->beta_class, copies, calls, reps, q[0], q[1], q[2],
                flops / (q[1] * 1e3), err);
        fflush(out);
        free(a), free(b), free(c), free(c0), free(t);
    }
}

/* b DGEMV calls against one DGEMM on packed vectors, per vector. */
static void batch(const struct shape *s, int ns, const int *bs, int nb, int reps, FILE *out)
{
    fprintf(out, "id\ttrans\tm\tn\tlda\tincx\tincy\tbeta\tb\tgemv_median_us_per_vector\t"
                 "gemm_median_us_per_vector\tgemm_over_gemv\tmax_rel_diff\n");
    for (int si = 0; si < ns; si++) {
        const struct shape *x = &s[si];
        if (strcmp(x->op, "dgemv") != 0)
            continue;
        int tr = x->ta != 'N' && x->ta != 'n', lx = tr ? x->m : x->n, ly = tr ? x->n : x->m;
        int ix = x->ldb, iy = x->ldc;
        size_t nxv = (size_t)(lx - 1) * abs(ix) + 1, nyv = (size_t)(ly - 1) * abs(iy) + 1;
        double alpha = 1.0, beta = beta_value(x);
        double *a = malloc(sizeof(double) * (size_t)x->lda * x->n);
        fill(a, (size_t)x->lda * x->n, 21 + si);
        for (int bi = 0; bi < nb; bi++) {
            int b = bs[bi];
            double *xs = malloc(sizeof(double) * nxv * b), *y0 = malloc(sizeof(double) * nyv * b);
            double *y1 = malloc(sizeof(double) * nyv * b), *y2 = malloc(sizeof(double) * nyv * b);
            double *X = malloc(sizeof(double) * (size_t)lx * b), *Y = malloc(sizeof(double) * (size_t)ly * b);
            fill(xs, nxv * b, 23 + b);
            fill(y0, nyv * b, 29 + b);
            double *tg = malloc(sizeof(double) * reps), *tm = malloc(sizeof(double) * reps);
            for (int r = 0; r < reps; r++) {
                memcpy(y1, y0, sizeof(double) * nyv * b);
                double t0 = now();
                for (int v = 0; v < b; v++)
                    gemv(&x->ta, &x->m, &x->n, &alpha, a, &x->lda, xs + nxv * v, &ix, &beta,
                         y1 + nyv * v, &iy, 1);
                tg[r] = (now() - t0) / b * 1e6;
                memcpy(y2, y0, sizeof(double) * nyv * b);
                t0 = now();
                for (int v = 0; v < b; v++)  /* pack X (and gather Y when beta reads it) */
                    for (int j = 0; j < lx; j++)
                        X[j + (size_t)lx * v] = xs[nxv * v + (size_t)(ix > 0 ? j : lx - 1 - j) * abs(ix)];
                if (beta != 0)
                    for (int v = 0; v < b; v++)
                        for (int i = 0; i < ly; i++)
                            Y[i + (size_t)ly * v] = y2[nyv * v + (size_t)(iy > 0 ? i : ly - 1 - i) * abs(iy)];
                char nt = 'N';
                gemm(&x->ta, &nt, &ly, &b, &lx, &alpha, a, &x->lda, X, &lx, &beta, Y, &ly, 1, 1);
                for (int v = 0; v < b; v++)  /* scatter Y */
                    for (int i = 0; i < ly; i++)
                        y2[nyv * v + (size_t)(iy > 0 ? i : ly - 1 - i) * abs(iy)] = Y[i + (size_t)ly * v];
                tm[r] = (now() - t0) / b * 1e6;
            }
            double diff = 0;
            for (size_t i = 0; i < nyv * b; i++) {
                double d = fabs(y1[i] - y2[i]) / fmax(1e-300, fabs(y1[i]) + fabs(y0[i]) + 1.0);
                if (d > diff)
                    diff = d;
            }
            double qg[3], qm[3];
            quartiles(tg, reps, qg);
            quartiles(tm, reps, qm);
            fprintf(out, "%s\t%c\t%d\t%d\t%d\t%d\t%d\t%s\t%d\t%.4f\t%.4f\t%.3f\t%.3e\n", x->id, x->ta,
                    x->m, x->n, x->lda, ix, iy, x->beta_class, b, qg[1], qm[1], qm[1] / qg[1], diff);
            fflush(out);
            free(xs), free(y0), free(y1), free(y2), free(X), free(Y), free(tg), free(tm);
        }
        free(a);
    }
}

int main(int argc, char **argv)
{
    if (argc < 3 || (strcmp(argv[1], "replay") && strcmp(argv[1], "batch")))
        die("usage: blas_replay replay|batch SHAPES.tsv --lib L --output NEW.tsv [options]");
    const char *lib = NULL, *output = NULL, *batches = "1,2,4,8,16,32,64";
    int reps = 15, calls = 200;
    double cold_mb = 0, start_at = 0;
    for (int i = 3; i < argc; i += 2) {
        const char *v = i + 1 < argc ? argv[i + 1] : NULL;
        if (!v)
            die("option without a value");
        if (!strcmp(argv[i], "--lib")) lib = v;
        else if (!strcmp(argv[i], "--output")) output = v;
        else if (!strcmp(argv[i], "--reps")) reps = atoi(v);
        else if (!strcmp(argv[i], "--calls")) calls = atoi(v);
        else if (!strcmp(argv[i], "--cold-mb")) cold_mb = atof(v);
        else if (!strcmp(argv[i], "--start-at")) start_at = atof(v);
        else if (!strcmp(argv[i], "--batches")) batches = v;
        else die("unknown option");
    }
    if (!lib || !output || reps < 3 || calls < 1)
        die("--lib and --output are required; --reps >= 3, --calls >= 1");
    /* OpenBLAS read its environment when it loaded, before main, so its thread
       count is set through its API; Accelerate is not loaded yet and reads it. */
    setenv("OPENBLAS_NUM_THREADS", getenv("OPENBLAS_NUM_THREADS") ? getenv("OPENBLAS_NUM_THREADS") : "1", 1);
    setenv("VECLIB_MAXIMUM_THREADS", getenv("VECLIB_MAXIMUM_THREADS") ? getenv("VECLIB_MAXIMUM_THREADS") : "1", 1);
    openblas_set_num_threads(atoi(getenv("OPENBLAS_NUM_THREADS")));
    load_library(lib);
    struct shape *s;
    int ns = read_shapes(argv[2], &s);
    FILE *out = fopen(output, "wx");
    if (!out)
        die("cannot create output (it must not exist)");
    fprintf(out, "# blas_replay %s lib=%s image=%s openblas_threads=%d veclib_threads=%s\n", argv[1],
            lib, lib_image, openblas_get_num_threads(), getenv("VECLIB_MAXIMUM_THREADS"));
    wait_until(start_at);
    if (!strcmp(argv[1], "replay")) {
        replay(s, ns, reps, calls, cold_mb, out);
    } else {
        int bs[32], nb = 0;
        for (const char *p = batches; *p && nb < 32;) {
            bs[nb++] = atoi(p);
            p = strchr(p, ',');
            if (!p)
                break;
            p++;
        }
        batch(s, ns, bs, nb, reps, out);
    }
    fclose(out);
    return 0;
}
