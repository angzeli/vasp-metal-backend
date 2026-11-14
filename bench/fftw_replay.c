/* v0.6 standalone FFTW benchmark on observed VASP transform shapes (no VASP).
 *
 *   fftw_replay replay SHAPES.tsv --lib PATH/libfftw3.dylib [--reps R] [--calls N]
 *                      --output NEW.tsv [--dump PREFIX]
 *   fftw_replay compare A.bin B.bin
 *
 * replay loads the given libfftw3 and, for each shape, repeats what VASP does
 * per transform: plan with FFTW_ESTIMATE, execute once, destroy. It times the
 * planning and the execution of N such calls, R times, and reports per-call
 * medians and quartiles. The input is restored before each execute (not timed),
 * so in-place transforms see the same data every time. One thread.
 *
 * Accuracy: each complex transform's result is checked against an independent
 * separable DFT computed here (relative max error, reported per shape). With
 * --dump, the first result of every shape is written to PREFIX.ID.bin; compare
 * prints the relative max difference of two dumps, for cross-library agreement
 * (this covers the real-to-complex shapes, which have no reference here).
 *
 * SHAPES.tsv: id kind rank n0 n1 n2 howmany istride idist ostride odist sign weight
 * kind dft_3d|many_dft|many_dft_r2c|many_dft_c2r; n0..n2 as VASP passed them to
 * the Fortran interface (Fortran order: reversed here into FFTW's C order);
 * weight is VASP's execute time for the shape (any unit), used by the analysis. */
#define _DARWIN_C_SOURCE
#include <complex.h>
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define FFTW_ESTIMATE (1U << 6)
typedef void *plan_t;
typedef plan_t (*many_dft_fn)(int, const int *, int, double complex *, const int *, int, int,
                              double complex *, const int *, int, int, int, unsigned);
typedef plan_t (*many_r2c_fn)(int, const int *, int, double *, const int *, int, int,
                              double complex *, const int *, int, int, unsigned);
typedef plan_t (*many_c2r_fn)(int, const int *, int, double complex *, const int *, int, int,
                              double *, const int *, int, int, unsigned);
static many_dft_fn plan_many_dft;
static many_r2c_fn plan_many_r2c;
static many_c2r_fn plan_many_c2r;
static void (*execute)(plan_t);
static void (*destroy)(plan_t);

struct shape {
    char id[32], kind[16];
    int rank, n[3], howmany, istride, idist, ostride, odist, sign;
    double weight;
};

__attribute__((noreturn)) static void die(const char *msg)
{
    fprintf(stderr, "fftw_replay: %s\n", msg);
    exit(2);
}

static uint64_t now_ns(void) { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }

static void *need(void *h, const char *name)
{
    void *p = dlsym(h, name);
    if (!p)
        die(name);
    return p;
}

static void load(const char *path)
{
    void *h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!h)
        die("cannot load --lib");
    plan_many_dft = (many_dft_fn)need(h, "fftw_plan_many_dft");
    plan_many_r2c = (many_r2c_fn)need(h, "fftw_plan_many_dft_r2c");
    plan_many_c2r = (many_c2r_fn)need(h, "fftw_plan_many_dft_c2r");
    *(void **)&execute = need(h, "fftw_execute");
    *(void **)&destroy = need(h, "fftw_destroy_plan");
}

static int read_shapes(const char *path, struct shape **out)
{
    FILE *f = fopen(path, "r");
    if (!f)
        die("cannot read SHAPES.tsv");
    int cap = 64, n = 0;
    struct shape *s = malloc(cap * sizeof *s);
    char line[512];
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#' || line[0] == '\n' || !strncmp(line, "id\t", 3))
            continue;
        if (n == cap)
            s = realloc(s, (cap *= 2) * sizeof *s);
        struct shape *x = &s[n];
        if (sscanf(line, "%31s %15s %d %d %d %d %d %d %d %d %d %d %lf", x->id, x->kind, &x->rank,
                   &x->n[0], &x->n[1], &x->n[2], &x->howmany, &x->istride, &x->idist,
                   &x->ostride, &x->odist, &x->sign, &x->weight) != 13 || x->rank < 1 || x->rank > 3)
            die("malformed shape line");
        n++;
    }
    fclose(f);
    *out = s;
    return n;
}

/* Transform length in FFTW's C order, and the buffer size (in doubles) covering
   input and output of an in-place transform. */
static void c_dims(const struct shape *s, int *d)
{
    for (int i = 0; i < s->rank; i++)
        d[i] = s->n[s->rank - 1 - i];
}

static long points(const struct shape *s)
{
    long p = 1;
    for (int i = 0; i < s->rank; i++)
        p *= s->n[i];
    return p;
}

static size_t buffer_doubles(const struct shape *s)
{
    long p = points(s), last = s->n[0];  /* the contiguous (C-last) dimension */
    long span = p / last;
    if (!strcmp(s->kind, "dft_3d"))
        return 2 * (size_t)p;
    if (!strcmp(s->kind, "many_dft")) {
        long in = (long)(s->howmany - 1) * s->idist + (p - 1) * s->istride + 1;
        long out = (long)(s->howmany - 1) * s->odist + (p - 1) * s->ostride + 1;
        return 2 * (size_t)(in > out ? in : out);
    }
    /* r2c: real input in units of doubles, complex output in units of complex. */
    long half = span * (last / 2 + 1);
    long r = (long)(s->howmany - 1) * (strcmp(s->kind, "many_dft_r2c") ? s->odist : s->idist)
             + (p - 1) + 1;
    long c = (long)(s->howmany - 1) * (strcmp(s->kind, "many_dft_r2c") ? s->idist : s->odist)
             + (half - 1) + 1;
    return (size_t)(r > 2 * c ? r : 2 * c);
}

static plan_t make_plan(const struct shape *s, double *buf)
{
    int d[3];
    c_dims(s, d);
    if (!strcmp(s->kind, "dft_3d"))
        return plan_many_dft(s->rank, d, 1, (double complex *)buf, NULL, 1, 0,
                             (double complex *)buf, NULL, 1, 0, s->sign, FFTW_ESTIMATE);
    if (!strcmp(s->kind, "many_dft"))
        return plan_many_dft(s->rank, d, s->howmany, (double complex *)buf, NULL, s->istride,
                             s->idist, (double complex *)buf, NULL, s->ostride, s->odist,
                             s->sign, FFTW_ESTIMATE);
    if (!strcmp(s->kind, "many_dft_r2c"))
        return plan_many_r2c(s->rank, d, s->howmany, buf, NULL, s->istride, s->idist,
                             (double complex *)buf, NULL, s->ostride, s->odist, FFTW_ESTIMATE);
    if (!strcmp(s->kind, "many_dft_c2r"))
        return plan_many_c2r(s->rank, d, s->howmany, (double complex *)buf, NULL, s->istride,
                             s->idist, buf, NULL, s->ostride, s->odist, FFTW_ESTIMATE);
    die("unknown kind");
}

/* Independent reference for complex transforms: a direct DFT along each
   dimension in turn (separable), with exact-angle twiddles. */
static void dft_axis(double complex *x, long count, long len, long stride, long dist,
                     long inner, long inner_stride, int sign)
{
    double complex *w = malloc(len * sizeof *w), *t = malloc(len * sizeof *t);
    for (long k = 0; k < len; k++)
        w[k] = cexp(sign * 2 * M_PI * I * (double)k / (double)len);
    for (long b = 0; b < count; b++)
        for (long q = 0; q < inner; q++) {
            double complex *v = x + b * dist + q * inner_stride;
            for (long k = 0; k < len; k++) {
                double complex acc = 0;
                for (long j = 0; j < len; j++)
                    acc += v[j * stride] * w[(j * k) % len];
                t[k] = acc;
            }
            for (long k = 0; k < len; k++)
                v[k * stride] = t[k];
        }
    free(w);
    free(t);
}

static void reference(const struct shape *s, double complex *x)
{
    int d[3];
    c_dims(s, d);
    if (!strcmp(s->kind, "dft_3d")) {  /* C order d0 x d1 x d2, d2 contiguous */
        dft_axis(x, d[0] * d[1], d[2], 1, d[2], 1, 0, s->sign);
        dft_axis(x, d[0], d[1], d[2], (long)d[1] * d[2], d[2], 1, s->sign);
        dft_axis(x, 1, d[0], (long)d[1] * d[2], 0, (long)d[1] * d[2], 1, s->sign);
    } else {  /* rank-1 many_dft */
        dft_axis(x, s->howmany, d[0], s->istride, s->idist, 1, 0, s->sign);
    }
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double relmax(const double *a, const double *b, size_t n)
{
    double num = 0, den = 0;
    for (size_t i = 0; i < n; i++) {
        double e = fabs(a[i] - b[i]), m = fabs(b[i]);
        if (e > num) num = e;
        if (m > den) den = m;
    }
    return den > 0 ? num / den : num;
}

static void replay(const struct shape *s, int reps, int calls, FILE *out, const char *dump)
{
    size_t nd = buffer_doubles(s);
    double *buf = aligned_alloc(64, ((nd * sizeof(double) + 63) / 64) * 64);
    double *pristine = malloc(nd * sizeof(double));
    for (size_t i = 0; i < nd; i++)  /* deterministic, O(1) values */
        pristine[i] = sin(0.37 * (double)i) + 0.25 * cos(1.3 * (double)(i % 97));
    double *tp = malloc(reps * sizeof(double)), *te = malloc(reps * sizeof(double));

    /* First call: result for the accuracy check and the dump. */
    memcpy(buf, pristine, nd * sizeof(double));
    plan_t p = make_plan(s, buf);
    if (!p)
        die("FFTW returned no plan");
    memcpy(buf, pristine, nd * sizeof(double));
    execute(p);
    destroy(p);
    double err = NAN;
    if (!strcmp(s->kind, "dft_3d") || (!strcmp(s->kind, "many_dft") && s->rank == 1)) {
        double *ref = malloc(nd * sizeof(double));
        memcpy(ref, pristine, nd * sizeof(double));
        reference(s, (double complex *)ref);
        err = relmax(buf, ref, nd);
        free(ref);
    }
    if (dump) {
        char path[512];
        snprintf(path, sizeof path, "%s.%s.bin", dump, s->id);
        FILE *f = fopen(path, "wbx");
        if (!f || fwrite(buf, sizeof(double), nd, f) != nd || fclose(f))
            die("cannot write dump");
    }

    for (int r = 0; r < reps; r++) {
        uint64_t plan_ns = 0, exec_ns = 0;
        for (int c = 0; c < calls; c++) {
            memcpy(buf, pristine, nd * sizeof(double));
            uint64_t t0 = now_ns();
            p = make_plan(s, buf);
            uint64_t t1 = now_ns();
            execute(p);
            uint64_t t2 = now_ns();
            destroy(p);
            plan_ns += t1 - t0;
            exec_ns += t2 - t1;
        }
        tp[r] = plan_ns / 1e3 / calls;
        te[r] = exec_ns / 1e3 / calls;
    }
    qsort(tp, reps, sizeof(double), cmp_double);
    qsort(te, reps, sizeof(double), cmp_double);
    fprintf(out, "%s\t%s\t%d\t%d\t%d\t%d\t%d\t%.6g\t%.3f\t%.3f\t%.3f\t%.3f\t%.3e\n", s->id, s->kind,
            s->n[0], s->n[1], s->n[2], s->howmany, s->sign, s->weight, tp[reps / 2],
            te[reps / 4], te[reps / 2], te[3 * reps / 4], err);
    fflush(out);
    free(buf);
    free(pristine);
    free(tp);
    free(te);
}

static double *read_all(const char *path, size_t *n)
{
    FILE *f = fopen(path, "rb");
    if (!f || fseek(f, 0, SEEK_END))
        die("cannot read dump");
    long bytes = ftell(f);
    rewind(f);
    double *x = malloc(bytes);
    *n = bytes / sizeof(double);
    if (fread(x, sizeof(double), *n, f) != *n)
        die("short dump");
    fclose(f);
    return x;
}

int main(int argc, char **argv)
{
    if (argc == 4 && !strcmp(argv[1], "compare")) {
        size_t na, nb;
        double *a = read_all(argv[2], &na), *b = read_all(argv[3], &nb);
        if (na != nb)
            die("dumps differ in size");
        printf("%.3e\n", relmax(a, b, na));
        return 0;
    }
    if (argc < 3 || strcmp(argv[1], "replay"))
        die("usage: fftw_replay replay SHAPES.tsv --lib PATH --output NEW.tsv [options] | compare A B");
    const char *lib = NULL, *output = NULL, *dump = NULL;
    int reps = 11, calls = 200;
    for (int i = 3; i < argc; i += 2) {
        const char *v = i + 1 < argc ? argv[i + 1] : NULL;
        if (!v)
            die("option without a value");
        if (!strcmp(argv[i], "--lib")) lib = v;
        else if (!strcmp(argv[i], "--output")) output = v;
        else if (!strcmp(argv[i], "--dump")) dump = v;
        else if (!strcmp(argv[i], "--reps")) reps = atoi(v);
        else if (!strcmp(argv[i], "--calls")) calls = atoi(v);
        else die("unknown option");
    }
    if (!lib || !output || reps < 3 || calls < 1)
        die("--lib and --output are required; --reps >= 3, --calls >= 1");
    load(lib);
    struct shape *s;
    int ns = read_shapes(argv[2], &s);
    FILE *out = fopen(output, "wx");
    if (!out)
        die("cannot create --output (it must not exist)");
    fprintf(out, "# lib=%s reps=%d calls=%d\n", lib, reps, calls);
    fprintf(out, "id\tkind\tn0\tn1\tn2\thowmany\tsign\tweight\tplan_us\texec_q1_us\texec_us"
                 "\texec_q3_us\tref_relmax\n");
    for (int i = 0; i < ns; i++)
        replay(&s[i], reps, calls, out, dump);
    fclose(out);
    return 0;
}
