/*
 * Deterministic shape-replay benchmark (v0.2).
 *
 * For each shape of a TSV shape set, operands are generated from a seed derived
 * from the shape id. Repetitions alternate, in alternating order, between
 *   CPU: the original OpenBLAS zgemm_ called directly (single thread), and
 *   GPU: the interposed Fortran-ABI zgemm_ in forced Metal mode,
 * timing host call to C ready. Correctness and determinism are checked outside
 * the timed regions. Wall-time, repetition and working-set caps are enforced.
 *
 * usage: shape_replay SHAPES SUMMARY_TSV RAW_TSV [--reps N] [--shape-seconds S]
 *                     [--total-seconds T] [--max-bytes B] [--only ID]
 */
#define _DARWIN_C_SOURCE
#include "vmb.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int openblas_get_num_threads(void);
char *openblas_get_config(void);
void dgemm_(const char *, const char *, const int *, const int *, const int *, const double *,
            const double *, const int *, const double *, const int *, const double *, double *,
            const int *, size_t, size_t);

#define MAX_REPS 64

struct shape {
    char id[64], family[32], provenance[40], role[16], beta[8];
    char ta, tb;
    int m, n, k, pa, pb, pc;
};

struct mat {
    int rows, cols, ld;
    double *v;
};

static double now(void) { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW) * 1e-9; }

static uint64_t rng;
static double urand(void)  /* splitmix64 -> [-1, 1) */
{
    uint64_t z = (rng += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    return (double)(z >> 11) * 0x1p-52 - 1.0;
}

static uint64_t seed_of(const char *id)
{
    uint64_t h = 1469598103934665603ull;  /* FNV-1a 64 */
    for (; *id; id++)
        h = (h ^ (unsigned char)*id) * 1099511628211ull;
    return h;
}

static struct mat mat_new(int rows, int cols, int pad, int random)
{
    struct mat x = {rows, cols, rows + pad, NULL};
    size_t len = 2 * (size_t)x.ld * cols;
    x.v = malloc(len * sizeof *x.v);
    for (size_t e = 0; x.v && e < len; e++)
        x.v[e] = random ? urand() : 0.0;
    return x;
}

static size_t mat_bytes(const struct mat *x) { return 2 * (size_t)x->ld * x->cols * sizeof(double); }

static int is_trans(char t) { return t != 'N' && t != 'n'; }

static int cmp_double(const void *x, const void *y)
{
    double a = *(const double *)x, b = *(const double *)y;
    return (a > b) - (a < b);
}

/* q-quantile of v[0..n) by linear interpolation; v is sorted in place. */
static double quantile(double *v, int n, double q)
{
    qsort(v, n, sizeof *v, cmp_double);
    double pos = q * (n - 1);
    int lo = (int)floor(pos), hi = lo + 1 < n ? lo + 1 : lo;
    return v[lo] + (pos - lo) * (v[hi] - v[lo]);
}

static int read_shapes(const char *path, struct shape *out, int max)
{
    FILE *in = fopen(path, "r");
    if (!in)
        return -1;
    char line[512];
    int count = 0;
    while (fgets(line, sizeof line, in) && count < max) {
        if (line[0] == '#' || strncmp(line, "id\t", 3) == 0 || line[0] == '\n')
            continue;
        struct shape *s = &out[count];
        if (sscanf(line, "%63s %31s %39s %15s %c %c %d %d %d %d %d %d %7s", s->id, s->family,
                   s->provenance, s->role, &s->ta, &s->tb, &s->m, &s->n, &s->k, &s->pa, &s->pb,
                   &s->pc, s->beta) == 13)
            count++;
    }
    fclose(in);
    return count;
}

/* S = |alpha| |op(A)| |op(B)| + |beta| |C0| via OpenBLAS dgemm on moduli, and
   the worst err/tol of got against the OpenBLAS reference. */
static double check(const struct shape *s, const double *alpha, const double *beta,
                    const struct mat *a, const struct mat *b, const struct mat *c0,
                    const struct mat *ref, const struct mat *got)
{
    struct mat aa = mat_new(a->rows, a->cols, 0, 0), ba = mat_new(b->rows, b->cols, 0, 0);
    size_t mn = (size_t)s->m * s->n;
    double *sc = calloc(mn, sizeof *sc), *g = malloc(16 * mn), *r = malloc(16 * mn);
    for (int j = 0; j < a->cols; j++)
        for (int i = 0; i < a->rows; i++)
            aa.v[i + (size_t)j * a->rows] =
                hypot(a->v[2 * (i + (size_t)j * a->ld)], a->v[2 * (i + (size_t)j * a->ld) + 1]);
    for (int j = 0; j < b->cols; j++)
        for (int i = 0; i < b->rows; i++)
            ba.v[i + (size_t)j * b->rows] =
                hypot(b->v[2 * (i + (size_t)j * b->ld)], b->v[2 * (i + (size_t)j * b->ld) + 1]);
    const double am = hypot(alpha[0], alpha[1]), zero = 0.0;
    char ta = is_trans(s->ta) ? 'T' : 'N', tb = is_trans(s->tb) ? 'T' : 'N';
    dgemm_(&ta, &tb, &s->m, &s->n, &s->k, &am, aa.v, &a->rows, ba.v, &b->rows, &zero, sc, &s->m,
           1, 1);
    double bm = hypot(beta[0], beta[1]);
    for (int j = 0; j < s->n; j++)
        for (int i = 0; i < s->m; i++) {
            size_t e = i + (size_t)j * s->m, p = 2 * (i + (size_t)j * c0->ld);
            sc[e] += bm * hypot(c0->v[p], c0->v[p + 1]);
            memcpy(g + 2 * e, got->v + p, 16);
            memcpy(r + 2 * e, ref->v + p, 16);
        }
    double ratio = vmb_tolerance_ratio(mn, g, r, sc, s->k);
    free(aa.v), free(ba.v), free(sc), free(g), free(r);
    return ratio;
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: %s SHAPES SUMMARY_TSV RAW_TSV [--reps N] [--shape-seconds S] "
                        "[--total-seconds T] [--max-bytes B] [--only ID]\n", argv[0]);
        return 2;
    }
    int reps = 7;
    double shape_seconds = 20.0, total_seconds = 300.0, max_bytes = 128.0 * 1024 * 1024;
    const char *only = NULL;
    for (int i = 4; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--reps")) reps = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--shape-seconds")) shape_seconds = atof(argv[i + 1]);
        else if (!strcmp(argv[i], "--total-seconds")) total_seconds = atof(argv[i + 1]);
        else if (!strcmp(argv[i], "--max-bytes")) max_bytes = atof(argv[i + 1]);
        else if (!strcmp(argv[i], "--only")) only = argv[i + 1];
        else { fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
    }
    if (reps < 3 || reps > MAX_REPS) {
        fprintf(stderr, "--reps must be in [3, %d]\n", MAX_REPS);
        return 2;
    }
    if (openblas_get_num_threads() != 1 || vmb_mode() != VMB_MODE_FORCE) {
        fprintf(stderr, "refusing: need OPENBLAS_NUM_THREADS=1 and VMB_MODE=force "
                        "(threads=%d mode=%d)\n", openblas_get_num_threads(), vmb_mode());
        return 2;
    }
    static struct shape shapes[256];
    int nshapes = read_shapes(argv[1], shapes, 256);
    FILE *sum = fopen(argv[2], "w"), *raw = fopen(argv[3], "w");
    if (nshapes <= 0 || !sum || !raw) {
        fprintf(stderr, "cannot read shapes or open outputs\n");
        return 2;
    }
    double init_s = 0.0;
    int have_device = vmb_device_init(&init_s);
    fprintf(sum, "# device=%s init_seconds=%.6f openblas_threads=%d openblas=%s\n",
            have_device ? vmb_device_name() : "none", init_s, openblas_get_num_threads(),
            openblas_get_config());
    fprintf(sum, "# reps=%d shape_seconds=%.1f total_seconds=%.1f max_bytes=%.0f\n", reps,
            shape_seconds, total_seconds, max_bytes);
    fprintf(sum, "id\tfamily\tprovenance\trole\ttransa\ttransb\tm\tn\tk\tlda\tldb\tldc\tbeta\t"
                 "work\tbytes\tstatus\treps\terr_tol\tdeterministic\t"
                 "cpu_med\tcpu_p25\tcpu_p75\tcpu_min\tcpu_max\t"
                 "gpu_med\tgpu_p25\tgpu_p75\tgpu_min\tgpu_max\tcpu_over_gpu\tverdict\t"
                 "pack_med\tencode_med\tgpu_exec_med\twait_med\tunpack_med\tfinish_med\n");
    fprintf(raw, "id\trep\tfirst\tcpu_s\tgpu_s\tpack\tencode\tgpu_exec\twait\tunpack\tfinish\n");
    if (!have_device) {
        fprintf(stderr, "no Metal device\n");
        return 1;
    }

    double t_all = now();
    for (int si = 0; si < nshapes; si++) {
        const struct shape *s = &shapes[si];
        if (only && strcmp(only, s->id) != 0)
            continue;
        if (now() - t_all > total_seconds) {
            fprintf(stderr, "total wall-time cap reached; stopping before %s\n", s->id);
            break;
        }
        rng = seed_of(s->id);
        double alpha[2] = {0.75, -0.5};
        double beta[2] = {0, 0};
        if (!strcmp(s->beta, "one")) beta[0] = 1.0;
        else if (!strcmp(s->beta, "other")) beta[0] = -0.3, beta[1] = 0.8;
        int ra = is_trans(s->ta) ? s->k : s->m, ca = is_trans(s->ta) ? s->m : s->k;
        int rb = is_trans(s->tb) ? s->n : s->k, cb = is_trans(s->tb) ? s->k : s->n;
        double bytes = 16.0 * ((double)(ra + s->pa) * ca + (double)(rb + s->pb) * cb +
                               4.0 * (s->m + s->pc) * (double)s->n);
        double work = (double)s->m * s->n * s->k;
        const char *status = "ok";
        if (bytes > max_bytes) {
            fprintf(sum, "%s\t%s\t%s\t%s\t%c\t%c\t%d\t%d\t%d\t%d\t%d\t%d\t%s\t%.0f\t%.0f\t"
                         "skipped-bytes\n", s->id, s->family, s->provenance, s->role, s->ta,
                    s->tb, s->m, s->n, s->k, ra + s->pa, rb + s->pb, s->m + s->pc, s->beta,
                    work, bytes);
            continue;
        }
        struct mat a = mat_new(ra, ca, s->pa, 1), b = mat_new(rb, cb, s->pb, 1);
        struct mat c0 = mat_new(s->m, s->n, s->pc, 1), cref = mat_new(s->m, s->n, s->pc, 0);
        struct mat cg = mat_new(s->m, s->n, s->pc, 0), cw = mat_new(s->m, s->n, s->pc, 0);
        size_t cbytes = mat_bytes(&c0);

        /* Untimed warm-up and correctness: reference, one forced GPU call. */
        memcpy(cref.v, c0.v, cbytes);
        vmb_cpu_zgemm(s->ta, s->tb, s->m, s->n, s->k, alpha, a.v, a.ld, b.v, b.ld, beta,
                      cref.v, cref.ld);
        memcpy(cg.v, c0.v, cbytes);
        uint64_t before[VMB_NSTAT], after[VMB_NSTAT];
        vmb_stats_get(before);
        zgemm_(&s->ta, &s->tb, &s->m, &s->n, &s->k, alpha, a.v, &a.ld, b.v, &b.ld, beta, cg.v,
               &cg.ld, 1, 1);
        vmb_stats_get(after);
        double err_tol = INFINITY;
        int deterministic = 1, done = 0;
        double cpu[MAX_REPS], gpu[MAX_REPS], ph[6][MAX_REPS];
        if (after[VMB_STAT_CONSUMED] - before[VMB_STAT_CONSUMED] != 1) {
            status = "not-offloaded";
        } else {
            err_tol = check(s, alpha, beta, &a, &b, &c0, &cref, &cg);
            if (!(err_tol <= 1.0))
                status = "inaccurate";
        }
        double t_shape = now();
        for (int r = 0; !strcmp(status, "ok") && r < reps; r++) {
            if (now() - t_shape > shape_seconds || now() - t_all > total_seconds) {
                status = "capped";
                break;
            }
            int gpu_first = r % 2;
            struct vmb_timing tm = {0};
            for (int o = 0; o < 2; o++) {
                int run_gpu = o == 0 ? gpu_first : !gpu_first;
                memcpy(cw.v, c0.v, cbytes);
                if (run_gpu) {
                    vmb_stats_get(before);
                    double t0 = now();
                    zgemm_(&s->ta, &s->tb, &s->m, &s->n, &s->k, alpha, a.v, &a.ld, b.v, &b.ld,
                           beta, cw.v, &cw.ld, 1, 1);
                    gpu[r] = now() - t0;
                    vmb_stats_get(after);
                    vmb_last_timing(&tm);
                    if (after[VMB_STAT_CONSUMED] - before[VMB_STAT_CONSUMED] != 1)
                        status = "not-offloaded";
                    if (memcmp(cw.v, cg.v, cbytes) != 0)
                        deterministic = 0;
                } else {
                    double t0 = now();
                    vmb_cpu_zgemm(s->ta, s->tb, s->m, s->n, s->k, alpha, a.v, a.ld, b.v, b.ld,
                                  beta, cw.v, cw.ld);
                    cpu[r] = now() - t0;
                }
            }
            double phase[6] = {tm.pack, tm.encode, tm.gpu, tm.wait, tm.unpack, tm.finish};
            for (int p = 0; p < 6; p++)
                ph[p][r] = phase[p];
            fprintf(raw, "%s\t%d\t%s\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\n", s->id, r,
                    gpu_first ? "gpu" : "cpu", cpu[r], gpu[r], tm.pack, tm.encode, tm.gpu,
                    tm.wait, tm.unpack, tm.finish);
            done = r + 1;
        }
        if (!deterministic)
            status = "nondeterministic";
        fprintf(sum, "%s\t%s\t%s\t%s\t%c\t%c\t%d\t%d\t%d\t%d\t%d\t%d\t%s\t%.0f\t%.0f\t%s\t%d\t"
                     "%.3e\t%d", s->id, s->family, s->provenance, s->role, s->ta, s->tb, s->m,
                s->n, s->k, a.ld, b.ld, c0.ld, s->beta, work, bytes, status, done, err_tol,
                deterministic);
        if (done >= 3) {
            double cmin = quantile(cpu, done, 0), cmax = quantile(cpu, done, 1);
            double gmin = quantile(gpu, done, 0), gmax = quantile(gpu, done, 1);
            double cmed = quantile(cpu, done, 0.5), gmed = quantile(gpu, done, 0.5);
            /* A win must not overlap: every GPU time below every CPU time. */
            const char *verdict = gmax < cmin ? "gpu-faster" : gmin > cmax ? "cpu-faster" : "overlap";
            fprintf(sum, "\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.3f\t%s",
                    cmed, quantile(cpu, done, 0.25), quantile(cpu, done, 0.75), cmin, cmax, gmed,
                    quantile(gpu, done, 0.25), quantile(gpu, done, 0.75), gmin, gmax, cmed / gmed,
                    verdict);
            for (int p = 0; p < 6; p++)
                fprintf(sum, "\t%.6e", quantile(ph[p], done, 0.5));
        }
        fprintf(sum, "\n");
        fflush(sum);
        fflush(raw);
        free(a.v), free(b.v), free(c0.v), free(cref.v), free(cg.v), free(cw.v);
    }
    fclose(sum);
    fclose(raw);
    return 0;
}
