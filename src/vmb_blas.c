/* v0.5 DGEMV and DGEMM telemetry interposers.
 *
 * Both routines are always forwarded, unchanged, to the OpenBLAS image: this
 * layer never computes a result. With profiling active (vmb_profile_level) it
 * aggregates calls and time per (shape, return address). At level REUSE it
 * also checks, per DGEMV call site, whether consecutive calls use the same
 * matrix contents and whether one call's output feeds the next call's input.
 * Matrix contents are hashed in memory only; no value, hash or address is
 * written, only offsets inside the calling image and run statistics. */
#define _DARWIN_C_SOURCE
#include "vmb.h"
#include "vmb_internal.h"

#include <os/lock.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef void (*dgemv_fn)(const char *, const int *, const int *, const double *,
                         const double *, const int *, const double *, const int *,
                         const double *, double *, const int *, size_t);
typedef void (*dgemm_fn)(const char *, const char *, const int *, const int *, const int *,
                         const double *, const double *, const int *, const double *,
                         const int *, const double *, double *, const int *, size_t, size_t);

void dgemv_(const char *trans, const int *m, const int *n, const double *alpha,
            const double *a, const int *lda, const double *x, const int *incx,
            const double *beta, double *y, const int *incy, size_t len_trans);
void dgemm_(const char *transa, const char *transb, const int *m, const int *n, const int *k,
            const double *alpha, const double *a, const int *lda, const double *b,
            const int *ldb, const double *beta, double *c, const int *ldc, size_t len_transa,
            size_t len_transb);

static dgemv_fn cpu_dgemv;
static dgemm_fn cpu_dgemm;
static char blas_library[1024];
static pthread_once_t resolve_once = PTHREAD_ONCE_INIT;

static void resolve(void)
{
    cpu_dgemv = (dgemv_fn)vmb_find_symbol("libopenblas", "dgemv_", (const void *)dgemv_,
                                          blas_library, sizeof blas_library);
    cpu_dgemm = (dgemm_fn)vmb_find_symbol("libopenblas", "dgemm_", (const void *)dgemm_, NULL, 0);
    if (!cpu_dgemv || !cpu_dgemm) {
        fprintf(stderr, "vmb: OpenBLAS dgemv_/dgemm_ not found among loaded images; "
                        "link libvmb.a ahead of -lopenblas\n");
        abort();
    }
}

/* ---- shape table ---------------------------------------------------------- */

enum { OP_DGEMV, OP_DGEMM };
#define SLOTS 16384  /* power of two; keys beyond this are only counted */

struct key {
    const void *site;
    int op, m, n, k, lda, ldb, ldc;  /* DGEMV: ldb = incx, ldc = incy, k = 0 */
    char ta, tb;
    signed char alpha_class, beta_class;
};

struct entry {
    struct key key;
    int used;
    uint64_t calls, ticks, max_ticks;
};

static struct entry table[SLOTS];
static uint64_t dropped, total_calls[2], total_ticks[2];
static os_unfair_lock lock = OS_UNFAIR_LOCK_INIT;

static signed char scalar_class(double v) { return v == 0.0 ? 0 : v == 1.0 ? 1 : 2; }

static uint32_t key_hash(const struct key *k)
{
    uint64_t h = 1469598103934665603ull ^ (uint64_t)(uintptr_t)k->site;
    const int f[] = {k->op, k->m, k->n, k->k, k->lda, k->ldb, k->ldc, k->ta, k->tb,
                     k->alpha_class, k->beta_class};
    for (size_t i = 0; i < sizeof f / sizeof f[0]; i++)
        h = (h ^ (uint32_t)f[i]) * 1099511628211ull;
    return (uint32_t)(h ^ (h >> 32));
}

static int key_same(const struct key *x, const struct key *y)
{
    return x->site == y->site && x->op == y->op && x->m == y->m && x->n == y->n &&
           x->k == y->k && x->lda == y->lda && x->ldb == y->ldb && x->ldc == y->ldc &&
           x->ta == y->ta && x->tb == y->tb && x->alpha_class == y->alpha_class &&
           x->beta_class == y->beta_class;
}

static void record(const struct key *k, uint64_t ticks)
{
    os_unfair_lock_lock(&lock);
    total_calls[k->op]++;
    total_ticks[k->op] += ticks;
    uint32_t h = key_hash(k);
    for (uint32_t probe = 0; probe < SLOTS; probe++) {
        struct entry *e = &table[(h + probe) & (SLOTS - 1)];
        if (!e->used) {
            *e = (struct entry){.key = *k, .used = 1, .calls = 1, .ticks = ticks,
                                .max_ticks = ticks};
            goto out;
        }
        if (key_same(&e->key, k)) {
            e->calls++;
            e->ticks += ticks;
            if (ticks > e->max_ticks)
                e->max_ticks = ticks;
            goto out;
        }
    }
    dropped++;
out:
    os_unfair_lock_unlock(&lock);
}

/* ---- DGEMV reuse probe (level REUSE) ------------------------------------- */

#define SITES 1024
#define RUN_BUCKETS 8  /* run lengths 1, 2, 3-4, 5-8, 9-16, 17-32, 33-64, >= 65 */

struct site_state {
    const void *site;
    int used, m, n, lda, run;
    uint64_t hash;
    const char *a, *y_lo, *y_hi;
    uint64_t calls, same_content, same_pointer_changed, feeds_next, runs[RUN_BUCKETS];
};

static struct site_state sites[SITES];
static uint64_t sites_dropped, feeds_from_previous_call;
static const char *prev_y_lo, *prev_y_hi;

static int bucket(int run)
{
    int b = 0;
    for (int len = 1; len < run && b < RUN_BUCKETS - 1; len *= 2)
        b++;
    return b;
}

/* Stored elements of the m x n matrix with leading dimension lda. */
static uint64_t matrix_hash(const double *a, int m, int n, int lda)
{
    uint64_t h = 1469598103934665603ull;
    for (int j = 0; j < n; j++) {
        const uint64_t *col = (const uint64_t *)(const void *)(a + (size_t)j * lda);
        for (int i = 0; i < m; i++)
            h = (h ^ col[i]) * 1099511628211ull;
    }
    return h;
}

static void span(const double *v, int len, int inc, const char **lo, const char **hi)
{
    size_t bytes = len > 0 ? ((size_t)(len - 1) * (size_t)(inc < 0 ? -inc : inc) + 1) * sizeof *v : 0;
    *lo = (const char *)v;
    *hi = *lo + bytes;
}

static int overlaps(const char *lo, const char *hi, const char *lo2, const char *hi2)
{
    return lo && lo2 && lo < hi2 && lo2 < hi;
}

static void reuse_probe(const void *site, char trans, int m, int n, const double *a, int lda,
                        const double *x, int incx, const double *y, int incy)
{
    int tr = trans != 'N' && trans != 'n';
    const char *x_lo, *x_hi, *y_lo, *y_hi;
    span(x, tr ? m : n, incx, &x_lo, &x_hi);
    span(y, tr ? n : m, incy, &y_lo, &y_hi);
    uint64_t h = matrix_hash(a, m, n, lda);
    os_unfair_lock_lock(&lock);
    if (overlaps(x_lo, x_hi, prev_y_lo, prev_y_hi))
        feeds_from_previous_call++;
    prev_y_lo = y_lo;
    prev_y_hi = y_hi;
    uint32_t slot = (uint32_t)(((uintptr_t)site >> 2) * 2654435761u);
    struct site_state *s = NULL;
    for (uint32_t probe = 0; probe < SITES; probe++) {
        struct site_state *c = &sites[(slot + probe) & (SITES - 1)];
        if (!c->used || c->site == site) {
            s = c;
            break;
        }
    }
    if (!s) {
        sites_dropped++;
        goto out;
    }
    if (!s->used) {
        *s = (struct site_state){.site = site, .used = 1};
    } else {
        int same_dims = s->m == m && s->n == n && s->lda == lda;
        int dep = overlaps(x_lo, x_hi, s->y_lo, s->y_hi);
        if (dep)
            s->feeds_next++;
        if (same_dims && s->hash == h) {
            s->same_content++;
        } else if (s->a == (const char *)a) {
            s->same_pointer_changed++;
        }
        if (same_dims && s->hash == h && !dep) {
            s->run++;
            goto keep;
        }
        s->runs[bucket(s->run)]++;
    }
    s->run = 1;
keep:
    s->calls++;
    s->m = m;
    s->n = n;
    s->lda = lda;
    s->hash = h;
    s->a = (const char *)a;
    s->y_lo = y_lo;
    s->y_hi = y_hi;
out:
    os_unfair_lock_unlock(&lock);
}

/* ---- interposers ---------------------------------------------------------- */

void dgemv_(const char *trans, const int *m, const int *n, const double *alpha,
            const double *a, const int *lda, const double *x, const int *incx,
            const double *beta, double *y, const int *incy, size_t len_trans)
{
    pthread_once(&resolve_once, resolve);
    int level = vmb_profile_level();
    if (level == VMB_PROFILE_OFF) {
        cpu_dgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy, len_trans);
        return;
    }
    const void *site = __builtin_return_address(0);
    if (level == VMB_PROFILE_REUSE && *m > 0 && *n > 0)
        reuse_probe(site, *trans, *m, *n, a, *lda, x, *incx, y, *incy);
    struct key k = {.site = site, .op = OP_DGEMV, .m = *m, .n = *n, .lda = *lda,
                    .ldb = *incx, .ldc = *incy, .ta = *trans, .tb = ' ',
                    .alpha_class = scalar_class(*alpha), .beta_class = scalar_class(*beta)};
    uint64_t t0 = vmb_ticks();
    cpu_dgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy, len_trans);
    record(&k, vmb_ticks() - t0);
}

void dgemm_(const char *transa, const char *transb, const int *m, const int *n, const int *k,
            const double *alpha, const double *a, const int *lda, const double *b,
            const int *ldb, const double *beta, double *c, const int *ldc, size_t len_transa,
            size_t len_transb)
{
    pthread_once(&resolve_once, resolve);
    if (vmb_profile_level() == VMB_PROFILE_OFF) {
        cpu_dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, len_transa,
                  len_transb);
        return;
    }
    struct key key = {.site = __builtin_return_address(0), .op = OP_DGEMM, .m = *m, .n = *n,
                      .k = *k, .lda = *lda, .ldb = *ldb, .ldc = *ldc, .ta = *transa,
                      .tb = *transb, .alpha_class = scalar_class(*alpha),
                      .beta_class = scalar_class(*beta)};
    uint64_t t0 = vmb_ticks();
    cpu_dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, len_transa,
              len_transb);
    record(&key, vmb_ticks() - t0);
}

/* ---- report section ------------------------------------------------------- */

static const char *const class_names[3] = {"zero", "one", "other"};

uint64_t vmb_blas_write_json(FILE *out)
{
    os_unfair_lock_lock(&lock);
    const char *lib = strrchr(blas_library, '/');
    fprintf(out, "{\"cpu_library\": ");
    vmb_json_string(out, blas_library[0] ? (lib ? lib + 1 : blas_library) : NULL);
    fprintf(out, ", \"totals\": {\"dgemv\": {\"calls\": %llu, \"seconds\": %.9f}, "
                 "\"dgemm\": {\"calls\": %llu, \"seconds\": %.9f}},\n  \"dropped_keys\": %llu,"
                 " \"shapes\": [",
            (unsigned long long)total_calls[OP_DGEMV], vmb_ticks_to_s(total_ticks[OP_DGEMV]),
            (unsigned long long)total_calls[OP_DGEMM], vmb_ticks_to_s(total_ticks[OP_DGEMM]),
            (unsigned long long)dropped);
    int first = 1;
    for (int i = 0; i < SLOTS; i++) {
        const struct entry *e = &table[i];
        if (!e->used)
            continue;
        const struct key *k = &e->key;
        fprintf(out, "%s\n   {\"op\": \"%s\", \"m\": %d, \"n\": %d, ", first ? "" : ",",
                k->op == OP_DGEMV ? "dgemv" : "dgemm", k->m, k->n);
        if (k->op == OP_DGEMV)
            fprintf(out, "\"trans\": \"%c\", \"lda\": %d, \"incx\": %d, \"incy\": %d, ", k->ta,
                    k->lda, k->ldb, k->ldc);
        else
            fprintf(out, "\"k\": %d, \"transa\": \"%c\", \"transb\": \"%c\", \"lda\": %d, "
                         "\"ldb\": %d, \"ldc\": %d, ",
                    k->k, k->ta, k->tb, k->lda, k->ldb, k->ldc);
        fprintf(out, "\"alpha\": \"%s\", \"beta\": \"%s\", \"calls\": %llu, \"seconds\": %.9f, "
                     "\"max_seconds\": %.9f, \"site\": ",
                class_names[k->alpha_class], class_names[k->beta_class],
                (unsigned long long)e->calls, vmb_ticks_to_s(e->ticks),
                vmb_ticks_to_s(e->max_ticks));
        vmb_site_write_json(out, k->site);
        fputc('}', out);
        first = 0;
    }
    fprintf(out, "]");
    if (vmb_profile_level() == VMB_PROFILE_REUSE) {
        fprintf(out, ",\n  \"reuse\": {\"run_buckets\": [\"1\", \"2\", \"3-4\", \"5-8\", "
                     "\"9-16\", \"17-32\", \"33-64\", \">=65\"], \"sites_dropped\": %llu, "
                     "\"feeds_from_previous_call\": %llu, \"sites\": [",
                (unsigned long long)sites_dropped, (unsigned long long)feeds_from_previous_call);
        first = 1;
        for (int i = 0; i < SITES; i++) {
            struct site_state *s = &sites[i];
            if (!s->used)
                continue;
            uint64_t runs[RUN_BUCKETS];
            memcpy(runs, s->runs, sizeof runs);
            runs[bucket(s->run)]++;  /* the open run */
            fprintf(out, "%s\n   {\"calls\": %llu, \"same_content\": %llu, "
                         "\"same_pointer_changed\": %llu, \"feeds_next\": %llu, \"runs\": [",
                    first ? "" : ",", (unsigned long long)s->calls,
                    (unsigned long long)s->same_content,
                    (unsigned long long)s->same_pointer_changed,
                    (unsigned long long)s->feeds_next);
            for (int b = 0; b < RUN_BUCKETS; b++)
                fprintf(out, "%s%llu", b ? ", " : "", (unsigned long long)runs[b]);
            fprintf(out, "], \"site\": ");
            vmb_site_write_json(out, s->site);
            fputc('}', out);
            first = 0;
        }
        fprintf(out, "]}");
    }
    fputc('}', out);
    uint64_t n = dropped;
    os_unfair_lock_unlock(&lock);
    return n;
}
