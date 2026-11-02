/* Opt-in aggregated ZGEMM shape telemetry: one row per distinct
   (m, n, k, trans, leading dimensions, beta class, outcome). Operand values and
   addresses are never recorded. */
#include "vmb.h"
#include "vmb_internal.h"

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>

#define SLOTS 4096  /* power of two; shapes beyond this are only counted */

struct entry {
    struct vmb_shape shape;
    int status, used;
    uint64_t count;
    double total, min, max;
};

static struct entry table[SLOTS];
static uint64_t dropped;
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

static int same(const struct vmb_shape *x, const struct vmb_shape *y)
{
    return x->m == y->m && x->n == y->n && x->k == y->k && x->lda == y->lda &&
           x->ldb == y->ldb && x->ldc == y->ldc && x->ta == y->ta && x->tb == y->tb &&
           x->beta_class == y->beta_class;
}

static uint32_t hash(const struct vmb_shape *s, int status)
{
    const int fields[] = {s->m, s->n, s->k, s->lda, s->ldb, s->ldc, s->ta, s->tb,
                          s->beta_class, status};
    uint32_t h = 2166136261u;  /* FNV-1a */
    for (size_t i = 0; i < sizeof fields / sizeof fields[0]; i++) {
        h ^= (uint32_t)fields[i];
        h *= 16777619u;
    }
    return h;
}

void vmb_telemetry_record(const struct vmb_shape *shape, int status, double seconds)
{
    pthread_mutex_lock(&lock);
    uint32_t h = hash(shape, status);
    for (uint32_t probe = 0; probe < SLOTS; probe++) {
        struct entry *e = &table[(h + probe) & (SLOTS - 1)];
        if (!e->used) {
            *e = (struct entry){.shape = *shape, .status = status, .used = 1,
                                .count = 1, .total = seconds, .min = seconds, .max = seconds};
            goto out;
        }
        if (e->status == status && same(&e->shape, shape)) {
            e->count++;
            e->total += seconds;
            if (seconds < e->min)
                e->min = seconds;
            if (seconds > e->max)
                e->max = seconds;
            goto out;
        }
    }
    dropped++;
out:
    pthread_mutex_unlock(&lock);
}

void vmb_telemetry_write(const char *path)
{
    static const char *const beta_names[3] = {"zero", "one", "other"};
    FILE *out = fopen(path, "w");
    if (!out) {
        fprintf(stderr, "vmb: cannot write telemetry to %s\n", path);
        return;
    }
    pthread_mutex_lock(&lock);
    fprintf(out, "# vmb shape telemetry v1; times are seconds per zgemm_ call (host entry "
                 "to return); dropped_shapes=%llu\n", (unsigned long long)dropped);
    fprintf(out, "m\tn\tk\ttransa\ttransb\tlda\tldb\tldc\tbeta\tstatus\tcount\t"
                 "total_s\tmin_s\tmax_s\n");
    for (int i = 0; i < SLOTS; i++) {
        const struct entry *e = &table[i];
        if (!e->used)
            continue;
        fprintf(out, "%d\t%d\t%d\t%c\t%c\t%d\t%d\t%d\t%s\t%d\t%llu\t%.9f\t%.9f\t%.9f\n",
                e->shape.m, e->shape.n, e->shape.k, e->shape.ta, e->shape.tb, e->shape.lda,
                e->shape.ldb, e->shape.ldc, beta_names[e->shape.beta_class], e->status,
                (unsigned long long)e->count, e->total, e->min, e->max);
    }
    pthread_mutex_unlock(&lock);
    fclose(out);
}
