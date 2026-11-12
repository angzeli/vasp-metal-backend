/* v0.5 FFTW telemetry interposers for the legacy Fortran interface VASP calls.
 *
 * Every entry point is forwarded, unchanged, to the loaded libfftw3 image; plans
 * are FFTW's own. With profiling active, each plan is mapped to its problem
 * shape (kind, sizes, batch, strides, sign, flags, in-place, call site) and the
 * table counts plans created and destroyed, planning time, executions and
 * execution time per shape. Many plans of one shape mean repeated planning.
 * No array contents or addresses are written.
 *
 * Opt-in plan cache (VMB_FFTW_PLAN_CACHE=1, or a build with PLAN_CACHE_DEFAULT=1).
 * VASP plans a 3D transform with FFTW_ESTIMATE, executes it once and destroys it,
 * every time. With the cache, dfftw_plan_dft_3d_ for an ESTIMATE plan returns a
 * handle onto a plan kept for the problem (sizes, sign, flags, in-place or not,
 * and the FFTW alignment of both arrays): FFTW's new-array execute rule makes
 * one plan valid for every array pair that matches those. The handle remembers
 * the arrays it was planned for, so dfftw_execute_ still transforms them, and
 * destroying it releases only the handle. ESTIMATE planning never touches the
 * arrays, so skipping it has no side effect. Everything else, and any request
 * the bounded tables cannot hold, goes to FFTW's planner unchanged. */
#define _DARWIN_C_SOURCE
#include "vmb.h"
#include "vmb_internal.h"

#include <os/lock.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef void *plan_t;  /* FFTW's plan; Fortran holds it in an INTEGER*8 */

enum { K_DFT_1D, K_DFT_3D, K_R2C_3D, K_C2R_3D, K_MANY_DFT, K_MANY_R2C, K_MANY_C2R, K_KINDS };
static const char *const kind_names[K_KINDS] = {
    "dft_1d", "dft_3d", "dft_r2c_3d", "dft_c2r_3d", "many_dft", "many_dft_r2c", "many_dft_c2r",
};
static const char *const entry_names[K_KINDS] = {
    "dfftw_plan_dft_1d_", "dfftw_plan_dft_3d_", "dfftw_plan_dft_r2c_3d_",
    "dfftw_plan_dft_c2r_3d_", "dfftw_plan_many_dft_", "dfftw_plan_many_dft_r2c_",
    "dfftw_plan_many_dft_c2r_",
};
enum { E_EXECUTE, E_DFT, E_R2C, E_C2R, E_DESTROY, E_ENTRIES };
static const char *const exec_names[E_ENTRIES] = {
    "dfftw_execute_", "dfftw_execute_dft_", "dfftw_execute_dft_r2c_", "dfftw_execute_dft_c2r_",
    "dfftw_destroy_plan_",
};

static void *plan_fn[K_KINDS], *exec_fn[E_ENTRIES];
static int (*alignment_of)(double *);
static char fftw_library[1024];
static pthread_once_t resolve_once = PTHREAD_ONCE_INIT;

void dfftw_plan_dft_1d_(plan_t *, const int *, void *, void *, const int *, const int *);
void dfftw_plan_dft_3d_(plan_t *, const int *, const int *, const int *, void *, void *,
                        const int *, const int *);
void dfftw_plan_dft_r2c_3d_(plan_t *, const int *, const int *, const int *, void *, void *,
                            const int *);
void dfftw_plan_dft_c2r_3d_(plan_t *, const int *, const int *, const int *, void *, void *,
                            const int *);
void dfftw_plan_many_dft_(plan_t *, const int *, const int *, const int *, void *, const int *,
                          const int *, const int *, void *, const int *, const int *,
                          const int *, const int *, const int *);
void dfftw_plan_many_dft_r2c_(plan_t *, const int *, const int *, const int *, void *,
                              const int *, const int *, const int *, void *, const int *,
                              const int *, const int *, const int *);
void dfftw_plan_many_dft_c2r_(plan_t *, const int *, const int *, const int *, void *,
                              const int *, const int *, const int *, void *, const int *,
                              const int *, const int *, const int *);
void dfftw_execute_(const plan_t *);
void dfftw_execute_dft_(const plan_t *, void *, void *);
void dfftw_execute_dft_r2c_(const plan_t *, void *, void *);
void dfftw_execute_dft_c2r_(const plan_t *, void *, void *);
void dfftw_destroy_plan_(plan_t *);

static void resolve(void)
{
    const void *self_plan[K_KINDS] = {
        (const void *)dfftw_plan_dft_1d_, (const void *)dfftw_plan_dft_3d_,
        (const void *)dfftw_plan_dft_r2c_3d_, (const void *)dfftw_plan_dft_c2r_3d_,
        (const void *)dfftw_plan_many_dft_, (const void *)dfftw_plan_many_dft_r2c_,
        (const void *)dfftw_plan_many_dft_c2r_,
    };
    const void *self_exec[E_ENTRIES] = {
        (const void *)dfftw_execute_, (const void *)dfftw_execute_dft_,
        (const void *)dfftw_execute_dft_r2c_, (const void *)dfftw_execute_dft_c2r_,
        (const void *)dfftw_destroy_plan_,
    };
    int ok = 1;
    for (int i = 0; i < K_KINDS; i++)
        ok &= (plan_fn[i] = vmb_find_symbol("libfftw3.", entry_names[i], self_plan[i],
                                            fftw_library, sizeof fftw_library)) != NULL;
    for (int i = 0; i < E_ENTRIES; i++)
        ok &= (exec_fn[i] = vmb_find_symbol("libfftw3.", exec_names[i], self_exec[i], NULL, 0)) != NULL;
    ok &= (*(void **)&alignment_of = vmb_find_symbol("libfftw3.", "fftw_alignment_of", NULL,
                                                     NULL, 0)) != NULL;
    if (!ok) {
        fprintf(stderr, "vmb: libfftw3 Fortran entry points not found among loaded images; "
                        "link libvmb.a ahead of -lfftw3\n");
        abort();
    }
}

/* ---- shape table and plan map --------------------------------------------- */

struct shape {
    const void *site;
    int kind, rank, n[3], howmany, istride, idist, ostride, odist, sign, flags, inplace;
};

struct entry {
    struct shape shape;
    int used;
    uint64_t plans, destroyed, plan_ticks, max_plan_ticks, execs, new_array_execs, exec_ticks,
        max_exec_ticks;
};

#define SHAPES 1024
#define PLANS 16384  /* live plans tracked; open addressing with tombstones */
static struct entry table[SHAPES];
static struct {
    plan_t plan;
    int shape;  /* -1 empty, -2 tombstone */
} plans[PLANS];
static uint64_t dropped_shapes, untracked_plans, unknown_execs, plan_ticks_total,
    exec_ticks_total, execs_total, plans_total;
static int plans_ready;
static os_unfair_lock lock = OS_UNFAIR_LOCK_INIT;

static uint32_t mix(uint64_t h) { return (uint32_t)(h ^ (h >> 29)) * 2654435761u; }

static int shape_same(const struct shape *x, const struct shape *y)
{
    return memcmp(x, y, sizeof *x) == 0;  /* zero-initialised, so padding is equal */
}

static int shape_index(const struct shape *s)
{
    uint64_t h = 1469598103934665603ull ^ (uint64_t)(uintptr_t)s->site;
    const int *f = &s->kind;
    for (int i = 0; i < 13; i++)
        h = (h ^ (uint32_t)f[i]) * 1099511628211ull;
    uint32_t start = mix(h);
    for (uint32_t probe = 0; probe < SHAPES; probe++) {
        int i = (int)((start + probe) & (SHAPES - 1));
        if (!table[i].used) {
            memset(&table[i], 0, sizeof table[i]);
            table[i].shape = *s;
            table[i].used = 1;
            return i;
        }
        if (shape_same(&table[i].shape, s))
            return i;
    }
    dropped_shapes++;
    return -1;
}

static void plans_init(void)
{
    if (plans_ready)
        return;
    for (int i = 0; i < PLANS; i++)
        plans[i].shape = -1;
    plans_ready = 1;
}

static uint32_t plan_slot(plan_t p) { return mix((uint64_t)(uintptr_t)p >> 4); }

static void plan_put(plan_t p, int shape)
{
    plans_init();
    uint32_t start = plan_slot(p);
    int tomb = -1;
    for (uint32_t probe = 0; probe < PLANS; probe++) {
        uint32_t i = (start + probe) & (PLANS - 1);
        if (plans[i].shape == -2 && tomb < 0)
            tomb = (int)i;
        else if (plans[i].shape == -1 || plans[i].plan == p) {
            if (plans[i].shape == -1 && tomb >= 0)
                i = (uint32_t)tomb;
            plans[i].plan = p;
            plans[i].shape = shape;
            return;
        }
    }
    if (tomb >= 0) {
        plans[tomb].plan = p;
        plans[tomb].shape = shape;
        return;
    }
    untracked_plans++;
}

/* Shape index of a live plan, or -1. With remove, the plan is forgotten. */
static int plan_get(plan_t p, int remove)
{
    plans_init();
    uint32_t start = plan_slot(p);
    for (uint32_t probe = 0; probe < PLANS; probe++) {
        uint32_t i = (start + probe) & (PLANS - 1);
        if (plans[i].shape == -1)
            return -1;
        if (plans[i].shape >= 0 && plans[i].plan == p) {
            int s = plans[i].shape;
            if (remove)
                plans[i].shape = -2;
            return s;
        }
    }
    return -1;
}

static void after_plan(const struct shape *s, plan_t p, uint64_t ticks)
{
    os_unfair_lock_lock(&lock);
    plans_total++;
    plan_ticks_total += ticks;
    int i = shape_index(s);
    if (i >= 0) {
        table[i].plans++;
        table[i].plan_ticks += ticks;
        if (ticks > table[i].max_plan_ticks)
            table[i].max_plan_ticks = ticks;
    }
    if (p && i >= 0)
        plan_put(p, i);
    else if (p)
        untracked_plans++;
    os_unfair_lock_unlock(&lock);
}

static void after_exec(plan_t p, int new_array, uint64_t ticks)
{
    os_unfair_lock_lock(&lock);
    execs_total++;
    exec_ticks_total += ticks;
    int i = plan_get(p, 0);
    if (i >= 0) {
        table[i].execs++;
        table[i].new_array_execs += (uint64_t)new_array;
        table[i].exec_ticks += ticks;
        if (ticks > table[i].max_exec_ticks)
            table[i].max_exec_ticks = ticks;
    } else {
        unknown_execs++;
    }
    os_unfair_lock_unlock(&lock);
}

/* ---- plan cache ------------------------------------------------------------ */

#ifndef VMB_PLAN_CACHE_DEFAULT
#define VMB_PLAN_CACHE_DEFAULT 0  /* build-time default for VMB_FFTW_PLAN_CACHE */
#endif
#define CONFIG_ERROR 78
/* FFTW planner flags (fftw3.h), kept here so this file needs no FFTW headers. */
#define F_ESTIMATE (1U << 6)
#define F_SLOW_PLANNERS ((1U << 3) | (1U << 5) | (1U << 21))  /* EXHAUSTIVE, PATIENT, WISDOM_ONLY */

static int cache_on;

__attribute__((constructor)) static void plan_cache_init(void)
{
    const char *v = getenv("VMB_FFTW_PLAN_CACHE");
    if (!v)
        cache_on = VMB_PLAN_CACHE_DEFAULT;
    else if (strcmp(v, "0") == 0 || strcmp(v, "1") == 0)
        cache_on = v[0] == '1';
    else {
        fprintf(stderr, "vmb: VMB_FFTW_PLAN_CACHE must be 0 or 1\n");
        exit(CONFIG_ERROR);
    }
}

struct cached {
    int used, n[3], sign, flags, inplace, align_in, align_out;
    plan_t real;
    uint64_t live;
};
struct handle {
    plan_t real;
    void *in, *out;
    int entry, next;
};

#define CACHED 64     /* distinct problems kept */
#define HANDLES 1024  /* handles alive at once (VASP holds one or two) */
static struct cached cache[CACHED];
static struct handle handles[HANDLES];
static int free_handle = -1, handles_ready;
static uint64_t cache_hits, cache_misses, cache_bypassed;
static os_unfair_lock cache_lock = OS_UNFAIR_LOCK_INIT;

/* The handle behind a plan value, or NULL for one of FFTW's own plans. */
static struct handle *as_handle(plan_t p)
{
    uintptr_t a = (uintptr_t)p, base = (uintptr_t)handles;
    if (a < base || a >= base + sizeof handles || (a - base) % sizeof(struct handle))
        return NULL;
    return (struct handle *)p;
}

static struct handle *handle_new(void)
{
    if (!handles_ready) {
        for (int i = 0; i < HANDLES; i++)
            handles[i].next = i + 1 < HANDLES ? i + 1 : -1;
        free_handle = 0;
        handles_ready = 1;
    }
    if (free_handle < 0)
        return NULL;
    struct handle *h = &handles[free_handle];
    free_handle = h->next;
    return h;
}

static void handle_free(struct handle *h)
{
    h->real = NULL;
    h->in = h->out = NULL;
    h->next = free_handle;
    free_handle = (int)(h - handles);
}

typedef void (*plan_3d_fn)(plan_t *, const int *, const int *, const int *, void *, void *,
                           const int *, const int *);

/* dfftw_plan_dft_3d_ through the cache. Returns 0 when the request is not
   cacheable or does not fit, and the caller plans it with FFTW. */
static int plan_3d_cached(plan_t *p, const int *nx, const int *ny, const int *nz, void *in,
                          void *out, const int *sign, const int *flags)
{
    if (!(*flags & F_ESTIMATE) || (*flags & F_SLOW_PLANNERS))
        return 0;
    struct cached key = {.used = 1, .n = {*nx, *ny, *nz}, .sign = *sign, .flags = *flags,
                         .inplace = in == out, .align_in = alignment_of(in),
                         .align_out = alignment_of(out)};
    os_unfair_lock_lock(&cache_lock);
    struct handle *h = handle_new();
    int slot = -1, empty = -1;
    for (int i = 0; h && i < CACHED; i++) {
        struct cached *c = &cache[i];
        if (!c->used) {
            if (empty < 0)
                empty = i;
        } else if (memcmp(c->n, key.n, sizeof key.n) == 0 && c->sign == key.sign &&
                   c->flags == key.flags && c->inplace == key.inplace &&
                   c->align_in == key.align_in && c->align_out == key.align_out) {
            slot = i;
            break;
        }
    }
    if (!h || (slot < 0 && empty < 0)) {
        if (h)
            handle_free(h);
        cache_bypassed++;
        os_unfair_lock_unlock(&cache_lock);
        return 0;
    }
    if (slot < 0) {
        plan_t real = NULL;
        ((plan_3d_fn)plan_fn[K_DFT_3D])(&real, nx, ny, nz, in, out, sign, flags);
        if (!real) {  /* FFTW could not plan it: report that as FFTW would */
            handle_free(h);
            os_unfair_lock_unlock(&cache_lock);
            *p = NULL;
            return 1;
        }
        key.real = real;
        cache[empty] = key;
        slot = empty;
        cache_misses++;
    } else {
        cache_hits++;
    }
    cache[slot].live++;
    h->real = cache[slot].real;
    h->in = in;
    h->out = out;
    h->entry = slot;
    *p = (plan_t)h;
    os_unfair_lock_unlock(&cache_lock);
    return 1;
}

/* Report section: whether the cache is on and how it was used. */
void vmb_fftw_cache_write_json(FILE *out)
{
    os_unfair_lock_lock(&cache_lock);
    uint64_t live = 0;
    int entries = 0;
    for (int i = 0; i < CACHED; i++)
        if (cache[i].used) {
            entries++;
            live += cache[i].live;
        }
    fprintf(out, "{\"enabled\": %s, \"hits\": %llu, \"misses\": %llu, \"bypassed\": %llu, "
                 "\"entries\": %d, \"live_handles\": %llu}",
            cache_on ? "true" : "false", (unsigned long long)cache_hits,
            (unsigned long long)cache_misses, (unsigned long long)cache_bypassed, entries,
            (unsigned long long)live);
    os_unfair_lock_unlock(&cache_lock);
}

/* ---- interposers ---------------------------------------------------------- */

#define PROFILING (pthread_once(&resolve_once, resolve), vmb_profile_level() != VMB_PROFILE_OFF)

static void base_shape(struct shape *s, int kind, const void *site, const int *sign,
                       const int *flags, const void *in, const void *out)
{
    memset(s, 0, sizeof *s);
    s->site = site;
    s->kind = kind;
    s->howmany = 1;
    s->sign = sign ? *sign : 0;
    s->flags = *flags;
    s->inplace = in == out;
}

void dfftw_plan_dft_1d_(plan_t *p, const int *n, void *in, void *out, const int *sign,
                        const int *flags)
{
    typedef void (*fn)(plan_t *, const int *, void *, void *, const int *, const int *);
    if (!PROFILING) {
        ((fn)plan_fn[K_DFT_1D])(p, n, in, out, sign, flags);
        return;
    }
    struct shape s;
    base_shape(&s, K_DFT_1D, __builtin_return_address(0), sign, flags, in, out);
    s.rank = 1;
    s.n[0] = *n;
    uint64_t t0 = vmb_ticks();
    ((fn)plan_fn[K_DFT_1D])(p, n, in, out, sign, flags);
    after_plan(&s, *p, vmb_ticks() - t0);
}

void dfftw_plan_dft_3d_(plan_t *p, const int *nx, const int *ny, const int *nz, void *in,
                        void *out, const int *sign, const int *flags)
{
    if (!PROFILING) {
        if (!cache_on || !plan_3d_cached(p, nx, ny, nz, in, out, sign, flags))
            ((plan_3d_fn)plan_fn[K_DFT_3D])(p, nx, ny, nz, in, out, sign, flags);
        return;
    }
    struct shape s;
    base_shape(&s, K_DFT_3D, __builtin_return_address(0), sign, flags, in, out);
    s.rank = 3;
    s.n[0] = *nx;
    s.n[1] = *ny;
    s.n[2] = *nz;
    uint64_t t0 = vmb_ticks();  /* with the cache, a hit's lookup time */
    if (!cache_on || !plan_3d_cached(p, nx, ny, nz, in, out, sign, flags))
        ((plan_3d_fn)plan_fn[K_DFT_3D])(p, nx, ny, nz, in, out, sign, flags);
    after_plan(&s, *p, vmb_ticks() - t0);
}

static void plan_real_3d(int kind, plan_t *p, const int *nx, const int *ny, const int *nz,
                         void *in, void *out, const int *flags, const void *site)
{
    typedef void (*fn)(plan_t *, const int *, const int *, const int *, void *, void *,
                       const int *);
    if (!PROFILING) {
        ((fn)plan_fn[kind])(p, nx, ny, nz, in, out, flags);
        return;
    }
    struct shape s;
    base_shape(&s, kind, site, NULL, flags, in, out);
    s.rank = 3;
    s.n[0] = *nx;
    s.n[1] = *ny;
    s.n[2] = *nz;
    uint64_t t0 = vmb_ticks();
    ((fn)plan_fn[kind])(p, nx, ny, nz, in, out, flags);
    after_plan(&s, *p, vmb_ticks() - t0);
}

void dfftw_plan_dft_r2c_3d_(plan_t *p, const int *nx, const int *ny, const int *nz, void *in,
                            void *out, const int *flags)
{
    plan_real_3d(K_R2C_3D, p, nx, ny, nz, in, out, flags, __builtin_return_address(0));
}

void dfftw_plan_dft_c2r_3d_(plan_t *p, const int *nx, const int *ny, const int *nz, void *in,
                            void *out, const int *flags)
{
    plan_real_3d(K_C2R_3D, p, nx, ny, nz, in, out, flags, __builtin_return_address(0));
}

static void many_shape(struct shape *s, const int *rank, const int *n, const int *howmany,
                       const int *istride, const int *idist, const int *ostride,
                       const int *odist)
{
    s->rank = *rank;
    for (int i = 0; i < 3 && i < *rank; i++)
        s->n[i] = n[i];
    if (*rank > 3)
        s->n[2] = -1;  /* more than three dimensions: sizes beyond the first two not kept */
    s->howmany = *howmany;
    s->istride = *istride;
    s->idist = *idist;
    s->ostride = *ostride;
    s->odist = *odist;
}

void dfftw_plan_many_dft_(plan_t *p, const int *rank, const int *n, const int *howmany,
                          void *in, const int *inembed, const int *istride, const int *idist,
                          void *out, const int *onembed, const int *ostride, const int *odist,
                          const int *sign, const int *flags)
{
    typedef void (*fn)(plan_t *, const int *, const int *, const int *, void *, const int *,
                       const int *, const int *, void *, const int *, const int *,
                       const int *, const int *, const int *);
    if (!PROFILING) {
        ((fn)plan_fn[K_MANY_DFT])(p, rank, n, howmany, in, inembed, istride, idist, out,
                                  onembed, ostride, odist, sign, flags);
        return;
    }
    struct shape s;
    base_shape(&s, K_MANY_DFT, __builtin_return_address(0), sign, flags, in, out);
    many_shape(&s, rank, n, howmany, istride, idist, ostride, odist);
    uint64_t t0 = vmb_ticks();
    ((fn)plan_fn[K_MANY_DFT])(p, rank, n, howmany, in, inembed, istride, idist, out, onembed,
                              ostride, odist, sign, flags);
    after_plan(&s, *p, vmb_ticks() - t0);
}

static void plan_many_real(int kind, plan_t *p, const int *rank, const int *n,
                           const int *howmany, void *in, const int *inembed,
                           const int *istride, const int *idist, void *out,
                           const int *onembed, const int *ostride, const int *odist,
                           const int *flags, const void *site)
{
    typedef void (*fn)(plan_t *, const int *, const int *, const int *, void *, const int *,
                       const int *, const int *, void *, const int *, const int *,
                       const int *, const int *);
    if (!PROFILING) {
        ((fn)plan_fn[kind])(p, rank, n, howmany, in, inembed, istride, idist, out, onembed,
                            ostride, odist, flags);
        return;
    }
    struct shape s;
    base_shape(&s, kind, site, NULL, flags, in, out);
    many_shape(&s, rank, n, howmany, istride, idist, ostride, odist);
    uint64_t t0 = vmb_ticks();
    ((fn)plan_fn[kind])(p, rank, n, howmany, in, inembed, istride, idist, out, onembed, ostride,
                        odist, flags);
    after_plan(&s, *p, vmb_ticks() - t0);
}

void dfftw_plan_many_dft_r2c_(plan_t *p, const int *rank, const int *n, const int *howmany,
                              void *in, const int *inembed, const int *istride,
                              const int *idist, void *out, const int *onembed,
                              const int *ostride, const int *odist, const int *flags)
{
    plan_many_real(K_MANY_R2C, p, rank, n, howmany, in, inembed, istride, idist, out, onembed,
                   ostride, odist, flags, __builtin_return_address(0));
}

void dfftw_plan_many_dft_c2r_(plan_t *p, const int *rank, const int *n, const int *howmany,
                              void *in, const int *inembed, const int *istride,
                              const int *idist, void *out, const int *onembed,
                              const int *ostride, const int *odist, const int *flags)
{
    plan_many_real(K_MANY_C2R, p, rank, n, howmany, in, inembed, istride, idist, out, onembed,
                   ostride, odist, flags, __builtin_return_address(0));
}

/* A handle executes its kept plan on the arrays it was planned for. */
static void execute(const plan_t *p)
{
    typedef void (*fn)(const plan_t *);
    typedef void (*fn_new)(const plan_t *, void *, void *);
    const struct handle *h = as_handle(*p);
    if (h)
        ((fn_new)exec_fn[E_DFT])(&h->real, h->in, h->out);
    else
        ((fn)exec_fn[E_EXECUTE])(p);
}

void dfftw_execute_(const plan_t *p)
{
    if (!PROFILING) {
        execute(p);
        return;
    }
    uint64_t t0 = vmb_ticks();
    execute(p);
    after_exec(*p, 0, vmb_ticks() - t0);
}

static void exec_new(int entry, const plan_t *p, void *in, void *out)
{
    typedef void (*fn)(const plan_t *, void *, void *);
    const struct handle *h = as_handle(*p);
    const plan_t *real = h ? &h->real : p;
    if (!PROFILING) {
        ((fn)exec_fn[entry])(real, in, out);
        return;
    }
    uint64_t t0 = vmb_ticks();
    ((fn)exec_fn[entry])(real, in, out);
    after_exec(*p, 1, vmb_ticks() - t0);
}

void dfftw_execute_dft_(const plan_t *p, void *in, void *out) { exec_new(E_DFT, p, in, out); }
void dfftw_execute_dft_r2c_(const plan_t *p, void *in, void *out) { exec_new(E_R2C, p, in, out); }
void dfftw_execute_dft_c2r_(const plan_t *p, void *in, void *out) { exec_new(E_C2R, p, in, out); }

void dfftw_destroy_plan_(plan_t *p)
{
    typedef void (*fn)(plan_t *);
    if (PROFILING) {
        os_unfair_lock_lock(&lock);
        int i = plan_get(*p, 1);
        if (i >= 0)
            table[i].destroyed++;
        os_unfair_lock_unlock(&lock);
    }
    struct handle *h = as_handle(*p);
    if (!h) {
        ((fn)exec_fn[E_DESTROY])(p);
        return;
    }
    os_unfair_lock_lock(&cache_lock);  /* the kept plan stays for the next request */
    cache[h->entry].live--;
    handle_free(h);
    os_unfair_lock_unlock(&cache_lock);
}

/* ---- report section ------------------------------------------------------- */

uint64_t vmb_fftw_write_json(FILE *out)
{
    os_unfair_lock_lock(&lock);
    const char *lib = strrchr(fftw_library, '/');
    fprintf(out, "{\"library\": ");
    vmb_json_string(out, fftw_library[0] ? (lib ? lib + 1 : fftw_library) : NULL);
    fprintf(out, ", \"totals\": {\"plans\": %llu, \"plan_seconds\": %.9f, \"executes\": %llu, "
                 "\"execute_seconds\": %.9f},\n  \"dropped_shapes\": %llu, "
                 "\"untracked_plans\": %llu, \"unknown_executes\": %llu, \"shapes\": [",
            (unsigned long long)plans_total, vmb_ticks_to_s(plan_ticks_total),
            (unsigned long long)execs_total, vmb_ticks_to_s(exec_ticks_total),
            (unsigned long long)dropped_shapes, (unsigned long long)untracked_plans,
            (unsigned long long)unknown_execs);
    int first = 1;
    for (int i = 0; i < SHAPES; i++) {
        const struct entry *e = &table[i];
        if (!e->used)
            continue;
        const struct shape *s = &e->shape;
        fprintf(out, "%s\n   {\"kind\": \"%s\", \"rank\": %d, \"n\": [%d, %d, %d], "
                     "\"howmany\": %d, \"istride\": %d, \"idist\": %d, \"ostride\": %d, "
                     "\"odist\": %d, \"sign\": %d, \"flags\": %d, \"inplace\": %s, "
                     "\"plans\": %llu, \"destroyed\": %llu, \"plan_seconds\": %.9f, "
                     "\"max_plan_seconds\": %.9f, \"executes\": %llu, \"new_array_executes\": %llu, "
                     "\"execute_seconds\": %.9f, \"max_execute_seconds\": %.9f, \"site\": ",
                first ? "" : ",", kind_names[s->kind], s->rank, s->n[0], s->n[1], s->n[2],
                s->howmany, s->istride, s->idist, s->ostride, s->odist, s->sign, s->flags,
                s->inplace ? "true" : "false", (unsigned long long)e->plans,
                (unsigned long long)e->destroyed, vmb_ticks_to_s(e->plan_ticks),
                vmb_ticks_to_s(e->max_plan_ticks), (unsigned long long)e->execs,
                (unsigned long long)e->new_array_execs, vmb_ticks_to_s(e->exec_ticks),
                vmb_ticks_to_s(e->max_exec_ticks));
        vmb_site_write_json(out, s->site);
        fputc('}', out);
        first = 0;
    }
    fprintf(out, "]}");
    uint64_t n = dropped_shapes;
    os_unfair_lock_unlock(&lock);
    return n;
}
