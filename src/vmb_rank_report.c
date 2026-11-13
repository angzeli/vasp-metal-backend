/* Per-rank reports for MPI runs (VMB_REPORT_DIR; docs/CONTRACT-v0.3.md).
 *
 * The route is configured in a library constructor so that a rank which never
 * calls ZGEMM still claims its rank and publishes a zero-call report. Rank and
 * world size come from the Open MPI launcher environment; no MPI library is
 * used. Metal is never initialised here. Every file is created through a held
 * directory descriptor with O_EXCL|O_NOFOLLOW and published by hard link, so a
 * report is never overwritten and never redirected through a symbolic link. */
#define _DARWIN_C_SOURCE
#include "vmb.h"
#include "vmb_internal.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <dlfcn.h>
#include <mach/mach_time.h>
#include <math.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

#ifndef VMB_SOURCE_COMMIT
#define VMB_SOURCE_COMMIT "unknown"
#endif
#ifndef VMB_SOURCE_DIGEST
#define VMB_SOURCE_DIGEST "unknown"
#endif
#ifndef VMB_SOURCE_DIRTY
#define VMB_SOURCE_DIRTY 1
#endif

#ifndef VMB_PROFILE_DEFAULT
#define VMB_PROFILE_DEFAULT 0  /* build-time default for VMB_PROFILE: 0, 1 or 2 (reuse) */
#endif

#define CONFIG_ERROR 78  /* EX_CONFIG */
#define MAX_WORLD 4096

static struct {
    int active, telemetry, dir_fd;
    pid_t owner;
    int rank, world, local_rank, local_size;
    double start;
    char dir[PATH_MAX], attempt[65];
} rr = {.dir_fd = -1, .local_rank = -1, .local_size = -1};

int vmb_rank_telemetry(void) { return rr.active && rr.telemetry; }

/* Set once in the constructor, before the program runs; read on every call. */
static int profile_level;

int vmb_profile_level(void) { return profile_level; }

const char *vmb_profile_level_name(int level)
{
    static const char *const names[3] = {"off", "profile", "reuse"};
    return names[level];
}

static mach_timebase_info_data_t timebase;

uint64_t vmb_ticks(void) { return mach_absolute_time(); }

double vmb_ticks_to_s(uint64_t ticks)
{
    if (timebase.denom == 0)
        mach_timebase_info(&timebase);
    return (double)ticks * timebase.numer / timebase.denom * 1e-9;
}

static double wall_now(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

static void fail(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    fputs("vmb: report configuration error: ", stderr);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
    _exit(CONFIG_ERROR);
}

/* Canonical non-negative decimal: no sign, space or leading zero. -1 if absent,
   -2 if malformed or too large. */
static long env_index(const char *name)
{
    const char *s = getenv(name);
    if (!s)
        return -1;
    size_t n = strlen(s);
    if (n == 0 || n > 6 || (n > 1 && s[0] == '0'))
        return -2;
    long v = 0;
    for (size_t i = 0; i < n; i++) {
        if (s[i] < '0' || s[i] > '9')
            return -2;
        v = 10 * v + (s[i] - '0');
    }
    return v;
}

static int valid_attempt(const char *s)
{
    size_t n = strlen(s);
    if (n < 1 || n > 64)
        return 0;
    for (size_t i = 0; i < n; i++) {
        char c = s[i];
        int alnum = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9');
        if (!alnum && (i == 0 || (c != '.' && c != '_' && c != '-')))
            return 0;
    }
    return 1;
}

/* World size and rank from Open MPI; singleton when neither is set. */
static void read_rank(int *rank, int *world)
{
    long r = env_index("OMPI_COMM_WORLD_RANK"), w = env_index("OMPI_COMM_WORLD_SIZE");
    if (r == -2 || w == -2)
        fail("malformed OMPI_COMM_WORLD_RANK or OMPI_COMM_WORLD_SIZE");
    if ((r == -1) != (w == -1))
        fail("only one of OMPI_COMM_WORLD_RANK and OMPI_COMM_WORLD_SIZE is set");
    if (r == -1) {
        r = 0;
        w = 1;
        if (getenv("PMIX_RANK"))
            fail("PMIX_RANK is set without the Open MPI rank variables");
    }
    if (w < 1 || w > MAX_WORLD || r >= w)
        fail("rank %ld outside world size %ld", r, w);
    long p = env_index("PMIX_RANK");
    if (p == -2 || (p >= 0 && p != r))
        fail("PMIX_RANK contradicts OMPI_COMM_WORLD_RANK");
    long lr = env_index("OMPI_COMM_WORLD_LOCAL_RANK"), ls = env_index("OMPI_COMM_WORLD_LOCAL_SIZE");
    if (lr == -2 || ls == -2 || (lr >= 0 && (ls < 1 || lr >= ls || ls > w)))
        fail("malformed or contradictory Open MPI local rank");
    rr.local_rank = (int)lr;
    rr.local_size = (int)ls;
    *rank = (int)r;
    *world = (int)w;
}

static int create_exclusive(const char *name)
{
    return openat(rr.dir_fd, name, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
}

void vmb_json_string(FILE *out, const char *s)
{
    if (!s) {
        fputs("null", out);
        return;
    }
    fputc('"', out);
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        if (*p == '"' || *p == '\\')
            fprintf(out, "\\%c", *p);
        else if (*p < 0x20)
            fprintf(out, "\\u%04x", *p);
        else
            fputc(*p, out);
    }
    fputc('"', out);
}

void vmb_site_write_json(FILE *out, const void *addr)
{
    Dl_info info;
    if (!addr || !dladdr(addr, &info) || !info.dli_fname) {
        fputs("null", out);
        return;
    }
    const char *base = strrchr(info.dli_fname, '/');
    fprintf(out, "{\"image\": ");
    vmb_json_string(out, base ? base + 1 : info.dli_fname);
    fprintf(out, ", \"offset\": %llu, \"symbol\": ",
            (unsigned long long)((const char *)addr - (const char *)info.dli_fbase));
    vmb_json_string(out, info.dli_sname);
    if (info.dli_sname && info.dli_saddr)
        fprintf(out, ", \"symbol_offset\": %llu}",
                (unsigned long long)((const char *)addr - (const char *)info.dli_saddr));
    else
        fprintf(out, ", \"symbol_offset\": null}");
}

/* Publish the stream written by fill() as NAME: temporary file, fsync, then a
   hard link that fails rather than replace an existing name. */
static int publish(const char *name, void (*fill)(FILE *, void *), void *arg)
{
    char tmp[96];
    snprintf(tmp, sizeof tmp, ".%s.%d.tmp", name, (int)getpid());
    int fd = create_exclusive(tmp);
    if (fd < 0)
        return -1;
    FILE *out = fdopen(fd, "w");
    if (!out) {
        close(fd);
        return -1;
    }
    fill(out, arg);
    int ok = fflush(out) == 0 && !ferror(out) && fsync(fd) == 0;
    ok = fclose(out) == 0 && ok;
    if (ok && linkat(rr.dir_fd, tmp, rr.dir_fd, name, 0) != 0)
        ok = 0;
    if (ok)
        unlinkat(rr.dir_fd, tmp, 0);  /* a failed publication keeps its temporary file */
    fsync(rr.dir_fd);
    return ok ? 0 : -1;
}

static void write_claim(FILE *out, void *arg)
{
    (void)arg;
    fprintf(out, "{\"schema\": \"vmb.rank-claim.v1\", \"attempt_id\": ");
    vmb_json_string(out, rr.attempt);
    fprintf(out, ", \"rank\": %d, \"world_size\": %d, \"pid\": %d, \"start_unix\": %.6f}\n",
            rr.rank, rr.world, (int)rr.owner, rr.start);
}

static void write_telemetry(FILE *out, void *arg)
{
    *(uint64_t *)arg = vmb_telemetry_write_stream(out);
}

struct report_args {
    const char *telemetry_file, *profile_file;
    uint64_t dropped;
};

static void write_profile(FILE *out, void *arg)
{
    (void)arg;
    fprintf(out, "{\"schema\": \"vmb.rank-profile.v1\", \"complete\": true, \"attempt_id\": ");
    vmb_json_string(out, rr.attempt);
    fprintf(out, ", \"rank\": %d, \"world_size\": %d, \"pid\": %d, \"level\": \"%s\",\n",
            rr.rank, rr.world, (int)getpid(), vmb_profile_level_name(profile_level));
    fprintf(out, " \"blas\": ");
    vmb_blas_write_json(out);
    fprintf(out, ",\n \"fftw\": ");
    vmb_fftw_write_json(out);
    fprintf(out, "}\n");
}

static void write_report(FILE *out, void *arg)
{
    const struct report_args *ra = arg;
    struct vmb_snapshot s;
    vmb_snapshot_get(&s);
    static const char *const counter_names[VMB_STAT_REASON_BASE] = {
        "calls", "gpu_submitted", "gpu_completed", "consumed", "cpu_calls", "checked",
    };
    char host[256] = "", exe_raw[PATH_MAX], exe[PATH_MAX];
    uint32_t size = sizeof exe_raw;
    gethostname(host, sizeof host - 1);
    const char *exe_path = NULL;
    struct stat st;
    int have_stat = 0;
    if (_NSGetExecutablePath(exe_raw, &size) == 0 && realpath(exe_raw, exe)) {
        exe_path = exe;
        have_stat = stat(exe, &st) == 0;
    }
    const char *device = vmb_gpu_device_name();

    fprintf(out, "{\"schema\": \"vmb.rank-report.v1\", \"complete\": true,\n");
    fprintf(out, " \"backend\": {\"name\": \"vasp-metal-backend\", \"version\": \"%s\", "
                 "\"source_commit\": \"%s\", \"source_digest\": \"%s\", \"source_dirty\": %s},\n",
            VMB_VERSION, VMB_SOURCE_COMMIT, VMB_SOURCE_DIGEST, VMB_SOURCE_DIRTY ? "true" : "false");
    fprintf(out, " \"attempt_id\": ");
    vmb_json_string(out, rr.attempt);
    fprintf(out, ", \"rank\": %d, \"world_size\": %d, ", rr.rank, rr.world);
    if (rr.local_rank >= 0)
        fprintf(out, "\"local_rank\": %d, \"local_size\": %d, ", rr.local_rank, rr.local_size);
    else
        fprintf(out, "\"local_rank\": null, \"local_size\": null, ");
    fprintf(out, "\"rank_source\": \"%s\",\n",
            getenv("OMPI_COMM_WORLD_RANK") ? "openmpi" : "singleton");
    fprintf(out, " \"process\": {\"pid\": %d, \"ppid\": %d, \"hostname\": ", (int)getpid(),
            (int)getppid());
    vmb_json_string(out, host);
    fprintf(out, ", \"executable\": ");
    vmb_json_string(out, exe_path);
    if (have_stat)
        fprintf(out, ", \"executable_stat\": [%llu, %llu, %lld, %lld]},\n",
                (unsigned long long)st.st_dev, (unsigned long long)st.st_ino,
                (long long)st.st_size,
                (long long)st.st_mtimespec.tv_sec * 1000000000LL + st.st_mtimespec.tv_nsec);
    else
        fprintf(out, ", \"executable_stat\": null},\n");
    fprintf(out, " \"times\": {\"start_unix\": %.6f, \"end_unix\": %.6f},\n", rr.start, wall_now());
    fprintf(out, " \"mode\": \"%s\", \"check\": %s,\n", vmb_mode_name(s.mode),
            s.check ? "true" : "false");
    fprintf(out, " \"device\": {\"initialized\": %s, \"name\": ", device ? "true" : "false");
    vmb_json_string(out, device);
    fprintf(out, "},\n \"cpu_library\": ");
    vmb_json_string(out, s.cpu_library && *s.cpu_library ? s.cpu_library : NULL);
    fprintf(out, ",\n \"counters\": {");
    for (int i = 0; i < VMB_STAT_REASON_BASE; i++)
        fprintf(out, "%s\"%s\": %llu", i ? ", " : "", counter_names[i],
                (unsigned long long)s.stats[i]);
    fprintf(out, "},\n \"statuses\": {");
    for (int i = 0; i < VMB_NSTATUS; i++)
        fprintf(out, "%s\"%s\": %llu", i ? ", " : "", vmb_status_name(i),
                (unsigned long long)s.stats[VMB_STAT_REASON_BASE + i]);
    fprintf(out, "},\n");
    if (s.stats[VMB_STAT_CHECKED] == 0)
        fprintf(out, " \"max_check_ratio\": null, \"max_check_ratio_state\": \"none\",\n");
    else if (isfinite(s.max_check_ratio))
        fprintf(out, " \"max_check_ratio\": %.17g, \"max_check_ratio_state\": \"finite\",\n",
                s.max_check_ratio);
    else
        fprintf(out, " \"max_check_ratio\": null, \"max_check_ratio_state\": \"nonfinite\",\n");
    /* Optional timing fields (v0.4): additive, absent from v0.3 reports. */
    fprintf(out, " \"timing\": {\"try_s\": %.9f, \"cpu_s\": %.9f},\n \"work_decades\": [",
            s.try_s, s.cpu_s);
    for (int i = 0; i < 16; i++)
        fprintf(out, "%s%llu", i ? ", " : "", (unsigned long long)s.work_decades[i]);
    fprintf(out, "],\n");
    /* Optional profile declaration (v0.5): additive, absent from earlier reports. */
    if (ra->profile_file)
        fprintf(out, " \"profile\": {\"file\": \"%s\", \"level\": \"%s\"},\n",
                ra->profile_file, vmb_profile_level_name(profile_level));
    else if (profile_level != VMB_PROFILE_OFF)
        fprintf(out, " \"profile\": {\"file\": null, \"level\": \"%s\"},\n",
                vmb_profile_level_name(profile_level));
    /* FFTW plan cache declaration (v0.5): additive; every v0.5 report has it. */
    fprintf(out, " \"fftw_plan_cache\": ");
    vmb_fftw_cache_write_json(out);
    fprintf(out, ",\n");
    /* DGEMV/DGEMM library declaration (v0.5): additive; every v0.5 report has it. */
    fprintf(out, " \"dblas_library\": ");
    vmb_dblas_write_json(out);
    fprintf(out, ",\n");
    if (ra->telemetry_file)
        fprintf(out, " \"telemetry\": {\"file\": \"%s\", \"dropped_shapes\": %llu}}\n",
                ra->telemetry_file, (unsigned long long)ra->dropped);
    else
        fprintf(out, " \"telemetry\": null}\n");
}

static void report_at_exit(void)
{
    if (!rr.active || getpid() != rr.owner)
        return;  /* forked children never publish for the claiming process */
    char name[64], telemetry[64], profile[64];
    struct report_args ra = {NULL, NULL, 0};
    if (profile_level != VMB_PROFILE_OFF) {
        snprintf(profile, sizeof profile, "rank-%05d.profile.json", rr.rank);
        if (publish(profile, write_profile, NULL) == 0)
            ra.profile_file = profile;
        else
            fprintf(stderr, "vmb: rank %d profile not published (%s)\n", rr.rank, strerror(errno));
    }
    if (rr.telemetry) {
        snprintf(telemetry, sizeof telemetry, "rank-%05d.telemetry.tsv", rr.rank);
        if (publish(telemetry, write_telemetry, &ra.dropped) == 0)
            ra.telemetry_file = telemetry;
        else
            fprintf(stderr, "vmb: rank %d telemetry not published (%s)\n", rr.rank, strerror(errno));
    }
    snprintf(name, sizeof name, "rank-%05d.report.json", rr.rank);
    if (publish(name, write_report, &ra) != 0)
        fprintf(stderr, "vmb: rank %d report not published (%s)\n", rr.rank, strerror(errno));
}

/* The directory must be canonical (no symbolic link anywhere), a directory
   owned by this user and writable by nobody else. The descriptor is kept for
   the lifetime of the process, so later path changes cannot redirect writes. */
static void open_dir(const char *dir)
{
    char real[PATH_MAX];
    if (dir[0] != '/' || strlen(dir) >= sizeof rr.dir || !realpath(dir, real) || strcmp(real, dir))
        fail("VMB_REPORT_DIR must be an existing canonical absolute directory");
    int fd = open(dir, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    struct stat st;
    if (fd < 0 || fstat(fd, &st) != 0 || !S_ISDIR(st.st_mode))
        fail("cannot open VMB_REPORT_DIR %s", dir);
    if (st.st_uid != geteuid() || (st.st_mode & (S_IWGRP | S_IWOTH)))
        fail("VMB_REPORT_DIR must be owned by this user and not group/world writable");
    snprintf(rr.dir, sizeof rr.dir, "%s", dir);
    rr.dir_fd = fd;
}

__attribute__((constructor)) static void rank_report_init(void)
{
    const char *dir = getenv("VMB_REPORT_DIR"), *telemetry = getenv("VMB_TELEMETRY");
    const char *report = getenv("VMB_REPORT"), *attempt = getenv("VMB_ATTEMPT_ID");
    if (!dir && !telemetry && !report && !attempt && !getenv("VMB_EXPECT_WORLD_SIZE") &&
        !getenv("VMB_PROFILE"))
        return;  /* no reporting requested: the launcher environment is not examined */
    int rank, world;
    read_rank(&rank, &world);
    if (!dir) {
        if (getenv("VMB_PROFILE"))
            fail("VMB_PROFILE requires VMB_REPORT_DIR and VMB_TELEMETRY=rank");
        if (attempt || getenv("VMB_EXPECT_WORLD_SIZE"))
            fail("VMB_ATTEMPT_ID/VMB_EXPECT_WORLD_SIZE require VMB_REPORT_DIR");
        if (telemetry && strcmp(telemetry, "rank") == 0)
            fail("VMB_TELEMETRY=rank requires VMB_REPORT_DIR");
        if (world > 1 && ((report && *report) || (telemetry && *telemetry)))
            fail("literal VMB_REPORT/VMB_TELEMETRY paths are shared by all %d ranks; "
                 "use VMB_REPORT_DIR", world);
        return;
    }
    if (report)
        fail("VMB_REPORT must be unset when VMB_REPORT_DIR is set");
    if (telemetry && strcmp(telemetry, "rank") != 0)
        fail("VMB_TELEMETRY must be unset or 'rank' when VMB_REPORT_DIR is set");
    if (!attempt || !valid_attempt(attempt))
        fail("VMB_ATTEMPT_ID is missing or not 1-64 characters of [A-Za-z0-9._-]");
    long expect = env_index("VMB_EXPECT_WORLD_SIZE");
    if (expect == -2 || (expect >= 0 && expect != world))
        fail("world size %d contradicts VMB_EXPECT_WORLD_SIZE", world);
    open_dir(dir);
    snprintf(rr.attempt, sizeof rr.attempt, "%s", attempt);
    rr.rank = rank;
    rr.world = world;
    rr.telemetry = telemetry != NULL;
    const char *level = getenv("VMB_PROFILE");
    if (level && !rr.telemetry)
        fail("VMB_PROFILE requires VMB_TELEMETRY=rank");
    if (!level)
        profile_level = rr.telemetry ? VMB_PROFILE_DEFAULT : VMB_PROFILE_OFF;
    else if (strcmp(level, "0") == 0)
        profile_level = VMB_PROFILE_OFF;
    else if (strcmp(level, "1") == 0)
        profile_level = VMB_PROFILE_ON;
    else if (strcmp(level, "reuse") == 0)
        profile_level = VMB_PROFILE_REUSE;
    else
        fail("VMB_PROFILE must be 0, 1 or reuse");
    rr.owner = getpid();
    rr.start = wall_now();
    char claim[64];
    snprintf(claim, sizeof claim, "rank-%05d.claim.json", rank);
    if (publish(claim, write_claim, NULL) != 0)
        fail("rank %d already claimed or claim not writable in %s (%s)", rank, dir, strerror(errno));
    /* Mode and check come from the environment even for ranks that never call
       ZGEMM; reading it does not resolve OpenBLAS or initialise Metal. */
    vmb_config_reload();
    rr.active = 1;
    if (atexit(report_at_exit) != 0)
        fail("cannot register the rank report");
}
