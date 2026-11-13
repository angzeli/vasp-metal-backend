/* Metal device, runtime kernel compilation, packing and dispatch. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <string.h>

#include "vmb_internal.h"
#include "zgemm_df64_metal.h"  /* generated from src/zgemm_df64.metal */

/* Must match struct Params in zgemm_df64.metal. */
struct params {
    uint32_t m, n, k, ta, tb, rows_a, rows_b, beta_zero;
    float alpha[4] __attribute__((aligned(16)));  /* re.hi, re.lo, im.hi, im.lo */
    float beta[4];
};
_Static_assert(sizeof(struct params) == 64, "layout must match the Metal struct");

static pthread_mutex_t init_lock = PTHREAD_MUTEX_INITIALIZER;
static int state;  /* 0 untried, 1 ready, -1 unavailable */
static id<MTLDevice> device;
static id<MTLComputePipelineState> pipeline;
static id<MTLCommandQueue> queue;
static id<MTLBuffer> buffers[4];  /* A, B, C, Out; grown on demand */
static char device_name[256];
static char init_error[512];

static void fail(const char *what, NSError *error)
{
    snprintf(init_error, sizeof init_error, "%s%s%s", what, error ? ": " : "",
             error ? error.localizedDescription.UTF8String : "");
    state = -1;
}

static void initialize(void)
{
    device = MTLCreateSystemDefaultDevice();
    if (!device) {
        fail("MTLCreateSystemDefaultDevice returned nil", nil);
        return;
    }
    snprintf(device_name, sizeof device_name, "%s", device.name.UTF8String);

    /* The error-free transformations need IEEE float semantics. */
    MTLCompileOptions *options = [MTLCompileOptions new];
    options.mathMode = MTLMathModeSafe;
    options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
    NSString *source = [[NSString alloc] initWithBytes:vmb_kernel_source
                                                length:vmb_kernel_source_len
                                              encoding:NSUTF8StringEncoding];
    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
    if (!library) {
        fail("kernel compilation failed", error);
        return;
    }
    id<MTLFunction> function = [library newFunctionWithName:@"zgemm_df64"];
    if (!function) {
        fail("kernel function zgemm_df64 missing", nil);
        return;
    }
    pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) {
        fail("pipeline creation failed", error);
        return;
    }
    queue = [device newCommandQueue];
    if (!queue) {
        fail("command queue creation failed", nil);
        return;
    }
    state = 1;
}

int vmb_gpu_available(int simulate_no_device)
{
    if (simulate_no_device)
        return 0;
    pthread_mutex_lock(&init_lock);
    if (state == 0) {
        @autoreleasepool {
            initialize();
        }
    }
    int ready = state == 1;
    pthread_mutex_unlock(&init_lock);
    return ready;
}

const char *vmb_gpu_device_name(void) { return state == 1 ? device_name : NULL; }

const char *vmb_gpu_init_error(void) { return init_error; }

static id<MTLBuffer> ensure(int slot, size_t bytes)
{
    if (!buffers[slot] || buffers[slot].length < bytes)
        buffers[slot] = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    return buffers[slot];
}

/* x -> hi + lo with hi = fl32(x); exact to about 2^-48 relative for |x| <= 1. */
static inline void split(double x, float *d)
{
    float hi = (float)x;
    d[0] = hi;
    d[1] = (float)(x - (double)hi);
}

/* Multiplying by a normal power of two rounds exactly like ldexp, so the
   fast path is bit-identical; other exponents keep the per-element ldexp. */
static int exact_factor(int e) { return e >= -1000 && e <= 1000; }

/* Dense float4 copy of the stored rows x cols block, scaled by 2^-e. */
static void pack(float *dst, const double *src, int rows, int cols, int ld, int e)
{
    const double f = exact_factor(-e) ? ldexp(1.0, -e) : 0.0;
    for (int j = 0; j < cols; j++) {
        const double *z = src + 2 * (size_t)j * ld;
        float *d = dst + 4 * (size_t)j * rows;
        if (f != 0.0) {
            for (int i = 0; i < rows; i++) {
                split(z[2 * i] * f, d + 4 * i);
                split(z[2 * i + 1] * f, d + 4 * i + 2);
            }
        } else {
            for (int i = 0; i < rows; i++) {
                split(ldexp(z[2 * i], -e), d + 4 * i);
                split(ldexp(z[2 * i + 1], -e), d + 4 * i + 2);
            }
        }
    }
}

int vmb_gpu_zgemm(struct vmb_gpu_job *job, int *submitted)
{
    *submitted = 0;
    job->t_pack = job->t_encode = job->t_gpu = job->t_wait = job->t_unpack = 0.0;
    job->t_queue = job->t_wake = 0.0;
    double t0 = vmb_now();
    size_t na = (size_t)job->rows_a * job->cols_a, nb = (size_t)job->rows_b * job->cols_b;
    size_t mn = (size_t)job->m * job->n;
    @autoreleasepool {
        id<MTLBuffer> a = ensure(0, 16 * na), b = ensure(1, 16 * nb);
        id<MTLBuffer> c = ensure(2, job->beta_zero ? 16 : 16 * mn), out = ensure(3, 16 * mn);
        if (!a || !b || !c || !out)
            return -1;
        pack(a.contents, job->a, job->rows_a, job->cols_a, job->lda, job->sa);
        pack(b.contents, job->b, job->rows_b, job->cols_b, job->ldb, job->sb);
        if (!job->beta_zero)
            pack(c.contents, job->c, job->m, job->n, job->ldc, job->sc);
        double t1 = vmb_now();
        job->t_pack = t1 - t0;

        struct params p = {
            .m = (uint32_t)job->m, .n = (uint32_t)job->n, .k = (uint32_t)job->k,
            .ta = (uint32_t)job->ta, .tb = (uint32_t)job->tb,
            .rows_a = (uint32_t)job->rows_a, .rows_b = (uint32_t)job->rows_b,
            .beta_zero = (uint32_t)job->beta_zero,
        };
        split(job->alpha_s[0], p.alpha);
        split(job->alpha_s[1], p.alpha + 2);
        split(job->beta_s[0], p.beta);
        split(job->beta_s[1], p.beta + 2);

        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (!cb || !enc)
            return -1;
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:a offset:0 atIndex:0];
        [enc setBuffer:b offset:0 atIndex:1];
        [enc setBuffer:c offset:0 atIndex:2];
        [enc setBuffer:out offset:0 atIndex:3];
        [enc setBytes:&p length:sizeof p atIndex:4];
        NSUInteger w = pipeline.threadExecutionWidth;
        NSUInteger h = MIN(pipeline.maxTotalThreadsPerThreadgroup / w, (NSUInteger)8);
        [enc dispatchThreads:MTLSizeMake((NSUInteger)job->m, (NSUInteger)job->n, 1)
            threadsPerThreadgroup:MTLSizeMake(w, h, 1)];
        [enc endEncoding];
        [cb commit];
        *submitted = 1;
        double t2 = vmb_now();
        job->t_encode = t2 - t1;
        [cb waitUntilCompleted];
        double t3 = vmb_now();
        job->t_wait = t3 - t2;
        if (cb.status != MTLCommandBufferStatusCompleted || cb.error)
            return -1;
        if (cb.GPUEndTime > cb.GPUStartTime) {
            job->t_gpu = cb.GPUEndTime - cb.GPUStartTime;
            /* Metal's GPU timestamps share the host's mach time base with vmb_now. */
            job->t_queue = cb.GPUStartTime - t2;
            job->t_wake = t3 - cb.GPUEndTime;
        }

        const float *o = out.contents;
        if (exact_factor(job->so)) {
            const double f = ldexp(1.0, job->so);
            for (size_t e = 0; e < 2 * mn; e++)
                job->out[e] = ((double)o[2 * e] + (double)o[2 * e + 1]) * f;
        } else {
            for (size_t e = 0; e < 2 * mn; e++)
                job->out[e] = ldexp((double)o[2 * e] + (double)o[2 * e + 1], job->so);
        }
        job->t_unpack = vmb_now() - t3;
    }
    return 0;
}
