// Complex GEMM in float-float ("df64") arithmetic: one thread per element of
//   Out = alpha * op(A) * op(B) + beta * C
// on dense, power-of-two pre-scaled column-major inputs. A complex element is
// stored as float4 (re.hi, re.lo, im.hi, im.lo).
//
// The error-free transformations below require IEEE round-to-nearest float
// add/mul and a fused fma; the host compiles this source with MTLMathModeSafe.
#include <metal_stdlib>
using namespace metal;

struct df { float hi; float lo; };

static inline df two_sum(float a, float b)
{
    float s = a + b;
    float v = s - a;
    float e = (a - (s - v)) + (b - v);
    return {s, e};
}

static inline df fast_two_sum(float a, float b)  // requires |a| >= |b|
{
    float s = a + b;
    float e = b - (s - a);
    return {s, e};
}

static inline df two_prod(float a, float b)
{
    float p = a * b;
    float e = fma(a, b, -p);
    return {p, e};
}

static inline df df_add(df x, df y)
{
    df s = two_sum(x.hi, y.hi);
    df t = two_sum(x.lo, y.lo);
    s.lo += t.hi;
    s = fast_two_sum(s.hi, s.lo);
    s.lo += t.lo;
    return fast_two_sum(s.hi, s.lo);
}

static inline df df_neg(df x) { return {-x.hi, -x.lo}; }

static inline df df_mul(df x, df y)
{
    df p = two_prod(x.hi, y.hi);
    p.lo = fma(x.hi, y.lo, p.lo);
    p.lo = fma(x.lo, y.hi, p.lo);
    return fast_two_sum(p.hi, p.lo);
}

struct Params {
    uint m, n, k, ta, tb, rows_a, rows_b, beta_zero;  // ta/tb: 0 = N, 1 = T, 2 = C
    float4 alpha;
    float4 beta;
};

// Element (row, col) of op(M) for M stored dense with `rows` rows.
static inline float4 load_op(device const float4 *M, uint row, uint col, uint rows, uint trans)
{
    float4 v = trans == 0 ? M[row + col * rows] : M[col + row * rows];
    if (trans == 2) {
        v.z = -v.z;
        v.w = -v.w;
    }
    return v;
}

kernel void zgemm_df64(device const float4 *A [[buffer(0)]],
                       device const float4 *B [[buffer(1)]],
                       device const float4 *C [[buffer(2)]],
                       device float4 *Out [[buffer(3)]],
                       constant Params &p [[buffer(4)]],
                       uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= p.m || gid.y >= p.n)
        return;
    uint i = gid.x, j = gid.y;

    df re = {0.0f, 0.0f}, im = {0.0f, 0.0f};
    for (uint l = 0; l < p.k; ++l) {
        float4 a = load_op(A, i, l, p.rows_a, p.ta);
        float4 b = load_op(B, l, j, p.rows_b, p.tb);
        df ar = {a.x, a.y}, ai = {a.z, a.w}, br = {b.x, b.y}, bi = {b.z, b.w};
        re = df_add(re, df_mul(ar, br));
        re = df_add(re, df_neg(df_mul(ai, bi)));
        im = df_add(im, df_mul(ar, bi));
        im = df_add(im, df_mul(ai, br));
    }

    df alr = {p.alpha.x, p.alpha.y}, ali = {p.alpha.z, p.alpha.w};
    df out_re = df_add(df_mul(alr, re), df_neg(df_mul(ali, im)));
    df out_im = df_add(df_mul(alr, im), df_mul(ali, re));
    if (p.beta_zero == 0) {  // BLAS: C is not read when beta == 0
        float4 c = C[i + j * p.m];
        df cr = {c.x, c.y}, ci = {c.z, c.w};
        df btr = {p.beta.x, p.beta.y}, bti = {p.beta.z, p.beta.w};
        out_re = df_add(out_re, df_add(df_mul(btr, cr), df_neg(df_mul(bti, ci))));
        out_im = df_add(out_im, df_add(df_mul(btr, ci), df_mul(bti, cr)));
    }
    Out[i + j * p.m] = float4(out_re.hi, out_re.lo, out_im.hi, out_im.lo);
}
