# Precision policy

Apple GPUs have no FP64 arithmetic. The ZGEMM kernel therefore computes in
float-float ("df64") arithmetic: each real number is an unevaluated sum
`hi + lo` of two IEEE binary32 values, combined with error-free transformations
(TwoSum, FMA-based TwoProd). The kernel source is compiled at runtime with
`MTLMathModeSafe` and precise math functions so that float add/mul keep IEEE
round-to-nearest semantics; fast-math reassociation would break the
transformations.

## Representation and range

- Each FP64 input is split as `hi = fl32(x)`, `lo = fl32(x - hi)`, about 48
  significant bits (relative representation error about 2^-48).
- Binary32 has an exponent range of roughly 2^±126, far smaller than FP64. Before
  packing, A, B and C are each scaled by an exact power of two so that their
  largest component lies in [0.5, 1). alpha and beta absorb the scale factors and
  the output scale `2^so` is chosen so the result is O(1) on the GPU. The final
  rescale on the host is exact (`ldexp`).
- Consequence: elements more than about 2^-126 below the largest element of
  their matrix are flushed; their contribution is below the absolute tolerance.

## Tolerance (fixed before testing)

For each element of `C_new = alpha op(A) op(B) + beta C`, with complex moduli,

    S_ij  = |alpha| * sum_l |op(A)_il| |op(B)_lj| + |beta| |C_ij|
    |C_gpu - C_exact|_ij <= ATOL + RTOL * S_ij
    RTOL  = 8 (k + 2) 2^-48        (about 2.8e-14 (k + 2))
    ATOL  = 1e-18 * max_ij S_ij

The relative part is a gamma_k-style bound in units of the df64 rounding error;
the absolute part covers flushed elements. Both are defined in `include/vmb.h`
(`VMB_RTOL`, `VMB_ATOL`) and used unchanged by the tests and by the optional
in-process shadow check (`VMB_CHECK=1`).

## Inputs the GPU path declines

These calls go to the original OpenBLAS `zgemm_` with C untouched by the GPU:

- NaN or Inf in alpha, beta, the referenced part of A or B, or of C when
  beta != 0 (CPU semantics for non-finite propagation are preserved);
- quick-return cases (m, n or k = 0, alpha = 0) and invalid arguments (so the
  original BLAS error reporting applies);
- problems below `VMB_MIN_WORK` or above `VMB_MAX_WORK` / `VMB_MAX_BYTES`;
- no Metal device or kernel build failure.

A GPU result that fails to complete, contains a non-finite element, or (in
shadow-check mode) exceeds the tolerance is discarded before any write to C,
and the call is recomputed on the CPU.
