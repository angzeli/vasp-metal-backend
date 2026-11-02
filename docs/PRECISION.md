# Precision policy (v0.2)

Apple GPUs have no FP64 arithmetic. The ZGEMM kernel therefore computes in
float-float ("df64") arithmetic: each real number is an unevaluated sum
`hi + lo` of two IEEE binary32 values, combined with error-free transformations
(TwoSum, FMA-based TwoProd). The kernel source is compiled at runtime with
`MTLMathModeSafe` and precise math functions so that float add/mul keep IEEE
round-to-nearest semantics; fast-math reassociation would break the
transformations. No lower-precision shortcut is used anywhere.

## Representation and scaling

- Each FP64 input is split as `hi = fl32(x)`, `lo = fl32(x - hi)`, about 48
  significant bits (relative representation error about 2^-48).
- A, B and C are each scaled by one exact power of two so that their largest
  component lies in [0.5, 1); alpha and beta absorb the factors and the output
  scale `2^so` keeps every partial sum bounded. The host rescale is exact.

## Why scaling alone is not enough (fixed in v0.2)

Binary32 covers roughly 2^±126. Matrix-wide scaling cannot represent an element
far below its matrix maximum: in v0.1 (`d3be811`), `A = diag(2^90, 2^-90)`,
`B = diag(2^-90, 2^90)` mapped the small entries to 2^-181 before splitting,
both binary32 parts flushed to zero, and a forced GPU call returned the zero
matrix instead of I. The v0.1 absolute floor assumed lost terms were small
relative to `max S`; that is false when the scales of A and B compensate.
The v0.1 Si2 integration run did not contain such ranges (802/802 shadow
checks passed) but the guarantee did not hold in general.

## Unscaled overflow hole (found in PR #1 review, fixed before merge)

The first v0.2 envelope (`0344fdb`) bounded only the *scaled* output exponent
`so`, and formed its lower bound `L` (below) as a plain FP64 product. Neither
covers the unscaled values the CPU path forms. With a small alpha, `so` is
small even when the unscaled products or their sums overflow, and an
overflowing `L` became `+inf`, which made the absolute tolerance infinite and
the underflow budget always pass. Two regressions (N,N, m = n = 1, k = 2,
alpha = 2^-500, beta = 0, real entries, forced, shadow check off) showed it:

| Case | Unscaled values | Exact result | `0344fdb` forced GPU | OpenBLAS |
| --- | --- | --- | --- | --- |
| A = [2^600, 2^440], B = [2^440, 2^600]^T | each product 2^1040 | 2^541 | 0, consumed as success | (inf, NaN) |
| A = [2^511, 2^511], B = [2^512, 2^512]^T | products 2^1023 (finite), sum 2^1024 | 2^524 | 2^524, consumed | (inf, NaN) |

The second case returns the mathematically right value but still breaks the
contract, which is to reproduce the installed CPU implementation whenever the
FP64 evaluation is not safely finite. A `!isfinite(prod)` test alone would not
catch it, because every individual product is finite.

## Exponent-range safety envelope

Before any GPU work, the host computes, in the same pass that checks
finiteness, the per-index maxima `cA_l = max_i |op(A)_il|` and
`rB_l = max_j |op(B)_lj|` (component maxima).

**Overflow bound on unscaled CPU intermediates.** Let `e(x) = ilogb(x) + 1`
(so `x < 2^e(x)`), formed for each maximum without multiplying anything. With
`2k <= 2^kbits`, `ea`/`eb`/`ec` the exponents of the A, B, C maxima, and
`ealpha`/`ebeta` those of `max(|Re|, |Im|)` of alpha and beta:

    ep = max_l ( e(cA_l) + e(rB_l) )          bounds every real product x*y
    es = ep + kbits + 1                        bounds every partial sum, in any
                                               order, of the 2k real products
                                               forming a real or imaginary part
                                               (< 2k 2^ep; +1 margin)
    eα = ealpha + max(es, ea, eb) + 1          alpha times that sum, or alpha
                                               applied first to an operand
                                               (reference BLAS forms alpha*B)
    E  = max(es, eα)                           if beta*C is absent
    E  = max(es, max(eα, ebeta + ec + 1) + 1)  otherwise (beta*C and the final sum)

Every unscaled value of these forms has magnitude below `2^E`. This is pure
integer arithmetic, so it cannot overflow. The sum term `es` stands on its own,
so a tiny alpha cannot hide an overflowing sum. The bound assumes the CPU path
forms only these intermediates (component products, sums of them in any order
or grouping, alpha applied before or after, beta·C and the final add). That
holds for conventional ZGEMM, but not for algorithms such as 3M/Gauss
multiplication, whose sums of components can be about twice as large; the
`+1` margin and the 2^24 headroom below the FP64 limit absorb that.

Only after `E <= 1000` holds is the lower bound formed:

    L = max( max(|Re alpha|, |Im alpha|) * max_l cA_l rB_l ,
             max(|Re beta|,  |Im beta|)  * max |C|          )   (C term only if beta != 0)

Each product is then below 2^1000, so `L` is finite. If `E > 1000`, `L` is set
to `+inf`, which is rejected explicitly and never treated as a permissive
tolerance.

    L = max( max(|Re alpha|, |Im alpha|) * max_l cA_l rB_l ,
             max(|Re beta|,  |Im beta|)  * max |C|          )   (C term only if beta != 0)

is a lower bound of `max_ij S_ij` (a single product term, and component maxima
never exceed moduli). In scaled units every packed value and every partial sum
is bounded by 2k, and every binary32 operation that underflows, whether it
produces a subnormal or is flushed to zero, adds an absolute error of at most
2^-126. Each k-term performs about 110 binary32 operations and several input
components may be flushed, so 2^-110 per term (and two more terms for the
alpha/beta epilogue) is a generous bound. The GPU path is used only if

    E  <= 1000                                     (unscaled CPU intermediates)
    so <= 1000                                     (scaled output and GPU host values)
    L  finite and L >= 2^-900                      (no FP64 subnormal results)
    (k + 2) * 2^(so - 110)  <=  0.5 * ATOL(L)      (underflow budget)

Otherwise the call is a counted `range` fallback to the original OpenBLAS
`zgemm_`, made before any GPU submission or write to C. With
`ATOL(L) <= ATOL(max S)`, the underflow error is at most half of every
element's tolerance, and the ordinary rounding error is relative to `S_ij` as
before.

**What the overflow bounds guarantee.** If the GPU path is taken, every
unscaled value of the forms listed above is below 2^1000, and so is the
scaled output scale `2^so`. An FP64 evaluation of the call therefore cannot
overflow, and no `inf - inf` can arise, so a finite GPU result does not hide
a non-finite CPU result. Any call that might overflow stays on the CPU, which
reproduces the installed implementation's non-finite results bit for bit.
The bounds are conservative: they may decline calls whose actual evaluation is
safe, never the reverse. They do not claim that OpenBLAS's own result is
accurate in any other sense.

The envelope is a property of the operands, not of the shape, and applies in
both `auto` and `force` modes; neither can bypass it. An all-zero product
(`L = 0`) also falls back.

## Tolerance (unchanged from v0.1, fixed before testing)

For each element of `C_new = alpha op(A) op(B) + beta C`, with complex moduli,

    S_ij  = |alpha| * sum_l |op(A)_il| |op(B)_lj| + |beta| |C_ij|
    |C_gpu - C_exact|_ij <= ATOL + RTOL * S_ij
    RTOL  = 8 (k + 2) 2^-48        (about 2.8e-14 (k + 2))
    ATOL  = 1e-18 * max_ij S_ij

Both are defined in `include/vmb.h` (`VMB_RTOL`, `VMB_ATOL`).

## Fail-closed comparison

`vmb_tolerance_ratio` implements the comparison used by the shadow check
(`VMB_CHECK=1`) and by the tests. It returns the worst `err / tol` and fails
closed:

- an exact match passes, even when the tolerance is zero (`S = 0`);
- a positive error against a zero tolerance fails;
- a non-finite or negative scale, a non-finite result or reference, or an
  overflowing difference fails;
- a failure is returned immediately, so a later passing element can never
  overwrite it. (v0.1 tracked the maximum with `!(r <= max)`, which let a later
  finite ratio replace an earlier NaN.)

## Calls the GPU path declines

These go to the original OpenBLAS `zgemm_` with C untouched by the GPU:

- mode `off`, or mode `auto` outside a measured winning shape region;
- NaN or Inf in alpha, beta, the referenced part of A or B, or of C when
  beta != 0;
- quick-return cases (m, n or k = 0, alpha = 0) and invalid arguments;
- problems below `VMB_MIN_WORK` or above `VMB_MAX_WORK` / `VMB_MAX_BYTES`;
- operands outside the exponent-range envelope;
- no Metal device or kernel build failure.

A GPU result that fails to complete, contains a non-finite element, or (in
shadow-check mode) fails the comparison is discarded before any write to C,
and the call is recomputed on the CPU.

## Tested evidence

`build/test_zgemm` covers the d3be811 regression (2×2 and embedded in an
ordinary 64³ call), mixed scales of 2^±10, 2^±40 and 2^±200, products near FP64
underflow and overflow, `inf - inf`, beta·C imbalance of 2^±300, an all-zero
operand, the comparison rules above, and a perturbed GPU result rejected by
the in-process shadow check. `build/test_fortran_abi` repeats the regression
through the Fortran ABI against a quad-precision oracle.

The two unscaled-overflow regressions above (`overflow:` cases, forced, check
off) fail on `0344fdb` and pass with the fix:
- the direct GPU attempt returns `range` (status 11) and leaves a sentinel C
  unchanged;
- through `zgemm_`, the counter deltas are `range` +1, GPU submitted 0,
  consumed 0, CPU fallback +1;
- the result is bit-identical to a direct call to the loaded OpenBLAS, which
  returns `(inf, NaN)` for both.

Two positive controls with the same shape stay on the GPU (submitted +1,
consumed +1, CPU 0, bit-identical to OpenBLAS):
- alpha = 2^-500 with moderate entries, which shows that a tiny alpha by itself
  does not trigger the fallback;
- an ordinary alpha = 1 call.

Every earlier case keeps its routing and its err/tol.
