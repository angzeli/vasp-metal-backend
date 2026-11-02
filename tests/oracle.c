/* Independent FP64 reference ZGEMM: plain loops, no BLAS, no library code. */
#include "oracle.h"

#include <math.h>

static void elem(const double *x, int ld, char trans, int row, int col, double *re, double *im)
{
    int t = trans != 'N' && trans != 'n';
    int conj = trans == 'C' || trans == 'c';
    const double *z = t ? x + 2 * ((long)col + (long)row * ld) : x + 2 * ((long)row + (long)col * ld);
    *re = z[0];
    *im = conj ? -z[1] : z[1];
}

void oracle_zgemm(char ta, char tb, int m, int n, int k, const double alpha[2],
                  const double *a, int lda, const double *b, int ldb, const double beta[2],
                  double *c, int ldc, double *scale)
{
    int beta_zero = beta[0] == 0.0 && beta[1] == 0.0;
    double aa = hypot(alpha[0], alpha[1]), ab = hypot(beta[0], beta[1]);
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++) {
            double sr = 0.0, si = 0.0, s = 0.0;
            for (int l = 0; l < k; l++) {
                double xr, xi, yr, yi;
                elem(a, lda, ta, i, l, &xr, &xi);
                elem(b, ldb, tb, l, j, &yr, &yi);
                sr += xr * yr - xi * yi;
                si += xr * yi + xi * yr;
                s += hypot(xr, xi) * hypot(yr, yi);
            }
            double *z = c + 2 * ((long)i + (long)j * ldc);
            double rr = alpha[0] * sr - alpha[1] * si, ri = alpha[0] * si + alpha[1] * sr;
            s *= aa;
            if (!beta_zero) {
                s += ab * hypot(z[0], z[1]);
                rr += beta[0] * z[0] - beta[1] * z[1];
                ri += beta[0] * z[1] + beta[1] * z[0];
            }
            z[0] = rr;
            z[1] = ri;
            if (scale)
                scale[i + (long)j * m] = s;
        }
}
