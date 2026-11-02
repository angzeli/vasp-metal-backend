#ifndef ORACLE_H
#define ORACLE_H

/* C = alpha op(A) op(B) + beta C in plain FP64 loops (C not read if beta == 0).
   If scale is non-NULL it receives S_ij = |alpha| sum |op(A)||op(B)| + |beta||C_ij|
   (dense m x n) computed from the input C. */
void oracle_zgemm(char ta, char tb, int m, int n, int k, const double alpha[2],
                  const double *a, int lda, const double *b, int ldb, const double beta[2],
                  double *c, int ldc, double *scale);

#endif
