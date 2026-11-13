/* gfortran-convention complex dot products for a VASP linked directly against
 * Accelerate (#9). Accelerate's legacy Fortran BLAS returns complex function
 * results through a hidden first argument (the f2c convention); gfortran expects
 * them by value, so VASP's zdotc/cdotc calls crash on it
 * (tests/test_accelerate_abi.f90). Linked ahead of Accelerate, these two bind
 * VASP's references and compute the same dots through Accelerate's CBLAS. Every
 * other BLAS/LAPACK function VASP imports passes the ABI test unwrapped. */
#include <complex.h>

void cblas_zdotc_sub(int n, const void *x, int incx, const void *y, int incy, void *dotc);
void cblas_cdotc_sub(int n, const void *x, int incx, const void *y, int incy, void *dotc);

double complex zdotc_(const int *n, const double complex *x, const int *incx,
                      const double complex *y, const int *incy)
{
    double complex r;
    cblas_zdotc_sub(*n, x, *incx, y, *incy, &r);
    return r;
}

float complex cdotc_(const int *n, const float complex *x, const int *incx,
                     const float complex *y, const int *incy)
{
    float complex r;
    cblas_cdotc_sub(*n, x, *incx, y, *incy, &r);
    return r;
}
