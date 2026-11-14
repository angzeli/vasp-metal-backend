! ABI check for linking VASP directly against a BLAS/LAPACK (#9): calls every
! value-returning BLAS/LAPACK function that vasp_std imports, declared the way
! VASP's Fortran declares them (plain EXTERNAL functions, gfortran convention),
! and compares each result with a value computed here. Inputs are small integers,
! so every expected result is exact.
!
! Accelerate's legacy BLAS follows the f2c convention for complex functions
! (the result is returned through a hidden first argument), which gfortran does
! not use, so cdotc and zdotc are expected to fail or crash there. Each function
! is selected by name on the command line, so one crash does not hide the rest:
!   test_accelerate_abi NAME   -> prints "NAME ok" or "NAME FAIL got ... want ..."
program test_accelerate_abi
  implicit none
  integer, parameter :: dp = kind(1.0d0), sp = kind(1.0)
  integer, parameter :: n = 5
  character(len=16) :: name
  real(dp) :: dx(n), dy(n), got, want
  complex(dp) :: zx(n), zy(n), zgot, zwant, za(3, 2)
  complex(sp) :: cx(n), cy(n), cgot, cwant
  integer :: i, igot, iwant
  real(dp) :: work(3)
  real(dp), external :: ddot, dasum, dnrm2, dzasum, zlange
  complex(dp), external :: zdotc
  complex(sp), external :: cdotc
  integer, external :: idamax, izamax, ilaenv

  call get_command_argument(1, name)
  do i = 1, n
    dx(i) = i
    dy(i) = 2 * i - 7
    zx(i) = cmplx(i, -i + 2, dp)
    zy(i) = cmplx(3 - i, 2 * i, dp)
    cx(i) = cmplx(i, -i + 2, sp)
    cy(i) = cmplx(3 - i, 2 * i, sp)
  end do
  za = reshape([cmplx(1, 2, dp), cmplx(-3, 0, dp), cmplx(0, 4, dp), &
                cmplx(2, -2, dp), cmplx(5, 0, dp), cmplx(-1, -1, dp)], [3, 2])

  select case (trim(name))
  case ('ddot')
    call check_real(ddot(n, dx, 1, dy, 1), sum(dx * dy))
  case ('dasum')
    call check_real(dasum(n, dy, 1), sum(abs(dy)))
  case ('dnrm2')
    call check_real(dnrm2(4, [3.0_dp, 4.0_dp, 12.0_dp, 0.0_dp], 1), 13.0_dp)
  case ('dzasum')
    call check_real(dzasum(n, zx, 1), sum(abs(real(zx)) + abs(aimag(zx))))
  case ('zlange')  ! max-abs and one norms of a 3x2 complex matrix
    call check_real(zlange('M', 3, 2, za, 3, work), 5.0_dp)
    call check_real(zlange('1', 3, 2, za, 3, work), max(sqrt(5.0_dp) + 3 + 4, &
                    sqrt(8.0_dp) + 5 + sqrt(2.0_dp)))
  case ('idamax')
    call check_int(idamax(n, dy, 1), 1)
  case ('izamax')
    call check_int(izamax(n, zy, 1), 5)
  case ('ilaenv')  ! block size for ZHETRD: a positive integer from any LAPACK
    igot = ilaenv(1, 'ZHETRD', 'U', 100, -1, -1, -1)
    if (igot >= 1 .and. igot <= 512) then
      print '(a, " ok (", i0, ")")', trim(name), igot
    else
      print '(a, " FAIL got ", i0)', trim(name), igot
    end if
  case ('zdotc')
    zgot = zdotc(n, zx, 1, zy, 1)
    zwant = sum(conjg(zx) * zy)
    if (zgot == zwant) then
      print '(a, " ok")', trim(name)
    else
      print '(a, " FAIL got ", 2es12.4, " want ", 2es12.4)', trim(name), zgot, zwant
    end if
  case ('cdotc')
    cgot = cdotc(n, cx, 1, cy, 1)
    cwant = sum(conjg(cx) * cy)
    if (cgot == cwant) then
      print '(a, " ok")', trim(name)
    else
      print '(a, " FAIL got ", 2es12.4, " want ", 2es12.4)', trim(name), cgot, cwant
    end if
  case default
    print '("unknown function: ", a)', trim(name)
    stop 2
  end select

contains

  subroutine check_real(g, w)
    real(dp), intent(in) :: g, w
    got = g
    want = w
    if (abs(got - want) <= 1e-12_dp * max(1.0_dp, abs(want))) then
      print '(a, " ok")', trim(name)
    else
      print '(a, " FAIL got ", es22.14, " want ", es22.14)', trim(name), got, want
    end if
  end subroutine

  subroutine check_int(g, w)
    integer, intent(in) :: g, w
    igot = g
    iwant = w
    if (igot == iwant) then
      print '(a, " ok")', trim(name)
    else
      print '(a, " FAIL got ", i0, " want ", i0)', trim(name), igot, iwant
    end if
  end subroutine

end program
