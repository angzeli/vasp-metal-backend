! End-to-end Fortran ABI test of the interposed ZGEMM: column-major storage,
! submatrix starting elements with leading dimensions larger than the operand,
! all nine trans combinations, alpha/beta, beta = 0 with NaN-filled C, and the
! disabled route. The oracle is an independent quad-precision loop.
program test_fortran_abi
  use, intrinsic :: iso_c_binding
  use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
  implicit none

  interface
    subroutine vmb_stats_get(out) bind(c, name='vmb_stats_get')
      import :: c_int64_t
      integer(c_int64_t), intent(out) :: out(*)
    end subroutine
    subroutine vmb_config_reload() bind(c, name='vmb_config_reload')
    end subroutine
    integer(c_int) function setenv(name, val, overwrite) bind(c, name='setenv')
      import :: c_char, c_int
      character(kind=c_char), intent(in) :: name(*), val(*)
      integer(c_int), value :: overwrite
    end function
  end interface

  integer, parameter :: dp = kind(1.0d0), qp = selected_real_kind(30)
  ! Indices into vmb_stats_get (enum vmb_stat + 1).
  integer, parameter :: st_completed = 3, st_consumed = 4, st_cpu = 5
  integer, parameter :: lda = 14, ldb = 14, ldc = 10, m = 5, n = 6, k = 7
  integer, parameter :: ia = 2, ja = 3, ib = 3, jb = 2, ic = 2, jc = 2
  character(len=1), parameter :: flags(3) = ['N', 'T', 'C']
  complex(dp) :: a(lda, 14), b(ldb, 14), c0(ldc, 10), c(ldc, 10), alpha, beta
  integer(c_int64_t) :: s0(32), s1(32)
  integer :: x, y, failures, rc
  real(dp) :: worst
  complex(dp) :: da(2, 2), db(2, 2), dc(2, 2)

  failures = 0
  worst = 0
  call fill(a)
  call fill(b)
  call fill(c0)
  rc = setenv('VMB_ENABLE' // c_null_char, '1' // c_null_char, 1_c_int)
  call vmb_config_reload()

  alpha = (0.75_dp, -1.5_dp)
  beta = (-0.5_dp, 0.25_dp)
  do x = 1, 3
    do y = 1, 3
      call run(flags(x), flags(y), alpha, beta, .false., 1, 0)
    end do
  end do
  call run('C', 'N', (1.0_dp, 0.0_dp), (0.0_dp, 0.0_dp), .true., 1, 0)

  ! Regression for d3be811: diag(2^90, 2^-90) diag(2^-90, 2^90) = I exactly.
  ! Forced mode, shadow check off: the safety envelope must route this to the
  ! original OpenBLAS (counted) rather than consume a wrong GPU result.
  rc = setenv('VMB_MODE' // c_null_char, 'force' // c_null_char, 1_c_int)
  rc = setenv('VMB_CHECK' // c_null_char, '0' // c_null_char, 1_c_int)
  call vmb_config_reload()
  da = 0; db = 0; dc = (7.0_dp, 7.0_dp)
  da(1, 1) = 2.0_dp**90; da(2, 2) = 2.0_dp**(-90)
  db(1, 1) = 2.0_dp**(-90); db(2, 2) = 2.0_dp**90
  call vmb_stats_get(s0)
  call zgemm('N', 'N', 2, 2, 2, (1.0_dp, 0.0_dp), da, 2, db, 2, (0.0_dp, 0.0_dp), dc, 2)
  call vmb_stats_get(s1)
  write (*, '(a, 4f5.1, a, 2i2)') 'zgemm diag(2^+-90): C =', real(dc(1, 1)), real(dc(2, 2)), &
      abs(dc(1, 2)), abs(dc(2, 1)), '  gpu/cpu calls:', &
      s1(st_consumed) - s0(st_consumed), s1(st_cpu) - s0(st_cpu)
  if (dc(1, 1) /= (1.0_dp, 0.0_dp) .or. dc(2, 2) /= (1.0_dp, 0.0_dp) .or. &
      dc(1, 2) /= (0.0_dp, 0.0_dp) .or. dc(2, 1) /= (0.0_dp, 0.0_dp)) call flag('diag result /= I')
  if (s1(st_consumed) - s0(st_consumed) /= 0) call flag('diag consumed from GPU')
  if (s1(st_cpu) - s0(st_cpu) /= 1) call flag('diag CPU fallback count')
  rc = setenv('VMB_MODE' // c_null_char, 'off' // c_null_char, 1_c_int)
  call vmb_config_reload()
  call run('T', 'C', alpha, beta, .false., 0, 1)

  write (*, '(a, es10.3)') 'fortran abi: worst err/tol = ', worst
  if (failures /= 0) then
    write (*, '(a, i0, a)') 'FAIL: ', failures, ' fortran ABI failures'
    stop 1
  end if
  write (*, '(a)') 'PASS: fortran ABI (12 calls)'

contains

  subroutine fill(z)
    complex(dp), intent(out) :: z(:, :)
    integer :: i, j
    do j = 1, size(z, 2)
      do i = 1, size(z, 1)
        z(i, j) = cmplx(sin(1.3_dp * i + 0.7_dp * j), cos(0.9_dp * i - 1.1_dp * j) / (i + j), dp)
      end do
    end do
  end subroutine

  ! op(M)(r, s) for a submatrix of z starting at (i0, j0).
  complex(qp) function op(z, t, i0, j0, r, s)
    complex(dp), intent(in) :: z(:, :)
    character, intent(in) :: t
    integer, intent(in) :: i0, j0, r, s
    if (t == 'N') then
      op = z(i0 + r - 1, j0 + s - 1)
    else
      op = z(i0 + s - 1, j0 + r - 1)
      if (t == 'C') op = conjg(op)
    end if
  end function

  subroutine run(ta, tb, al, be, nan_c, want_gpu, want_cpu)
    character, intent(in) :: ta, tb
    complex(dp), intent(in) :: al, be
    logical, intent(in) :: nan_c
    integer, intent(in) :: want_gpu, want_cpu
    complex(qp) :: acc
    real(qp) :: s(m, n), sl, err
    complex(qp) :: ref(m, n)
    real(dp) :: tol, smax, ratio
    integer :: i, j, l
    logical :: outside_ok

    c = c0
    if (nan_c) c(ic:ic + m - 1, jc:jc + n - 1) = &
        cmplx(ieee_value(1.0_dp, ieee_quiet_nan), 0.0_dp, dp)
    do j = 1, n
      do i = 1, m
        acc = 0
        sl = 0
        do l = 1, k
          acc = acc + op(a, ta, ia, ja, i, l) * op(b, tb, ib, jb, l, j)
          sl = sl + abs(op(a, ta, ia, ja, i, l)) * abs(op(b, tb, ib, jb, l, j))
        end do
        ref(i, j) = cmplx(al, kind=qp) * acc
        s(i, j) = abs(al) * sl
        if (.not. nan_c) then
          ref(i, j) = ref(i, j) + cmplx(be, kind=qp) * c(ic + i - 1, jc + j - 1)
          s(i, j) = s(i, j) + abs(be) * abs(c(ic + i - 1, jc + j - 1))
        end if
      end do
    end do

    call vmb_stats_get(s0)
    call zgemm(ta, tb, m, n, k, al, a(ia, ja), lda, b(ib, jb), ldb, be, c(ic, jc), ldc)
    call vmb_stats_get(s1)

    smax = real(maxval(s), dp)
    ratio = 0
    do j = 1, n
      do i = 1, m
        err = abs(cmplx(c(ic + i - 1, jc + j - 1), kind=qp) - ref(i, j))
        tol = 1e-18_dp * smax + 8.0_dp * (k + 2) * 2.0_dp**(-48) * real(s(i, j), dp)
        ratio = max(ratio, real(err, dp) / tol)
      end do
    end do
    worst = max(worst, ratio)
    ! Every element outside the target block must be untouched.
    outside_ok = .true.
    do j = 1, size(c, 2)
      do i = 1, size(c, 1)
        if (i >= ic .and. i < ic + m .and. j >= jc .and. j < jc + n) cycle
        if (c(i, j) /= c0(i, j)) outside_ok = .false.
      end do
    end do

    write (*, '(a, 2a1, a, es10.3, a, 2i2)') 'zgemm ', ta, tb, '  err/tol=', ratio, &
        '  gpu/cpu calls:', s1(st_consumed) - s0(st_consumed), s1(st_cpu) - s0(st_cpu)
    if (.not. (ratio <= 1.0_dp)) call flag('error exceeds tolerance')
    if (.not. outside_ok) call flag('write outside the C block')
    if (s1(st_consumed) - s0(st_consumed) /= want_gpu) call flag('GPU consumption count')
    if (s1(st_completed) - s0(st_completed) /= want_gpu) call flag('GPU completion count')
    if (s1(st_cpu) - s0(st_cpu) /= want_cpu) call flag('CPU fallback count')
  end subroutine

  subroutine flag(what)
    character(*), intent(in) :: what
    failures = failures + 1
    write (*, '(2a)') '  FAIL: ', what
  end subroutine

end program test_fortran_abi
