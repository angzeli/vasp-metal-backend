#!/usr/bin/env bash
# Relink an existing, separately licensed VASP std object build with libvmb.a
# placed ahead of OpenBLAS, so the program's zgemm_ references bind to the
# backend. Nothing is compiled from VASP source; the object directory is only
# read. The executable is written to a new directory, which must not exist.
#
# usage: scripts/relink-vasp.sh OBJECT_DIR NEW_OUTPUT_DIR
#   OBJECT_DIR      VASP build/std directory holding *.o, lib/libdmy.a, parser/libparser.a
#   NEW_OUTPUT_DIR  receives vasp_std_vmb and link.log (keep it under private/)
set -euo pipefail

if [[ $# -ne 2 ]]; then
    sed -n '2,10p' "$0" >&2
    exit 2
fi
objdir=$1
out=$2
root=$(cd -- "$(dirname -- "$0")/.." && pwd)

# Toolchain of the validated CPU reference build (process-local selection).
gnu=/opt/homebrew/Cellar/gcc/16.1.0
mpi=/opt/homebrew/Cellar/open-mpi/5.0.9
scalapack=/opt/homebrew/Cellar/scalapack/2.2.3
openblas=/opt/homebrew/Cellar/openblas/0.3.34
fftw=/opt/homebrew/Cellar/fftw/3.3.11
export OMPI_FC=$gnu/bin/gfortran OMPI_CC=$gnu/bin/gcc-16 OMPI_CXX=$gnu/bin/g++-16

for f in "$root/build/libvmb.a" "$objdir/main.o" "$objdir/lib/libdmy.a" "$objdir/parser/libparser.a"; do
    [[ -f $f ]] || { echo "missing: $f" >&2; exit 2; }
done
mkdir "$out"  # fails if it already exists

objects=("$objdir"/*.o)
if ! "$mpi/bin/mpifort" -o "$out/vasp_std_vmb" "${objects[@]}" \
        "$objdir/parser/libparser.a" "$objdir/lib/libdmy.a" \
        "$root/build/libvmb.a" -framework Metal -framework Foundation -lobjc \
        -lstdc++ -L"$scalapack/lib" -lscalapack -L"$openblas/lib" -lopenblas \
        -L"$fftw/lib" -lfftw3 >"$out/link.log" 2>&1; then
    echo "link failed; private log: $out/link.log" >&2
    exit 1
fi
echo "linked ${#objects[@]} objects -> $out/vasp_std_vmb"
