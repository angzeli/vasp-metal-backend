#!/usr/bin/env bash
# Relink an existing, separately licensed VASP std object build with libvmb.a
# placed ahead of OpenBLAS, so the program's zgemm_ references bind to the
# backend. Nothing is compiled from VASP source; the object directory is only
# read. The executable is written to a new directory, which must not exist.
#
# usage: scripts/relink-vasp.sh [--port-root PORT_ROOT] [--direct-accelerate] [--fftw-prefix PREFIX]
#                               OBJECT_DIR NEW_OUTPUT_DIR
#   --direct-accelerate  baseline for #9: no libvmb.a and no OpenBLAS on the VASP
#                   link line; BLAS/LAPACK from Accelerate, plus
#                   scripts/accelerate-shim.c for the complex dot functions.
#                   ScaLAPACK still loads its own BLAS (recorded in relink.txt).
#   --fftw-prefix   link this FFTW install (PREFIX/lib/libfftw3.dylib, same
#                   version as the port's) instead of Homebrew's, e.g. a NEON
#                   build (v0.6); its path and library digest go in relink.txt.
#   PORT_ROOT       VASP port checkout whose toolchain to use: its
#                   config/toolchain-defaults.txt plus host-local
#                   private/toolchain.env, read by its scripts/toolchain.py.
#                   Without it, the original M3 Max port versions are used.
#   OBJECT_DIR      VASP build/std directory holding *.o, lib/libdmy.a, parser/libparser.a
#   NEW_OUTPUT_DIR  receives vasp_std_vmb, link.log and relink.txt (keep it private)
set -euo pipefail

port_root=
if [[ ${1:-} == --port-root ]]; then
    [[ $# -ge 2 ]] || { sed -n '2,21p' "$0" >&2; exit 2; }
    port_root=$2
    shift 2
fi
direct=0
if [[ ${1:-} == --direct-accelerate ]]; then
    direct=1
    shift
fi
fftw_prefix=
if [[ ${1:-} == --fftw-prefix ]]; then
    [[ $# -ge 2 && -f $2/lib/libfftw3.dylib ]] || { echo "--fftw-prefix needs PREFIX/lib/libfftw3.dylib" >&2; exit 2; }
    fftw_prefix=$(cd -- "$2" && pwd -P)
    shift 2
fi
if [[ $# -ne 2 ]]; then
    sed -n '2,21p' "$0" >&2
    exit 2
fi
objdir=$1
out=$2
root=$(cd -- "$(dirname -- "$0")/.." && pwd)
# A relinked executable contains licensed VASP code: inside this repository it
# may only be written under the ignored private/ directory.
parent=$(cd -- "$(dirname -- "$out")" 2>/dev/null && pwd -P) || { echo "output parent must exist: $out" >&2; exit 2; }
case "$parent/" in
    "$(cd -- "$root" && pwd -P)/private/"*) ;;
    "$(cd -- "$root" && pwd -P)/"*) echo "refusing to write a relinked VASP executable inside the repository outside private/: $out" >&2; exit 2;;
esac

PORT_GCC_VERSION=16.1.0 PORT_OPENMPI_VERSION=5.0.9 PORT_SCALAPACK_VERSION=2.2.3
PORT_OPENBLAS_VERSION=0.3.34 PORT_FFTW_VERSION=3.3.11 PORT_TOOLCHAIN_SOURCE=built-in
if [[ -n $port_root ]]; then
    toolchain=$(/usr/bin/python3 -I -B "$port_root/scripts/toolchain.py" --shell)
    eval "$toolchain"  # quoted PORT_*=VERSION exports produced by the port itself
    PORT_TOOLCHAIN_SOURCE="$port_root: $PORT_TOOLCHAIN_SOURCE"
fi
# Process-local selection of the validated toolchain.
gnu=/opt/homebrew/Cellar/gcc/$PORT_GCC_VERSION
mpi=/opt/homebrew/Cellar/open-mpi/$PORT_OPENMPI_VERSION
scalapack=/opt/homebrew/Cellar/scalapack/$PORT_SCALAPACK_VERSION
openblas=/opt/homebrew/Cellar/openblas/$PORT_OPENBLAS_VERSION
fftw=${fftw_prefix:-/opt/homebrew/Cellar/fftw/$PORT_FFTW_VERSION}
export OMPI_FC=$gnu/bin/gfortran OMPI_CC=$gnu/bin/gcc-${PORT_GCC_VERSION%%.*} OMPI_CXX=$gnu/bin/g++-${PORT_GCC_VERSION%%.*}

for f in "$root/build/libvmb.a" "$root/scripts/accelerate-shim.c" "$objdir/main.o" "$objdir/lib/libdmy.a" "$objdir/parser/libparser.a" \
         "$mpi/bin/mpifort" "$OMPI_FC"; do
    [[ -f $f ]] || { echo "missing: $f" >&2; exit 2; }
done
mkdir "$out"  # fails if it already exists

objects=("$objdir"/*.o)
if [[ $direct == 1 ]]; then
    # The executable keeps its usual name so the manifest tooling finds it.
    library=$out/accelerate-shim.o
    /usr/bin/clang -O2 -Wall -c "$root/scripts/accelerate-shim.c" -o "$library"
    # mpifort links with -flat_namespace: a symbol resolves to the first loaded
    # image that defines it, and ScaLAPACK loads OpenBLAS. Accelerate must come
    # first in the load order, or VASP silently runs OpenBLAS.
    first=(-Wl,-framework,Accelerate)
    link=("$library" -lstdc++ -L"$scalapack/lib" -lscalapack)
else
    library=$root/build/libvmb.a
    first=()
    link=("$library" -framework Metal -framework Foundation -lobjc
          -lstdc++ -L"$scalapack/lib" -lscalapack -L"$openblas/lib" -lopenblas)
fi
if ! "$mpi/bin/mpifort" -o "$out/vasp_std_vmb" ${first[@]+"${first[@]}"} "${objects[@]}" \
        "$objdir/parser/libparser.a" "$objdir/lib/libdmy.a" \
        "${link[@]}" -L"$fftw/lib" -lfftw3 >"$out/link.log" 2>&1; then
    echo "link failed; private log: $out/link.log" >&2
    exit 1
fi
{
    echo "objects=$objdir"
    echo "object_count=${#objects[@]}"
    echo "toolchain_source=$PORT_TOOLCHAIN_SOURCE"
    for key in PORT_GCC_VERSION PORT_OPENMPI_VERSION PORT_SCALAPACK_VERSION PORT_OPENBLAS_VERSION PORT_FFTW_VERSION; do
        echo "$key=${!key}"
    done
    echo "variant=$([[ $direct == 1 ]] && echo direct-accelerate || echo libvmb)"
    fftw_lib=$(otool -L "$out/vasp_std_vmb" | awk '/libfftw3/ {print $1}')
    echo "fftw=$fftw_lib"
    echo "fftw_sha256=$(shasum -a 256 "$fftw_lib" | cut -d' ' -f1)"
    echo "library_sha256=$(shasum -a 256 "$library" | cut -d' ' -f1)"
    if [[ $direct == 1 ]]; then
        # Open MPI's mpifort links with -flat_namespace, so VASP's BLAS references
        # resolve at run time by image load order, not at link time. Record that
        # order and the BLAS ScaLAPACK itself loads; which library actually ran
        # has to be checked on a running process.
        echo "flat_namespace=$(dyld_info -fixups "$out/vasp_std_vmb" | grep -cE 'flat-namespace>/_dgemm_$')"
        echo "load_order=$(otool -L "$out/vasp_std_vmb" | awk 'NR > 1 {n = split($1, p, "/"); printf "%s ", p[n]}')"
        echo "scalapack_loads=$(otool -L "$scalapack/lib/libscalapack.dylib" | grep -oE 'lib[A-Za-z]*blas[^ ]*' | tr '\n' ' ')"
    fi
    echo "relink_script_sha256=$(shasum -a 256 "$0" | cut -d' ' -f1)"
    echo "built_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"$out/relink.txt"
echo "linked ${#objects[@]} objects -> $out/vasp_std_vmb"
