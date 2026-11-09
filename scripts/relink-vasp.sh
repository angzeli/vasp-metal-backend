#!/usr/bin/env bash
# Relink an existing, separately licensed VASP std object build with libvmb.a
# placed ahead of OpenBLAS, so the program's zgemm_ references bind to the
# backend. Nothing is compiled from VASP source; the object directory is only
# read. The executable is written to a new directory, which must not exist.
#
# usage: scripts/relink-vasp.sh [--port-root PORT_ROOT] OBJECT_DIR NEW_OUTPUT_DIR
#   PORT_ROOT       VASP port checkout whose toolchain to use: its
#                   config/toolchain-defaults.txt plus host-local
#                   private/toolchain.env, read by its scripts/toolchain.py.
#                   Without it, the original M3 Max port versions are used.
#   OBJECT_DIR      VASP build/std directory holding *.o, lib/libdmy.a, parser/libparser.a
#   NEW_OUTPUT_DIR  receives vasp_std_vmb, link.log and relink.txt (keep it private)
set -euo pipefail

port_root=
if [[ ${1:-} == --port-root ]]; then
    [[ $# -ge 2 ]] || { sed -n '2,13p' "$0" >&2; exit 2; }
    port_root=$2
    shift 2
fi
if [[ $# -ne 2 ]]; then
    sed -n '2,13p' "$0" >&2
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
fftw=/opt/homebrew/Cellar/fftw/$PORT_FFTW_VERSION
export OMPI_FC=$gnu/bin/gfortran OMPI_CC=$gnu/bin/gcc-${PORT_GCC_VERSION%%.*} OMPI_CXX=$gnu/bin/g++-${PORT_GCC_VERSION%%.*}

for f in "$root/build/libvmb.a" "$objdir/main.o" "$objdir/lib/libdmy.a" "$objdir/parser/libparser.a" \
         "$mpi/bin/mpifort" "$OMPI_FC"; do
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
{
    echo "objects=$objdir"
    echo "object_count=${#objects[@]}"
    echo "toolchain_source=$PORT_TOOLCHAIN_SOURCE"
    for key in PORT_GCC_VERSION PORT_OPENMPI_VERSION PORT_SCALAPACK_VERSION PORT_OPENBLAS_VERSION PORT_FFTW_VERSION; do
        echo "$key=${!key}"
    done
    echo "library_sha256=$(shasum -a 256 "$root/build/libvmb.a" | cut -d' ' -f1)"
    echo "relink_script_sha256=$(shasum -a 256 "$0" | cut -d' ' -f1)"
    echo "built_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"$out/relink.txt"
echo "linked ${#objects[@]} objects -> $out/vasp_std_vmb"
