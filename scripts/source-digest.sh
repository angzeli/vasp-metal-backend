#!/bin/sh
# SHA-256 over the library's source files (name and bytes, sorted by name).
# It identifies the exact source of an uncommitted build; run from the root.
cd "$(dirname -- "$0")/.." || exit 1
for f in $(LC_ALL=C ls src/*.c src/*.m src/*.h src/*.metal include/*.h Makefile); do
    printf '%s %s\n' "$(shasum -a 256 < "$f" | cut -d' ' -f1)" "$f"
done | shasum -a 256 | cut -d' ' -f1
