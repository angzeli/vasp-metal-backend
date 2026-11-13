# Serial build of libvmb.a and its standalone tests. Run with -j1.
.NOTPARALLEL:

# Toolchain versions follow the VASP port's convention (PORT_* keys). Defaults
# are the original M3 Max port; `make PORT_ROOT=/path/to/port` reads that
# checkout's config/toolchain-defaults.txt plus its host-local
# private/toolchain.env through the port's own scripts/toolchain.py.
PORT_GCC_VERSION      ?= 16.1.0
PORT_OPENBLAS_VERSION ?= 0.3.34
PORT_FFTW_VERSION     ?= 3.3.11
ifdef PORT_ROOT
PORT_TOOLCHAIN := $(shell /usr/bin/python3 -I -B $(PORT_ROOT)/scripts/toolchain.py)
ifeq ($(PORT_TOOLCHAIN),)
$(error cannot read the toolchain of PORT_ROOT=$(PORT_ROOT))
endif
PORT_GCC_VERSION      := $(patsubst PORT_GCC_VERSION=%,%,$(filter PORT_GCC_VERSION=%,$(PORT_TOOLCHAIN)))
PORT_OPENBLAS_VERSION := $(patsubst PORT_OPENBLAS_VERSION=%,%,$(filter PORT_OPENBLAS_VERSION=%,$(PORT_TOOLCHAIN)))
PORT_FFTW_VERSION     := $(patsubst PORT_FFTW_VERSION=%,%,$(filter PORT_FFTW_VERSION=%,$(PORT_TOOLCHAIN)))
endif

CC       = clang
FC       = /opt/homebrew/Cellar/gcc/$(PORT_GCC_VERSION)/bin/gfortran
OPENBLAS = /opt/homebrew/Cellar/openblas/$(PORT_OPENBLAS_VERSION)
FFTW     = /opt/homebrew/Cellar/fftw/$(PORT_FFTW_VERSION)
BUILD    = build

# Build provenance embedded in rank reports. The digest covers the library's
# source files, so an uncommitted tree is still identified exactly.
SOURCE_COMMIT ?= $(shell git rev-parse HEAD 2>/dev/null || echo unknown)
SOURCE_DIGEST := $(shell scripts/source-digest.sh)
SOURCE_DIRTY  ?= $(shell test -z "$$(git status --porcelain -- src include Makefile scripts/source-digest.sh 2>/dev/null)" && git rev-parse HEAD >/dev/null 2>&1 && echo 0 || echo 1)
# Build-time default of VMB_PROFILE (0 off, 1 profile, 2 reuse probe). The VASP
# port runner passes no VMB_PROFILE, so profiled VASP runs use a binary built
# with PROFILE_DEFAULT=1 (or 2), selected by its own reviewed manifest.
PROFILE_DEFAULT ?= 0
# Build-time default of VMB_FFTW_PLAN_CACHE (0 off, 1 on), for the same reason.
PLAN_CACHE_DEFAULT ?= 0
# Build-time default of VMB_DBLAS_LIBRARY (0 openblas, 1 accelerate), likewise.
DBLAS_ACCELERATE_DEFAULT ?= 0
IDENTITY  = -DVMB_PROFILE_DEFAULT=$(PROFILE_DEFAULT) -DVMB_SOURCE_COMMIT='"$(SOURCE_COMMIT)"' -DVMB_SOURCE_DIGEST='"$(SOURCE_DIGEST)"' \
            -DVMB_SOURCE_DIRTY=$(SOURCE_DIRTY)

MACOS_MIN = 26.0
CFLAGS    = -O2 -Wall -Wextra -std=c11 -mmacosx-version-min=$(MACOS_MIN) -Iinclude -I$(BUILD)
OBJCFLAGS = -O2 -Wall -Wextra -fobjc-arc -mmacosx-version-min=$(MACOS_MIN) -Iinclude -I$(BUILD)
FFLAGS    = -O1 -Wall
# Link order matters: libvmb.a must precede OpenBLAS so zgemm_ resolves here.
LIBS      = $(BUILD)/libvmb.a -framework Metal -framework Foundation -lobjc \
            -L$(OPENBLAS)/lib -lopenblas

.PHONY: all test test-python bench clean FORCE
all: $(BUILD)/libvmb.a $(BUILD)/test_zgemm $(BUILD)/test_fortran_abi $(BUILD)/shape_replay \
     $(BUILD)/rank_probe $(BUILD)/test_blas_fftw $(BUILD)/blas_replay

$(BUILD):
	mkdir -p $@

$(BUILD)/zgemm_df64_metal.h: src/zgemm_df64.metal | $(BUILD)
	xxd -i -n vmb_kernel_source $< > $@

$(BUILD)/vmb_dispatch.o: src/vmb_dispatch.c include/vmb.h src/vmb_internal.h | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD)/vmb_metal.o: src/vmb_metal.m src/vmb_internal.h $(BUILD)/zgemm_df64_metal.h
	$(CC) $(OBJCFLAGS) -c $< -o $@

$(BUILD)/vmb_telemetry.o: src/vmb_telemetry.c include/vmb.h src/vmb_internal.h | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD)/vmb_blas.o: src/vmb_blas.c include/vmb.h src/vmb_internal.h $(BUILD)/source-identity | $(BUILD)
	$(CC) $(CFLAGS) -DVMB_DBLAS_ACCELERATE_DEFAULT=$(DBLAS_ACCELERATE_DEFAULT) -c $< -o $@

$(BUILD)/vmb_fftw.o: src/vmb_fftw.c include/vmb.h src/vmb_internal.h $(BUILD)/source-identity | $(BUILD)
	$(CC) $(CFLAGS) -DVMB_PLAN_CACHE_DEFAULT=$(PLAN_CACHE_DEFAULT) -c $< -o $@

$(BUILD)/vmb_rank_report.o: src/vmb_rank_report.c include/vmb.h src/vmb_internal.h $(BUILD)/source-identity | $(BUILD)
	$(CC) $(CFLAGS) $(IDENTITY) -c $< -o $@

# Rebuilt only when the embedded identity changes. The objects are also removed,
# because make 3.81 compares whole-second times and could keep a stale one.
$(BUILD)/source-identity: FORCE | $(BUILD)
	@echo '$(SOURCE_COMMIT) $(SOURCE_DIGEST) $(SOURCE_DIRTY) $(PROFILE_DEFAULT) $(PLAN_CACHE_DEFAULT) $(DBLAS_ACCELERATE_DEFAULT)' | cmp -s - $@ || \
	    { echo '$(SOURCE_COMMIT) $(SOURCE_DIGEST) $(SOURCE_DIRTY) $(PROFILE_DEFAULT) $(PLAN_CACHE_DEFAULT) $(DBLAS_ACCELERATE_DEFAULT)' > $@; \
	      rm -f $(BUILD)/vmb_rank_report.o $(BUILD)/vmb_fftw.o $(BUILD)/vmb_blas.o; }

$(BUILD)/libvmb.a: $(BUILD)/vmb_dispatch.o $(BUILD)/vmb_metal.o $(BUILD)/vmb_telemetry.o \
                   $(BUILD)/vmb_rank_report.o $(BUILD)/vmb_blas.o $(BUILD)/vmb_fftw.o
	rm -f $@
	ar rcs $@ $^

$(BUILD)/test_zgemm: tests/test_zgemm.c tests/oracle.c tests/oracle.h $(BUILD)/libvmb.a
	$(CC) $(CFLAGS) -Itests tests/test_zgemm.c tests/oracle.c $(LIBS) -o $@

$(BUILD)/test_fortran_abi: tests/test_fortran_abi.f90 $(BUILD)/libvmb.a
	$(FC) $(FFLAGS) -J$(BUILD) $< $(LIBS) -o $@

$(BUILD)/shape_replay: bench/shape_replay.c $(BUILD)/libvmb.a
	$(CC) $(CFLAGS) $< $(LIBS) -o $@

# DGEMV/DGEMM replay, OpenBLAS against Accelerate: linked to OpenBLAS directly,
# without libvmb.a, so the measured calls do not pass through the interposer.
$(BUILD)/blas_replay: bench/blas_replay.c | $(BUILD)
	$(CC) $(CFLAGS) $< -L$(OPENBLAS)/lib -lopenblas -o $@

# Harmless per-rank report fixture: calls zgemm_ only on chosen ranks.
$(BUILD)/rank_probe: tests/rank_probe.c $(BUILD)/libvmb.a
	$(CC) $(CFLAGS) $< $(LIBS) -o $@

# Forwarding of the v0.5 interposers, linked like VASP: libvmb.a ahead of
# OpenBLAS and FFTW.
$(BUILD)/test_blas_fftw: tests/test_blas_fftw.c $(BUILD)/libvmb.a
	$(CC) $(CFLAGS) -I$(FFTW)/include $< $(LIBS) -L$(FFTW)/lib -lfftw3 -o $@

test: all
	$(BUILD)/test_zgemm
	$(BUILD)/test_fortran_abi
	$(BUILD)/test_blas_fftw
	python3 -B tests/test_profile_route.py
	python3 -B tests/test_rank_reports.py
	python3 -B tests/test_vmb_shapes.py
	python3 -B tests/test_vmb_profile.py

test-python:
	python3 -B tests/test_rank_reports.py
	python3 -B tests/test_profile_route.py
	python3 -B tests/test_compare_runs.py
	python3 -B tests/test_vmb_shapes.py
	python3 -B tests/test_vmb_profile.py

clean:
	rm -rf $(BUILD)
