# Serial build of libvmb.a and its standalone tests. Run with -j1.
.NOTPARALLEL:

# Toolchain versions follow the VASP port's convention (PORT_* keys). Defaults
# are the original M3 Max port; `make PORT_ROOT=/path/to/port` reads that
# checkout's config/toolchain-defaults.txt plus its host-local
# private/toolchain.env through the port's own scripts/toolchain.py.
PORT_GCC_VERSION      ?= 16.1.0
PORT_OPENBLAS_VERSION ?= 0.3.34
ifdef PORT_ROOT
PORT_TOOLCHAIN := $(shell /usr/bin/python3 -I -B $(PORT_ROOT)/scripts/toolchain.py)
ifeq ($(PORT_TOOLCHAIN),)
$(error cannot read the toolchain of PORT_ROOT=$(PORT_ROOT))
endif
PORT_GCC_VERSION      := $(patsubst PORT_GCC_VERSION=%,%,$(filter PORT_GCC_VERSION=%,$(PORT_TOOLCHAIN)))
PORT_OPENBLAS_VERSION := $(patsubst PORT_OPENBLAS_VERSION=%,%,$(filter PORT_OPENBLAS_VERSION=%,$(PORT_TOOLCHAIN)))
endif

CC       = clang
FC       = /opt/homebrew/Cellar/gcc/$(PORT_GCC_VERSION)/bin/gfortran
OPENBLAS = /opt/homebrew/Cellar/openblas/$(PORT_OPENBLAS_VERSION)
BUILD    = build

# Build provenance embedded in rank reports. The digest covers the library's
# source files, so an uncommitted tree is still identified exactly.
SOURCE_COMMIT ?= $(shell git rev-parse HEAD 2>/dev/null || echo unknown)
SOURCE_DIGEST := $(shell scripts/source-digest.sh)
SOURCE_DIRTY  ?= $(shell test -z "$$(git status --porcelain -- src include Makefile scripts/source-digest.sh 2>/dev/null)" && git rev-parse HEAD >/dev/null 2>&1 && echo 0 || echo 1)
IDENTITY  = -DVMB_SOURCE_COMMIT='"$(SOURCE_COMMIT)"' -DVMB_SOURCE_DIGEST='"$(SOURCE_DIGEST)"' \
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
     $(BUILD)/rank_probe

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

$(BUILD)/vmb_rank_report.o: src/vmb_rank_report.c include/vmb.h src/vmb_internal.h $(BUILD)/source-identity | $(BUILD)
	$(CC) $(CFLAGS) $(IDENTITY) -c $< -o $@

# Rebuilt only when the embedded identity changes.
$(BUILD)/source-identity: FORCE | $(BUILD)
	@echo '$(SOURCE_COMMIT) $(SOURCE_DIGEST) $(SOURCE_DIRTY)' | cmp -s - $@ || \
	    echo '$(SOURCE_COMMIT) $(SOURCE_DIGEST) $(SOURCE_DIRTY)' > $@

$(BUILD)/libvmb.a: $(BUILD)/vmb_dispatch.o $(BUILD)/vmb_metal.o $(BUILD)/vmb_telemetry.o \
                   $(BUILD)/vmb_rank_report.o
	rm -f $@
	ar rcs $@ $^

$(BUILD)/test_zgemm: tests/test_zgemm.c tests/oracle.c tests/oracle.h $(BUILD)/libvmb.a
	$(CC) $(CFLAGS) -Itests tests/test_zgemm.c tests/oracle.c $(LIBS) -o $@

$(BUILD)/test_fortran_abi: tests/test_fortran_abi.f90 $(BUILD)/libvmb.a
	$(FC) $(FFLAGS) -J$(BUILD) $< $(LIBS) -o $@

$(BUILD)/shape_replay: bench/shape_replay.c $(BUILD)/libvmb.a
	$(CC) $(CFLAGS) $< $(LIBS) -o $@

# Harmless per-rank report fixture: calls zgemm_ only on chosen ranks.
$(BUILD)/rank_probe: tests/rank_probe.c $(BUILD)/libvmb.a
	$(CC) $(CFLAGS) $< $(LIBS) -o $@

test: all
	$(BUILD)/test_zgemm
	$(BUILD)/test_fortran_abi
	python3 -B tests/test_rank_reports.py

test-python:
	python3 -B tests/test_rank_reports.py
	python3 -B tests/test_compare_runs.py

clean:
	rm -rf $(BUILD)
