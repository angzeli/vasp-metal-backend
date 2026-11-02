# Serial build of libvmb.a and its standalone tests. Run with -j1.
.NOTPARALLEL:

CC       = clang
FC       = /opt/homebrew/Cellar/gcc/16.1.0/bin/gfortran
OPENBLAS = /opt/homebrew/Cellar/openblas/0.3.34
BUILD    = build

MACOS_MIN = 26.0
CFLAGS    = -O2 -Wall -Wextra -std=c11 -mmacosx-version-min=$(MACOS_MIN) -Iinclude -I$(BUILD)
OBJCFLAGS = -O2 -Wall -Wextra -fobjc-arc -mmacosx-version-min=$(MACOS_MIN) -Iinclude -I$(BUILD)
FFLAGS    = -O1 -Wall
# Link order matters: libvmb.a must precede OpenBLAS so zgemm_ resolves here.
LIBS      = $(BUILD)/libvmb.a -framework Metal -framework Foundation -lobjc \
            -L$(OPENBLAS)/lib -lopenblas

.PHONY: all test clean
all: $(BUILD)/libvmb.a $(BUILD)/test_zgemm $(BUILD)/test_fortran_abi

$(BUILD):
	mkdir -p $@

$(BUILD)/zgemm_df64_metal.h: src/zgemm_df64.metal | $(BUILD)
	xxd -i -n vmb_kernel_source $< > $@

$(BUILD)/vmb_dispatch.o: src/vmb_dispatch.c include/vmb.h src/vmb_internal.h | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD)/vmb_metal.o: src/vmb_metal.m src/vmb_internal.h $(BUILD)/zgemm_df64_metal.h
	$(CC) $(OBJCFLAGS) -c $< -o $@

$(BUILD)/libvmb.a: $(BUILD)/vmb_dispatch.o $(BUILD)/vmb_metal.o
	rm -f $@
	ar rcs $@ $^

$(BUILD)/test_zgemm: tests/test_zgemm.c tests/oracle.c tests/oracle.h $(BUILD)/libvmb.a
	$(CC) $(CFLAGS) -Itests tests/test_zgemm.c tests/oracle.c $(LIBS) -o $@

$(BUILD)/test_fortran_abi: tests/test_fortran_abi.f90 $(BUILD)/libvmb.a
	$(FC) $(FFLAGS) -J$(BUILD) $< $(LIBS) -o $@

test: all
	$(BUILD)/test_zgemm
	$(BUILD)/test_fortran_abi

clean:
	rm -rf $(BUILD)
