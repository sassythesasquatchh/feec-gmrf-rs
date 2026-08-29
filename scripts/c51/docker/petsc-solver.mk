# Build only the generalized Hermitian eigensolver required by C51. Keeping
# this makefile outside the FEEC source directory avoids cross-platform stale
# objects and Ubuntu's SLEPc `clean:`/FEEC `clean::` target conflict.

SOURCE_DIR ?= /workspace/feec/petsc-solver
VPATH := ${SOURCE_DIR}

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

.PHONY: all
all: ghiep.out

ghiep.out: ghiep.o
	${CLINKER} -o $@ $< ${SLEPC_EPS_LIB}
	${RM} $<
