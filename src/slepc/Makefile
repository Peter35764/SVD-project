CLINKER = mpicc

PETSC_DIR=/home/jess/develop/slepc/petsc
SLEPC_DIR=/home/jess/develop/slepc/slepc
PETSC_ARCH=arch-linux-c-debug

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

all: slepc_example

slepc_example: slepc_example.o #chkopts
	${CLINKER} -o slepc_example slepc_example.o ${SLEPC_SYS_LIB} ${PETSC_LIB}

clean_:
	${RM} slepc_example.o slepc_example

clang_format: # style linter
	clang-format -i clang_format/*.{c,cpp,h}
