CLVK_HEADERS = /opt/clvk/external/OpenCL-Headers/
CLVK_LIB = /opt/clvk/build/
COMMON_DIR = ../../C_common

ifndef CC
	CC = gcc
endif

CCFLAGS=-O3 -g3 -ffast-math -fopenmp -I $(CLVK_HEADERS) -I $(COMMON_DIR)

LDFLAGS  = -lm -lOpenCL -fopenmp -L$(CLVK_LIB)

# list out all .o’s
OBJS = \
  main.o \
  opencl.o \
  openmp.o \
  serial.o \
  sparse_formats.o \
  ziggurat.o \
  cl_utils.o \
  wtime.o \
  err_code.o \
  benchmark.o

all: spmv

spmv: $(OBJS)
	$(CC) $^ $(CCFLAGS) $(LDFLAGS) -o $@

%.o: %.c
	$(CC) -c $< $(CCFLAGS) -o $@

err_code.o: $(COMMON_DIR)/err_code.c $(COMMON_DIR)/err_code.h
	$(CC) -c $< $(CCFLAGS) -o $@

cl_utils.o: $(COMMON_DIR)/cl_utils.c $(COMMON_DIR)/cl_utils.h
	$(CC) -c $< $(CCFLAGS) -o $@

benchmark.o: $(COMMON_DIR)/benchmark.c $(COMMON_DIR)/benchmark.h
	$(CC) -c $< $(CCFLAGS) -o $@

wtime.o: $(COMMON_DIR)/wtime.c $(COMMON_DIR)/wtime.h
	$(CC) -c $< $(CCFLAGS) -o $@	


.PHONY: all clean
clean:
	rm -f spmv $(OBJS)