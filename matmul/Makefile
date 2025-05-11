CLVK_HEADERS = /opt/clvk/external/OpenCL-Headers/
CLVK_LIB = /opt/clvk/build/
COMMON_DIR = ../C_common

#
# Matrix Multiply example makefile
#
# History: Written  by Tim mattson, August 2010
#          Modified by Tom Deakin and Simon McIntosh-Smith, October 2012
#          Modified by Tom Deakin, July 2013
#          Modified by Tom Deakin, October 2014
#

ifndef CC
	CC = gcc
endif

CCFLAGS=-O3 -std=c99 -ffast-math -I $(CLVK_HEADERS) -I $(COMMON_DIR)

LIBS = -lm -lOpenCL -fopenmp -L $(CLVK_LIB) 

MMUL_OBJS = wtime.o
EXEC = mult


all: $(EXEC)

mult: $(MMUL_OBJS) matmul.c matrix_lib.c 
	$(CC) $^ $(CCFLAGS) $(LIBS) -o $(EXEC)

wtime.o: $(COMMON_DIR)/wtime.c
	$(CC) -c $^ $(CCFLAGS) -o $@

.c.o: 
	$(CC) -c $< $(CCFLAGS) -o $@


clean:
	rm -f $(MMUL_OBJS) $(EXEC)
