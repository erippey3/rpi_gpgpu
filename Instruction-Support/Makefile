CLVK_HEADERS = /opt/clvk/external/OpenCL-Headers/
CLVK_LIB = /opt/clvk/build/
COMMON_DIR = ../C_common

ifndef CC
	CC = gcc
endif

CCFLAGS=-O3 -g3 -std=c99 -ffast-math -I $(CLVK_HEADERS) -I $(COMMON_DIR)

LIBS = -lm -lOpenCL -fopenmp -L $(CLVK_LIB) 

TIME_OBJ = wtime.o



instructions: $(TIME_OBJ) instructions.c $(COMMON_DIR)/cl_utils.c
	$(CC) $^ $(CCFLAGS) $(LIBS) -I $(COMMON_DIR) -o $@

wtime.o: $(COMMON_DIR)/wtime.c
	$(CC) -c $^ $(CCFLAGS) -o $@

clean:
	rm -f instrucitons
	rm -f wtime.o
