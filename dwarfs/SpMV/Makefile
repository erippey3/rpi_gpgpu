CLVK_HEADERS = /opt/clvk/external/OpenCL-Headers/
CLVK_LIB = /opt/clvk/build/
COMMON_DIR = ../../C_common
OBJ_DIR = obj

ifndef CC
	CC = gcc
endif

CCFLAGS = -O3 -g3 -ffast-math -fopenmp -I $(CLVK_HEADERS) -I $(COMMON_DIR)
LDFLAGS = -lm -lOpenCL -fopenmp -L$(CLVK_LIB)

# Object files with path
OBJS = \
  $(OBJ_DIR)/main.o \
  $(OBJ_DIR)/opencl.o \
  $(OBJ_DIR)/openmp.o \
  $(OBJ_DIR)/serial.o \
  $(OBJ_DIR)/sparse_formats.o \
  $(OBJ_DIR)/ziggurat.o \
  $(OBJ_DIR)/cl_utils.o \
  $(OBJ_DIR)/wtime.o \
  $(OBJ_DIR)/err_code.o \
  $(OBJ_DIR)/benchmark.o \
  $(OBJ_DIR)/graph_gen.o \
  $(OBJ_DIR)/device_info.o

all: $(OBJ_DIR) spmv

# Create binary
spmv: $(OBJS)
	$(CC) $^ $(CCFLAGS) $(LDFLAGS) -o $@

# Compile .c to obj/.o
$(OBJ_DIR)/%.o: %.c
	$(CC) -c $< $(CCFLAGS) -o $@

$(OBJ_DIR)/err_code.o: $(COMMON_DIR)/err_code.c $(COMMON_DIR)/err_code.h
	$(CC) -c $< $(CCFLAGS) -o $@

$(OBJ_DIR)/cl_utils.o: $(COMMON_DIR)/cl_utils.c $(COMMON_DIR)/cl_utils.h
	$(CC) -c $< $(CCFLAGS) -o $@

$(OBJ_DIR)/benchmark.o: $(COMMON_DIR)/benchmark.c $(COMMON_DIR)/benchmark.h
	$(CC) -c $< $(CCFLAGS) -o $@

$(OBJ_DIR)/wtime.o: $(COMMON_DIR)/wtime.c $(COMMON_DIR)/wtime.h
	$(CC) -c $< $(CCFLAGS) -o $@

$(OBJ_DIR)/device_info.o: $(COMMON_DIR)/device_info.c $(COMMON_DIR)/device_info.h
	$(CC) -c $< $(CCFLAGS) -o $@

# Create obj/ if it doesn't exist
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

.PHONY: all clean
clean:
	rm -rf spmv $(OBJ_DIR)
