#ifndef SPMV_OPENCL_H
#define SPMV_OPENCL_H
#include "benchmark.h"
#include "sparse_formats.h"
#include <stdio.h>


// if benchmark is null, no time will be taken

void init_cl(int device_id, FILE* stream);
vector *opencl_csr_spmv(const csr_matrix *, const vector *, benchmark *b);
vector *opencl_coo_spmv(const coo_matrix *, const vector *, benchmark *b);
void shutdown_cl(FILE *stream);

#endif // SPMV_OPENCL_H