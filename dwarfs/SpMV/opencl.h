#ifndef SPMV_OPENCL_H
#define SPMV_OPENCL_H
#include "benchmark.h"
#include "sparse_formats.h"


// if benchmark is null, no time will be taken


vector *opencl_csr_spmv(const csr_matrix *, const vector *, benchmark *b);
vector *opencl_std_spmv(const std_matrix *, const vector *, benchmark *b);
vector *opencl_coo_spmv(const coo_matrix *, const vector *, benchmark *b);


#endif // SPMV_OPENCL_H