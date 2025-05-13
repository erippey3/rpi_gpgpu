#ifndef SPMV_OPENMP_H
#define SPMV_OPENMP_H
#include "benchmark.h"
#include "sparse_formats.h"


// if benchmark is null, no time will be taken


vector *openmp_csr_spmv(const csr_matrix *, const vector *, benchmark *b);
vector *openmp_coo_spmv(const coo_matrix *, const vector *, benchmark *b);


#endif // SPMV_OPENMP_H