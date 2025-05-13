#ifndef SPMV_SERIAL_H
#define SPMV_SERIAL_H
#include "benchmark.h"
#include "sparse_formats.h"


// if benchmark is null, no time will be taken


vector *serial_csr_spmv(const csr_matrix *, const vector *, benchmark *b);

vector *serial_coo_spmv(const coo_matrix *, const vector *, benchmark *b);


// foolish to think that you can hold any large matrix in standard form
vector *serial_std_spmv(const std_matrix *, const vector *, benchmark *b);


#endif