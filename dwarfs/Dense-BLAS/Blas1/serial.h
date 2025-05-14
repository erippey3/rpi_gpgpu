#ifndef DENSE_BLAS_SERIAL_H
#define DENSE_BLAS_SERIAL_H
#include "sparse_formats.h"
#include <benchmark.h>

/* returns dot product of v1 and v2*/
float serial_dot(vector v1, vector v2);

/* scales x by alpha and adds y in place*/
void serial_axpy(float alpha, vector x, vector y);

/* scales x by alpha in place */
void serial_scal(float alpha, vector x);

void serial_throughput_test(benchmark *b);






#endif //DENSE_BLAS_SERIAL_H