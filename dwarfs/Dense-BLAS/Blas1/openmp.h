#ifndef DENSE_BLAS_OPENMP_H
#define DENSE_BLAS_OPENMP_H
#include "sparse_formats.h"
#include <benchmark.h>


/* returns dot product of v1 and v2*/
float openmp_dot(vector v1, vector v2);

/* scales x by alpha and adds y in place*/
void openmp_axpy(float alpha, vector x, vector y);

/* scales x by alpha in place */
void openmp_scal(float alpha, vector x);


void openmp_throughput_test(benchmark *b);



#endif //DENSE_BLAS_OPENMP_H