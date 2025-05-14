#ifndef DENSE_BLAS_OPENCL_H
#define DENSE_BLAS_OPENCL_H
#include "sparse_formats.h"
#include <benchmark.h>



/* returns dot product of v1 and v2

All sources are saying that dot product is simple enough 
and requires so reduction so is simply not worth moving to the 
GPU
*/
float opencl_dot(vector v1, vector v2);

/* scales x by alpha and adds y in place*/
void opencl_axpy(float alpha, vector x, vector y);

/* scales x by alpha in place */
void opencl_scal(float alpha, vector x);


void opencl_throughput_test(benchmark *b);


/*setup and teardown functions for opencl*/
void init_cl(int device_id, FILE* stream);
void shutdown_cl(FILE *stream);


#endif //DENSE_BLAS_OPENCL_H