#ifndef WTIME_H
#define WTIME_H


#ifdef _OPENMP
#include <omp.h>
#else
#include <sys/time.h>
#endif

#include <stdlib.h>

double wtime();

#endif //WTIME_H