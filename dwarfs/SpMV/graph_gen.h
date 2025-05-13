#ifndef SPMV_GRAPH_GEN_H
#define SPMV_GRAPH_GEN_H

#include "sparse_formats.h"
#include <stdio.h>

void generate_random_graphs(const char *matrix_path, FILE *log);

//void generate_example_graphs(const char *matrix_path, FILE *log);

#endif // SPMV_GRAPH_GEN_H
