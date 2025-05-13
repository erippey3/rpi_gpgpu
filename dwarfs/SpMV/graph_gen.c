#include "graph_gen.h"
#include "sparse_formats.h"
#include <omp.h>
#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdlib.h>

static int file_exists(const char *filename) {
    return access(filename, F_OK) == 0;
}

void generate_missing_graphs(const char *matrix_path, FILE* log) {
    unsigned long seeds[] = {0xdeadbeef, 0x00c0ffee, 0xbad0c0de, 0xd15ea5e0, 0xcafebabe, 0xbaadf00d}; 
    int matrix_sizes[] = {10000, 50000, 100000, 500000, 1000000};
    int densities[] = {10000, 50000, 100000, 250000, 500000};
    float std_devs[] = {0.0, 0.1, 0.25, 0.5, 1.0};

    int num_seeds = sizeof(seeds) / sizeof(seeds[0]);
    int num_sizes = sizeof(matrix_sizes) / sizeof(matrix_sizes[0]);
    int num_densities = sizeof(densities) / sizeof(densities[0]);
    int num_devs = sizeof(std_devs) / sizeof(std_devs[0]);

    mkdir(matrix_path, 0755);

    #pragma omp parallel for collapse(4) schedule(dynamic)
    for (int j = 0; j < num_sizes; j++) {
        for (int k = 0; k < num_densities; k++) {
            for (int i = 0; i < num_seeds; i++) {
                for (int l = 0; l < num_devs; l++) {
                    char file_path[512];
                    snprintf(file_path, sizeof(file_path), 
                        "%s/size%d_d%d_dev%.2f_seed%lx.csr", 
                        matrix_path, matrix_sizes[j], densities[k], std_devs[l], seeds[i]);

                    if (!file_exists(file_path)) {
                        unsigned long local_seed = seeds[i];
                        csr_matrix *csr = (csr_matrix*) malloc(sizeof(csr_matrix));
                        *csr = rand_csr(matrix_sizes[j], densities[k], std_devs[l], &local_seed, log);
                        write_csr(csr, 1, file_path);
                        fprintf(log, "[Thread %d] Generated %s\n", omp_get_thread_num(), file_path);
                        free_csr(csr, 1);
                    }
                }
            }
        }
    }

    fprintf(log, "Missing matrix generation complete.\n");
}



