#include "sparse_formats.h"
#include "serial.h"
#include "benchmark.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <dirent.h>
#include <sys/types.h>
#include <omp.h>

#define MATRIX_PATH "./matrices"

static int is_dir_empty(const char *path) {
    DIR *dir = opendir(path);
    if (!dir) return 1; // Treat as empty if cannot open
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0) {
            closedir(dir);
            return 0; // Not empty
        }
    }
    closedir(dir);
    return 1;
}

static int ends_with_csr(const char *filename) {
    size_t len = strlen(filename);
    return len >= 4 && strcmp(filename + len - 4, ".csr") == 0;
}




int main() {
    unsigned long seeds[] = {0xdeadbeef, 0x00c0ffee, 0xbad0c0de, 0xd15ea5e0, 0xcafebabe, 0xbaadf00d}; 
    int matrix_sizes[] = {10000, 50000, 100000, 500000, 1000000};
    int densities[] = {10000, 50000, 100000, 250000, 500000};
    float std_devs[] = {0.0, 0.1, 0.25, 0.5, 1.0};

    int num_seeds = sizeof(seeds) / sizeof(seeds[0]);
    int num_sizes = sizeof(matrix_sizes) / sizeof(matrix_sizes[0]);
    int num_densities = sizeof(densities) / sizeof(densities[0]);
    int num_devs = sizeof(std_devs) / sizeof(std_devs[0]);

    if (!is_dir_empty(MATRIX_PATH)) {
        printf("Directory not empty. Assuming matrices already generated. Exiting.\n");
        return 0;
    }

    printf("Generating matrices...\n");

    int nthreads = omp_get_max_threads()/4;
    omp_set_num_threads(nthreads); 
    FILE *log = fopen("/dev/null", "w");


    #pragma omp parallel for collapse(4) schedule(dynamic)
    for (int j = 0; j < num_sizes; j++) {
        for (int k = 0; k < num_densities; k++) {
            for (int i = 0; i < num_seeds; i++) {
                for (int l = 0; l < num_devs; l++) {
                    // Avoid collisions in printf by doing I/O in critical section
                    char file_path[512];
                    snprintf(file_path, sizeof(file_path), 
                        MATRIX_PATH "/size%d_d%d_dev%.2f_seed%lx.csr", 
                        matrix_sizes[j], densities[k], std_devs[l], seeds[i]);

                    // Generate CSR and write to disk
                    unsigned long local_seed = seeds[i];
                    csr_matrix *csr = (csr_matrix *) malloc(sizeof(csr_matrix));
                    *csr = rand_csr(matrix_sizes[j], densities[k], std_devs[l], &local_seed, log);
                    
                    #pragma omp critical
                        write_csr(csr, 1, file_path);

                    printf("[Thread %d] Wrote %s\n", omp_get_thread_num(), file_path);


                    free_csr(csr, 1); // Free right after use
                }
            }
        }
    }

    printf("Matrix generation complete.\n");

    return 0;
}