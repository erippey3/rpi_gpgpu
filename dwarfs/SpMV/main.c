#include "sparse_formats.h"
#include "serial.h"
#include <stdlib.h>
#include <stdio.h>
#include <benchmark.h>


int main() {
    long *seeds = {0xdeadbeef, 0x00c0ffee, 0xbad0c0de, 0xd15ea5e0, 0xcafebabe, 0xbaadf00d};
    int *matrix_sizes = {10000, 50000, 100000, 500000, 1000000}; // 10,000 - 1,000,000
    int *densities = {10000, 50000, 100000, 250000, 500000} // 1%, 5%, 10%, 25%, 50%
    float *std_devs = {0.0, 0.1, 0.25, 0.5, 1.0} // from no to low to high deviation


    csr_matrix *csr = malloc(sizeof(csr_matrix));

    *csr = rand_csr(20000, 250000, 0.25, &seed, stdout);
    //print_csr_std(csr, stdout);
    vector *v = malloc(sizeof(vector));
    *v = rand_vector(20000, &seed, stdout);
    //print_vector(v, stdout);


    benchmark b = init_benchmark("Serial Cortex-A76", "CSR SpMV");
    for (int i = 0; i < 15; i++)
        serial_csr_spmv(csr, v, &b);

    print_stats(&b, stdout);
    



    free_csr(csr, 1);
    return 0;
}