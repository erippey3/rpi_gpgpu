#include "sparse_formats.h"
#include "serial.h"
#include "openmp.h"
#include "opencl.h"
#include "benchmark.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <dirent.h>
#include <sys/types.h>
#include <stdbool.h>
#include <device_info.h>
#include <err_code.h>


#define TEST_REPETITIONS 15
#define MAX_TEST_NAME_LEN 1024

char *graphs[] = {"matrices/barrier2-2/barrier2-2.mtx", "matrices/belgium_osm/belgium_osm.mtx", "matrices/ct20stif/ct20stif.mtx", "matrices/heart2/heart2.mtx", "matrices/mac_econ_fwd500/mac_econ_fwd500.mtx", "matrices/webbase-1M/webbase-1M.mtx"};




bool serial = false;
bool openmp = false;
bool opencl = false;

static void print_usage(const char *progname) {
    printf("Usage: %s [OPTIONS]\n", progname);
    printf("Options:\n");
    printf("  -s,  --serial       Use serial implementation\n");
    printf("  -mp, --openmp       Use OpenMP implementation\n");
    printf("  -cl, --opencl       Use OpenCL implementation\n");
    printf("  -h,  --help         Show this help message and exit\n");
}

static void parse_args(int argc, char *argv[])
{
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--serial") == 0) {
            serial = true;
        } else if (strcmp(argv[i], "-cl") == 0 || strcmp(argv[i], "--opencl") == 0) {
            opencl = true;
        } else if (strcmp(argv[i], "-mp") == 0 || strcmp(argv[i], "--openmp") == 0) {
            openmp = true;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            exit(0);
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            print_usage(argv[0]);
            exit(1);
        }
    }

    // Optionally print the selected modes
    printf("Serial: %s\n", serial ? "Enabled" : "Disabled");
    printf("OpenMP: %s\n", openmp ? "Enabled" : "Disabled");
    printf("OpenCL: %s\n", opencl ? "Enabled" : "Disabled");
}

int main(int argc, char *argv[]) {
    parse_args(argc, argv);

    // early exit if nothing else is needed
    if (!serial && !openmp && !opencl)
        return 0;

    if (opencl)
        init_cl(0, stdout);

    FILE *speedup_file = fopen("speedups.txt", "w");
    if (speedup_file == NULL) {
        fprintf(stderr, "Could not open speedup file\n");
        speedup_file = stdout;
    }

    

    for (int i = 0; i < sizeof(graphs) / sizeof(char *); i++){
        // load coordinate matrix from file as that is how .mtx are stored
        coo_matrix *coo = (coo_matrix *) malloc(sizeof(coo_matrix));
        *coo = load_matrix_market_to_coo(graphs[i], stdout);

        // get a vector proportionate to the size of the matrix
        vector * v = (vector *) malloc(sizeof(vector));
        unsigned long seed = 0xdeadbeef;
        *v = rand_vector(coo->num_cols, &seed, stdout);


        // define values for testing and benchmarking
        char test_name[MAX_TEST_NAME_LEN];
        vector * v1 = NULL;
        vector * v2;
        // two types of graph representations time 3 device types
        benchmark b[6];

        // test coo spmv serially
        if (serial) {
            snprintf(test_name, MAX_TEST_NAME_LEN, "serial coordinate SpMV: %s", graphs[i]);
            b[0] = init_benchmark(get_cpu_name(), test_name);

            // run the test multiple times
            for (int t = 0; t < TEST_REPETITIONS; t++){

                v2 = serial_coo_spmv(coo, v, &b[0]);

                // compare equality by transitive property
                if (v1) {
                    check(vector_is_equal(v1, v2, stdout), "serial_coo_spmv(): produce results that are not consistent\n");
                }
                v1 = v2;
            }

            print_stats(&b[0], stdout);
        }

        // test coo spmv multithreaded 
        if (openmp) {
            snprintf(test_name, MAX_TEST_NAME_LEN, "thread parallel coordinate SpMV: %s", graphs[i]);
            b[1] = init_benchmark(get_cpu_name(), test_name);

                        // run the test multiple times
            for (int t = 0; t < TEST_REPETITIONS; t++){

                v2 = openmp_coo_spmv(coo, v, &b[1]);

                // compare equality by transitive property
                if (v1) {
                    check(vector_is_equal(v1, v2, stdout), "openmp_coo_spmv(): produce results that are not consistent\n");
                }
                v1 = v2;
            }

            print_stats(&b[1], stdout);
        }

        // test coo spmv on opencl kernels
        if (opencl) {

        }


        // translate the coordinate matrix to Compressed Sparse Row Matrix
        csr_matrix *csr = (csr_matrix *) malloc(sizeof(csr_matrix));
        *csr = coo_to_csr(coo, stdout);
        free_coo(coo, 1);


        // test csr spmv serially
        if (serial) {
            snprintf(test_name, MAX_TEST_NAME_LEN, "serial CSR SpMV: %s", graphs[i]);
            b[3] = init_benchmark(get_cpu_name(), test_name);

            // run the test multiple times
            for (int t = 0; t < TEST_REPETITIONS; t++){

                v2 = serial_csr_spmv(csr, v, &b[3]);

                // compare equality by transitive property
                if (v1) {
                    check(vector_is_equal(v1, v2, stdout), "serial_csr_spmv(): produce results that are not consistent\n");
                }
                v1 = v2;
            }

            print_stats(&b[3], stdout);
        }


        // test csr spmv multithreaded 
        if (openmp) {
            snprintf(test_name, MAX_TEST_NAME_LEN, "thread parallel CSR SpMV: %s", graphs[i]);
            b[4] = init_benchmark(get_cpu_name(), test_name);

            // run the test multiple times
            for (int t = 0; t < TEST_REPETITIONS; t++){

                v2 = openmp_csr_spmv(csr, v, &b[4]);

                // compare equality by transitive property
                if (v1) {
                    check(vector_is_equal(v1, v2, stdout), "openmp_csr_spmv(): produce results that are not consistent\n");
                }
                v1 = v2;
            }

            print_stats(&b[4], stdout);
        }

        // test csr spmv on opencl kernels
        if (opencl) {
            snprintf(test_name, MAX_TEST_NAME_LEN, "GPU CSR SpMV: %s", graphs[i]);
            b[5] = init_benchmark(get_gpu_name(), test_name);

            // run the test multiple times
            for (int t = 0; t < TEST_REPETITIONS; t++){

                v2 = opencl_csr_spmv(csr, v, &b[5]);

                // compare equality by transitive property
                if (v1) {
                    check(vector_is_equal(v1, v2, stdout), "opencl_csr_spmv(): produce results that are not consistent\n");
                }
                v1 = v2;
            }

            print_stats(&b[5], stdout);
        }

        
        if (serial && openmp){
            print_speedup(b[0], b[1], speedup_file);
            print_speedup(b[3], b[4], speedup_file); //csr speedup
        }

        if (serial && opencl) {
            print_speedup(b[3], b[5], speedup_file); // csr speedup
        }

        if (openmp && opencl) {
            print_speedup(b[4], b[5], speedup_file); //csr speedup
        }



    }

    fclose(speedup_file);
    
    return 0;
}