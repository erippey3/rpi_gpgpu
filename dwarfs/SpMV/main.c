#include "sparse_formats.h"
#include "serial.h"
#include "openmp.h"
#include "opencl.h"
#include "benchmark.h"
#include "graph_gen.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <dirent.h>
#include <sys/types.h>
#include <stdbool.h>

#define TEST_REPETITIONS 15

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

    

    for (int i = 0; i < sizeof(graphs) / sizeof(char *); i++){
        coo_matrix *matrix = (coo_matrix *) malloc(sizeof(coo_matrix));

        *matrix = load_matrix_market_to_coo(graphs[i], stdout);

        if (serial) {
            
        }


        free_coo(matrix, 1);


    }


    return 0;
}