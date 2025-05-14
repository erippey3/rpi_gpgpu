#include "serial.h"
#include "openmp.h"
#include "opencl.h"
#include "benchmark.h"
#include <device_info.h>
#include <stdio.h>


#define MAX_TEST_NAME_LEN 1024

int main(){
    char test_name[MAX_TEST_NAME_LEN];

    snprintf(test_name, MAX_TEST_NAME_LEN, "serial blas1 throughput test");
    benchmark b = init_benchmark(get_cpu_name(), test_name);
    serial_throughput_test(&b);
    openmp_throughput_test(&b);

    init_cl(0, stdout);
    opencl_throughput_test(&b);



    return 0;
}