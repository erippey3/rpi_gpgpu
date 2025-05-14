#include "serial.h"
#include <stdlib.h>
#include "throughput.h"
#include <wtime.h>


/* returns dot product of v1 and v2*/
float serial_dot(vector v1, vector v2){
    if (v1.length != v2.length) {
        fprintf(stderr, "serial_dot, v1 and v2 do not have equal length");
        return -1;
    }
    float sum = 0;

    for (int i = 0; i < v1.length; i++){
        sum += v1.data[i] * v2.data[i];
    }

    return sum;
}

/* scales x by alpha and adds y in place*/
void serial_axpy(float alpha, vector x, vector y){
    if (x.length != y.length) {
        fprintf(stderr, "serial_dot, v1 and v2 do not have equal length");
        return;
    }

    for (int i = 0; i < x.length; i++)
        x.data[i] = alpha * x.data[i] + y.data[i];
}

/* scales x by alpha in place */
void serial_scal(float alpha, vector x){
    for (int i = 0; i < x.length; i++)
        x.data[i] = alpha * x.data[i];
}


void serial_throughput_test(benchmark *b){
    unsigned long seed1 = (unsigned long)(rand() * 100000);
    unsigned long seed2 = (unsigned long)(rand() * 100000);
    vector x = rand_vector(LENGTH, &seed1, stdout);
    vector y = rand_vector(LENGTH, &seed2, stdout);
    float alpha = 2.0f;

    double start, elapsed;
    float result;

    // Test DOT
    b->test_name = "DOT";
    start = wtime();
    for (int i = 0; i < NUM_TRIALS; i++) {
        result = serial_dot(x, y);
    }
    elapsed = wtime() - start;
    double dot_flops = LENGTH * 2.0 * NUM_TRIALS; // one multiply + one add per element
    printf("DOT: %.2f GFLOPS\n", (dot_flops / elapsed) / 1e9);

    // Test AXPY
    b->test_name = "AXPY";
    start = wtime();
    for (int i = 0; i < NUM_TRIALS; i++) {
        serial_axpy(alpha, x, y);
    }
    elapsed = wtime() - start;
    double axpy_flops = LENGTH * 2.0 * NUM_TRIALS;
    printf("AXPY: %.2f GFLOPS\n", (axpy_flops / elapsed) / 1e9);

    // Test SCAL
    b->test_name = "SCAL";
    start = wtime();
    for (int i = 0; i < NUM_TRIALS; i++) {
        serial_scal(alpha, x);
    }
    elapsed = wtime() - start;
    double scal_flops = LENGTH * 1.0 * NUM_TRIALS; // one multiply per element
    printf("SCAL: %.2f GFLOPS\n", (scal_flops / elapsed) / 1e9);

    free(x.data);
    free(y.data);
}