#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "benchmark.h"

benchmark init_benchmark(const char* device_name, const char* test_name) {
    benchmark b;
    b.num_runs = 0;
    b.run_times = NULL;
    b.device_name = (char *) malloc(strlen(device_name) + 1);
    snprintf(b.device_name, strlen(device_name) + 1, "%s", device_name);
    b.test_name = (char *) malloc(strlen(test_name) + 1);
    snprintf(b.test_name, strlen(test_name) + 1, "%s", test_name);
    b.min_time = INFINITY;
    b.max_time = -INFINITY;
    b.total_time = 0.0;
    b.stddev = 0.0;
    return b;
}

void add_runtime(benchmark *b, double time) {
    b->run_times = realloc(b->run_times, (b->num_runs + 1) * sizeof(double));
    if (!b->run_times) {
        fprintf(stderr, "Memory allocation failed in add_runtime\n");
        exit(EXIT_FAILURE);
    }
    b->run_times[b->num_runs] = time;
    b->num_runs++;
    b->total_time += time;

    if (time < b->min_time) b->min_time = time;
    if (time > b->max_time) b->max_time = time;
}

double avg_runtime(const benchmark *b) {
    return (b->num_runs == 0) ? 0.0 : b->total_time / b->num_runs;
}

double compute_stddev(const benchmark *b) {
    if (b->num_runs <= 1) return 0.0;
    double mean = avg_runtime(b);
    double sum_sq_diff = 0.0;
    for (int i = 0; i < b->num_runs; ++i) {
        double diff = b->run_times[i] - mean;
        sum_sq_diff += diff * diff;
    }
    return sqrt(sum_sq_diff / (b->num_runs - 1));
}

void print_stats(benchmark *b, FILE* stream) {
    b->stddev = compute_stddev(b);
    fprintf(stream, "Benchmark %s running on device %s\n", b->test_name, b->device_name);
    fprintf(stream, "  Runs      : %d\n", b->num_runs);
    fprintf(stream, "  Avg time  : %.6f s\n", avg_runtime(b));
    fprintf(stream, "  Min time  : %.6f s\n", b->min_time);
    fprintf(stream, "  Max time  : %.6f s\n", b->max_time);
    fprintf(stream, "  Std Dev   : %.6f s\n\n", b->stddev);
}

void free_benchmark(benchmark *b) {
    free(b->run_times);
    free(b->device_name);
    b->run_times = NULL;
    b->device_name = NULL;
    b->num_runs = 0;
}
