#ifndef C_COMMON_BENCHMARK_H
#define C_COMMON_BENCHMARK_H
#include <stdio.h>


typedef struct benchmark {
    int num_runs;           // Total number of recorded runs
    double *run_times;       // Dynamically allocated/reallocated
    char *device_name;      // Describes device (e.g., "RTX 3080", "Intel N100")
    char *test_name;        // Describes specific benchmark running
    
    double min_time;         // Best runtime
    double max_time;         // Worst runtime
    double total_time;       // Sum of all runtimes
    double stddev;           // Standard deviation
} benchmark;


benchmark init_benchmark(const char* device_name, const char* test_name);
void add_runtime(benchmark *b, double time);
double avg_runtime(const benchmark *b);
double compute_stddev(const benchmark *b);
void print_stats(benchmark *b, FILE* stream);
void free_benchmark(benchmark *b);



#endif //C_COMMON_BENCHMARK_H