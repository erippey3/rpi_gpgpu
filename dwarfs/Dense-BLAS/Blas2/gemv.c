#include "sparse_formats.h"
#include "benchmark.h"
#include "device_info.h"
#include <stdio.h>
#include <omp.h>
#include <wtime.h>
#include <err_code.h>
#include <math.h>
#include <cl_utils.h>
#include <device_picker.h>



static vector serial_gemv(std_matrix m, vector v);
static vector mp_gemv(std_matrix m, vector v);
static vector cl_gemv(std_matrix m, vector v);
static void init_cl(int device_id, FILE *stream);
void shutdown_cl(FILE *stream);

unsigned int matrix_size = 10000;
unsigned int num_trials = 15;
cl_int err;
cl_device_id device;
cl_context context;
cl_command_queue commands;
cl_program program;

const char *program_files[] = {"gemv.cl"};



int main() {
    // 100 % filled matrix
    std_matrix m = rand_std_matrix(matrix_size, 1000000, stdout);
    unsigned long seed = rand() * 100000;
    vector v = rand_vector(matrix_size, &seed, stdout);
    double start_time, delta_time;


    benchmark b_serial = init_benchmark(get_cpu_name(), "Serial GEMV");
    benchmark b_parallel = init_benchmark(get_cpu_name(), "OpenMP GEMV");

    init_cl(0, stdout);


    for (int i = 0; i < num_trials; i++){
        start_time = wtime();
        vector ser_result = serial_gemv(m, v);
        delta_time = wtime() - start_time;
        add_runtime(&b_serial, delta_time);

        start_time = wtime();
        vector par_result = mp_gemv(m, v);
        delta_time = wtime() - start_time;
        add_runtime(&b_parallel, delta_time);

        check(vector_is_equal(&ser_result, &par_result, stdout), "gemv_main(): vectors are not equal");
        free(ser_result.data);
        free(par_result.data);
    }
    print_stats(&b_serial, stdout);
    print_stats(&b_parallel, stdout);


    shutdown_cl(stdout);
}


vector serial_gemv(std_matrix m, vector v){
    vector result;    
    if (m.num_cols != v.length)
    {
        fprintf(stderr, "serial_gemv matrix num_cols != vector length\n");
        result;
    }



    result.length = m.num_rows;
    result.data = (float *) malloc(sizeof(float) * result.length);

    
    for (int i = 0; i < m.num_rows; i++){
        float sum = 0;
        for (int j = 0; j < m.num_cols; j++){
            sum += v.data[j] * m.matrix[i * m.num_cols + j];
        }
        v.data[i] = sum;
    }

    return result;
}

vector mp_gemv(std_matrix m, vector v){
    vector result;    
    if (m.num_cols != v.length)
    {
        fprintf(stderr, "serial_gemv matrix num_cols != vector length\n");
        result;
    }



    result.length = m.num_rows;
    result.data = (float *) malloc(sizeof(float) * result.length);


    // static as this is a evenly distributed work equation
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m.num_rows; i++){
        float sum = 0;
        for (int j = 0; j < m.num_cols; j++){
            sum += v.data[j] * m.matrix[i * m.num_cols + j];
        }
        v.data[i] = sum;
    }

    return result;
}








// initialized cl device, context, commands, and program.
void init_cl(int device_id, FILE *stream){
    cl_device_id devices[MAX_DEVICES];
    unsigned numDevices = getDeviceList(devices);
    char name[MAX_INFO_STRING];
    int deviceIndex = device_id;

    while (deviceIndex >= numDevices || deviceIndex < 0)
    {
        fprintf(stream,  "\nplease pick a valid device\n");
        for (int i = 0; i < numDevices; i++){
            device = devices[i];
            getDeviceName(device, name);
            fprintf(stream, "Device %d: %s\n", i, name);
        }

        scanf("%d", &deviceIndex);
    }

    device = devices[deviceIndex];

    getDeviceName(device, name);
    printf("\nUsing OpenCL device: %s\n", name);

    context = clCreateContext(0, 1, &device, NULL, NULL, &err);
    checkError(err, "Creating context");

    commands = clCreateCommandQueue(context, device, 0, &err);
    checkError(err, "Creating command queue");

    program = build_program_from_files(context, &device, 1, program_files, 1, NULL);

}


void shutdown_cl(FILE *stream){
    clReleaseProgram(program);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);
}