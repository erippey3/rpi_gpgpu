#include "cl_utils.h"
#include "err_code.h"
#include "device_picker.h"
#include "wtime.h"
#include <stdbool.h>
#include <time.h>
#include <assert.h>
#include <math.h>

static int size = 2<<22;
static cl_int err;
static float tolerance = 0.001;

// Global state
cl_device_id device;
cl_context context;
cl_command_queue commands;
cl_program program;

cl_device_id devices[MAX_DEVICES];
cl_context contexts[MAX_DEVICES];
cl_command_queue commands_lst[MAX_DEVICES];
cl_program programs[MAX_DEVICES];
char device_names[MAX_DEVICES][MAX_INFO_STRING];
unsigned num_devices;



// Fill `buf` with `num_bytes` random values
static void fill_random_bytes(uint8_t *buf, size_t num_bytes, bool non_zero) {
    if (!buf || num_bytes == 0) return;

    for (size_t i = 0; i < num_bytes; ++i) {
        uint8_t val;
        do {
            val = (uint8_t)(rand() % 256);
        } while (non_zero && val == 0);
        buf[i] = val;
    }
}

// Fills `arr` with `num_floats` random float values between `low` and `high`
void fill_random_floats(float *arr, int num_floats, float low, float high) {
    if (!arr || num_floats <= 0 || high <= low) return;

    for (int i = 0; i < num_floats; ++i) {
        float scale = rand() / (float) RAND_MAX; // value between 0.0 and 1.0
        arr[i] = low + scale * (high - low);
    }
}


// dumps the binary of a memory location 
static void hexdump(uint8_t *buff, size_t num_bytes) {
    printf("\n");
    for (int i = 0; i < num_bytes; i++) {
        printf("%x", buff);
    }
    printf("\n");
}

// sets global state to point to a specific OpenCL device
static void set_active_device(int index) {
    device = devices[index];
    context = contexts[index];
    commands = commands_lst[index];
    program = programs[index];
}


/**
This method needs revisiting, bools in CL are 32 bit whereas 8 bit in C 

This method causes V3D to hang and prevents further programs from using the 
V3D. 
*/
static void test_bool_support()
{
    bool *h_a;
    bool *h_b;
    bool *h_c;
    bool *cpu_check;
    double start_time;
    double run_time;

    cl_mem d_a, d_b, d_c; // matricies in device memory

    printf("----------------------------------------------");
    printf("Testing bool support on all CL devices");
    printf("----------------------------------------------");

    h_a = (bool *)malloc(size * sizeof(bool));
    h_b = (bool *)malloc(size * sizeof(bool));
    h_c = (bool *)malloc(size * sizeof(bool));
    cpu_check = (bool *)malloc(size * sizeof(bool));

    fill_random_bytes(h_a, size*sizeof(bool), false);
    fill_random_bytes(h_b, size*sizeof(bool), false);



    printf("\nChecking serial cpu time\n");
    start_time = wtime();
    for (int i = 0; i < size; i++){
        cpu_check[i] = h_a[i] | h_b[i];
    }

    run_time = wtime() - start_time;
    printf("Serial Implementation took %f\n", run_time);

    for (int i = 1; i < num_devices; i++){
        printf("\nChecking time for device %s\n", device_names[i]);
        set_active_device(i);

        // Create kernel for checking booleans
        cl_kernel kernel = clCreateKernel(program, "bool_support", &err);
        checkError(err, "Creating kernel");

        d_a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(bool) * size, h_a, &err);
        checkError(err, "Creating buffer d_a");
        
        d_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(bool) * size, h_b, &err);
        checkError(err, "Creating buffer d_b");

        d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
            sizeof(bool) * size, NULL, &err);
        checkError(err, "Creating buffer d_c");

        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b);
        err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_c);
        checkError(err, "Setting kernel arguments");

        start_time = wtime();

        const size_t global[1] = {size};
        err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, global, NULL, 0, NULL, NULL); 
        checkError(err, "Enqueuing kernel");


        err = clFinish(commands);
        checkError(err, "Waiting for commands to finish");

        run_time = wtime() - start_time;
        printf("Implementation on %s took %f\n", device_names[i], run_time);

        err = clEnqueueReadBuffer(
            commands, d_c, CL_TRUE, 0,
            sizeof(bool) * size, h_c,
            0, NULL, NULL);
        checkError(err, "Reading back buffer d_c");

        if (size < 64){
            hexdump(h_c, sizeof(bool) * size);
            hexdump(cpu_check, sizeof(bool) * size);
        }

        for (int i = 0; i < size; i++) {
            assert(h_c[i] == cpu_check[i]);
        }

        clReleaseMemObject(d_a);
        clReleaseMemObject(d_b);
        clReleaseMemObject(d_c);
        clReleaseKernel(kernel);
    }

    free(h_a);
    free(h_b);
    free(h_c);
    free(cpu_check);
}

/**
This method causes V3D to hang and prevents further programs from using the 
V3D. 
*/
static void test_char_support()
{
    char *h_a;
    char *h_b;
    char *h_c;
    char *cpu_check;
    double start_time;
    double run_time;

    cl_mem d_a, d_b, d_c; // matricies in device memory

    printf("----------------------------------------------");
    printf("Testing char support on all CL devices");
    printf("----------------------------------------------");

    h_a = (char *)malloc(size * sizeof(char));
    h_b = (char *)malloc(size * sizeof(char));
    h_c = (char *)malloc(size * sizeof(char));
    cpu_check = (char *)malloc(size * sizeof(char));

    fill_random_bytes(h_a, size*sizeof(char), false);
    fill_random_bytes(h_b, size*sizeof(char), false);



    printf("\nChecking serial cpu time\n");
    start_time = wtime();
    for (int i = 0; i < size; i++){
        cpu_check[i] = h_a[i] + h_b[i];
    }

    run_time = wtime() - start_time;
    printf("Serial Implementation took %f\n", run_time);

    for (int i = 0; i < num_devices; i++){
        printf("\nChecking time for device %s\n", device_names[i]);
        set_active_device(i);

        // Create kernel for checking chars
        cl_kernel kernel = clCreateKernel(program, "char_support", &err);
        checkError(err, "Creating kernel");

        d_a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(char) * size, h_a, &err);
        checkError(err, "Creating buffer d_a");
        
        d_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(char) * size, h_b, &err);
        checkError(err, "Creating buffer d_b");

        d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
            sizeof(char) * size, NULL, &err);
        checkError(err, "Creating buffer d_c");

        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b);
        err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_c);
        checkError(err, "Setting kernel arguments");

        start_time = wtime();

        const size_t global[1] = {size};
        err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, global, NULL, 0, NULL, NULL); 
        checkError(err, "Enqueuing kernel");


        err = clFinish(commands);
        checkError(err, "Waiting for commands to finish");

        run_time = wtime() - start_time;
        printf("Implementation on %s took %f\n", device_names[i], run_time);

        err = clEnqueueReadBuffer(
            commands, d_c, CL_TRUE, 0,
            sizeof(char) * size, h_c,
            0, NULL, NULL);
        checkError(err, "Reading back buffer d_c");

        if (size < 64){
            hexdump(h_c, sizeof(char) * size);
            hexdump(cpu_check, sizeof(char) * size);
        }

        for (int i = 0; i < size; i++) {
            assert(h_c[i] == cpu_check[i]);
        }

        clReleaseMemObject(d_a);
        clReleaseMemObject(d_b);
        clReleaseMemObject(d_c);
        clReleaseKernel(kernel);
    }

    free(h_a);
    free(h_b);
    free(h_c);
    free(cpu_check);
}

/**
This method does not cause V3D to hang
*/
static void test_int_support()
{
    int *h_a;
    int *h_b;
    int *h_c;
    int *cpu_check;
    double start_time;
    double run_time;

    cl_mem d_a, d_b, d_c; // matricies in device memory

    printf("----------------------------------------------");
    printf("Testing int support on all CL devices");
    printf("----------------------------------------------");

    h_a = (int *)malloc(size * sizeof(int));
    h_b = (int *)malloc(size * sizeof(int));
    h_c = (int *)malloc(size * sizeof(int));
    cpu_check = (int *)malloc(size * sizeof(int));

    fill_random_bytes(h_a, size*sizeof(int), false);
    fill_random_bytes(h_b, size*sizeof(int), false);



    printf("\nChecking serial cpu time\n");
    start_time = wtime();
    for (int i = 0; i < size; i++){
        cpu_check[i] = h_a[i] + h_b[i];
    }

    run_time = wtime() - start_time;
    printf("Serial Implementation took %f\n", run_time);

    for (int i = 0; i < num_devices; i++){
        printf("\nChecking time for device %s\n", device_names[i]);
        set_active_device(i);

        // Create kernel for checking ints
        cl_kernel kernel = clCreateKernel(program, "int_support", &err);
        checkError(err, "Creating kernel");

        d_a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(int) * size, h_a, &err);
        checkError(err, "Creating buffer d_a");
        
        d_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(int) * size, h_b, &err);
        checkError(err, "Creating buffer d_b");

        d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
            sizeof(int) * size, NULL, &err);
        checkError(err, "Creating buffer d_c");

        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b);
        err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_c);
        checkError(err, "Setting kernel arguments");

        start_time = wtime();

        const size_t global[1] = {size};
        const size_t local[1] = {256};
        err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, global, local, 0, NULL, NULL); 
        checkError(err, "Enqueuing kernel");


        err = clFinish(commands);
        checkError(err, "Waiting for commands to finish");

        run_time = wtime() - start_time;
        printf("Implementation on %s took %f\n", device_names[i], run_time);

        err = clEnqueueReadBuffer(
            commands, d_c, CL_TRUE, 0,
            sizeof(int) * size, h_c,
            0, NULL, NULL);
        checkError(err, "Reading back buffer d_c");

        if (size < 64){
            hexdump(h_c, sizeof(int) * size);
            hexdump(cpu_check, sizeof(int) * size);
        }

        for (int i = 0; i < size; i++) {
            assert(h_c[i] == cpu_check[i]);
        }

        clReleaseMemObject(d_a);
        clReleaseMemObject(d_b);
        clReleaseMemObject(d_c);
        clReleaseKernel(kernel);
    }

    free(h_a);
    free(h_b);
    free(h_c);
    free(cpu_check);
}

/**
This method does not cause V3D to hang
*/
static void test_float_support()
{
    float *h_a;
    float *h_b;
    float *h_c;
    float *cpu_check;
    double start_time;
    double run_time;

    cl_mem d_a, d_b, d_c; // matricies in device memory

    printf("----------------------------------------------");
    printf("Testing float support on all CL devices");
    printf("----------------------------------------------");

    h_a = (float *)malloc(size * sizeof(float));
    h_b = (float *)malloc(size * sizeof(float));
    h_c = (float *)malloc(size * sizeof(float));
    cpu_check = (float *)malloc(size * sizeof(float));

    fill_random_bytes(h_a, size*sizeof(float), false);
    fill_random_bytes(h_b, size*sizeof(float), false);



    printf("\nChecking serial cpu time\n");
    start_time = wtime();
    for (int i = 0; i < size; i++){
        cpu_check[i] = h_a[i] + h_b[i];
    }

    run_time = wtime() - start_time;
    printf("Serial Implementation took %f\n", run_time);

    for (int i = 0; i < num_devices; i++){
        printf("\nChecking time for device %s\n", device_names[i]);
        set_active_device(i);

        // Create kernel for checking floats
        cl_kernel kernel = clCreateKernel(program, "float_support", &err);
        checkError(err, "Creating kernel");

        d_a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(float) * size, h_a, &err);
        checkError(err, "Creating buffer d_a");
        
        d_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(float) * size, h_b, &err);
        checkError(err, "Creating buffer d_b");

        d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
            sizeof(float) * size, NULL, &err);
        checkError(err, "Creating buffer d_c");

        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b);
        err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_c);
        checkError(err, "Setting kernel arguments");

        start_time = wtime();

        const size_t global[1] = {size};
        err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, global, NULL, 0, NULL, NULL); 
        checkError(err, "Enqueuing kernel");


        err = clFinish(commands);
        checkError(err, "Waiting for commands to finish");

        run_time = wtime() - start_time;
        printf("Implementation on %s took %f\n", device_names[i], run_time);

        err = clEnqueueReadBuffer(
            commands, d_c, CL_TRUE, 0,
            sizeof(float) * size, h_c,
            0, NULL, NULL);
        checkError(err, "Reading back buffer d_c");

        if (size < 64){
            hexdump(h_c, sizeof(float) * size);
            hexdump(cpu_check, sizeof(float) * size);
        }

        for (int i = 0; i < size; i++) {
            assert((h_c[i] - cpu_check[i]) * (h_c[i] - cpu_check[i]) < tolerance);
        }

        clReleaseMemObject(d_a);
        clReleaseMemObject(d_b);
        clReleaseMemObject(d_c);
        clReleaseKernel(kernel);
    }

    free(h_a);
    free(h_b);
    free(h_c);
    free(cpu_check);
}

/**
Opencl size_t types are unsigned integers
*/
static void test_size_t_support()
{
    uint32_t*h_a;
    uint32_t*h_b;
    uint32_t*h_c;
    uint32_t*cpu_check;
    double start_time;
    double run_time;

    cl_mem d_a, d_b, d_c; // matricies in device memory

    printf("----------------------------------------------");
    printf("Testing size_t support on all CL devices");
    printf("----------------------------------------------");

    h_a = (uint32_t*)malloc(size * sizeof(uint32_t));
    h_b = (uint32_t*)malloc(size * sizeof(uint32_t));
    h_c = (uint32_t*)malloc(size * sizeof(uint32_t));
    cpu_check = (uint32_t*)malloc(size * sizeof(uint32_t));

    fill_random_bytes(h_a, size*sizeof(uint32_t), false);
    fill_random_bytes(h_b, size*sizeof(uint32_t), false);



    printf("\nChecking serial cpu time\n");
    start_time = wtime();
    for (int i = 0; i < size; i++){
        cpu_check[i] = h_a[i] + h_b[i];
    }

    run_time = wtime() - start_time;
    printf("Serial Implementation took %f\n", run_time);

    for (int i = 0; i < num_devices; i++){
        printf("\nChecking time for device %s\n", device_names[i]);
        set_active_device(i);

        // Create kernel for checking size_t
        cl_kernel kernel = clCreateKernel(program, "size_t_support", &err);
        checkError(err, "Creating kernel");

        d_a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(uint32_t) * size, h_a, &err);
        checkError(err, "Creating buffer d_a");
        
        d_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(uint32_t) * size, h_b, &err);
        checkError(err, "Creating buffer d_b");

        d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
            sizeof(uint32_t) * size, NULL, &err);
        checkError(err, "Creating buffer d_c");

        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b);
        err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_c);
        checkError(err, "Setting kernel arguments");

        start_time = wtime();

        const size_t global[1] = {size};
        err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, global, NULL, 0, NULL, NULL); 
        checkError(err, "Enqueuing kernel");


        err = clFinish(commands);
        checkError(err, "Waiting for commands to finish");

        run_time = wtime() - start_time;
        printf("Implementation on %s took %f\n", device_names[i], run_time);

        err = clEnqueueReadBuffer(
            commands, d_c, CL_TRUE, 0,
            sizeof(uint32_t) * size, h_c,
            0, NULL, NULL);
        checkError(err, "Reading back buffer d_c");

        if (size < 64){
            hexdump(h_c, sizeof(uint32_t) * size);
            hexdump(cpu_check, sizeof(uint32_t) * size);
        }

        for (int i = 0; i < size; i++) {
            assert(h_c[i] == cpu_check[i]);
        }

        clReleaseMemObject(d_a);
        clReleaseMemObject(d_b);
        clReleaseMemObject(d_c);
        clReleaseKernel(kernel);
    }

    free(h_a);
    free(h_b);
    free(h_c);
    free(cpu_check);
}

static void test_operation_support()
{

    float *h_f_a;
    float *h_f_b;
    float *h_f_c;
    float *cpu_f_check;
    int *h_i_a;
    int *h_i_b;
    int *h_i_c;
    int *cpu_i_check;
    double start_time;
    double run_time;

    cl_mem d_f_a, d_f_b, d_f_c, d_i_a, d_i_b, d_i_c; // device, int/float, array a/b/c

    printf("----------------------------------------------");
    printf("Testing operation support on all CL devices");
    printf("----------------------------------------------");


    h_f_a = (float *)malloc(size * sizeof(float)); //  standing for host - float, array a
    h_f_b = (float *)malloc(size * sizeof(float));
    h_f_c = (float *)malloc(size * sizeof(float));

    h_i_a = (int *)malloc(size * sizeof(int)); //  standing for host - int, array a
    h_i_b = (int *)malloc(size * sizeof(int));
    h_i_c = (int *)malloc(size * sizeof(int));

    // randomly fill all of these bytes with any uint8 value from 1 - 255
    fill_random_bytes(h_f_a, size * sizeof(float), true);
    fill_random_bytes(h_f_b, size * sizeof(float), true);
    fill_random_bytes(h_i_a, size * sizeof(int), true);
    fill_random_bytes(h_i_b, size * sizeof(int), true);

    // This for loop runs and compares the basic operations on all OpenCL devices
    for (int i = 0; i < num_devices; i++){
        printf("\nChecking support for basic aritmetic ops (+ - * / %%) for device %s\n", device_names[i]);
        set_active_device(i);

        cl_kernel kernel = clCreateKernel(program, "arithmetic_support", &err);
        checkError(err, "Creating kernel");

        d_f_a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * size, h_f_a, &err);
        checkError(err, "Creating buffer d_f_a");
        d_f_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * size, h_f_b, &err);
        checkError(err, "Creating buffer d_f_b");
        d_f_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * size, NULL, &err);
        checkError(err, "Creating buffer d_f_c");
        d_i_a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * size, h_i_a, &err);
        checkError(err, "Creating buffer d_i_a");
        d_i_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * size, h_i_b, &err);
        checkError(err, "Creating buffer d_i_b");
        d_i_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * size, NULL, &err);
        checkError(err, "Creating buffer d_i_c");

        // setting the arguments for the kernel
        err =  clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_f_a);
        err |=  clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_f_b);
        err |=  clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_f_c);
        err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_i_a);
        err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_i_b);
        err |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &d_i_c);
        checkError(err, "Setting kernel arguments");
         

        start_time = wtime();

        // We simply need to know if it succeeds
        const size_t global[1] = {size/5}; // we will chop off the last points, as this is more about testing operation support
        err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, global, NULL, 0, NULL, NULL); 
        checkError(err, "Enqueuing kernel");


        err = clFinish(commands);
        checkError(err, "Waiting for commands to finish");

        run_time = wtime() - start_time;

        printf("Arithmetic Kernel on %s took %f\n", device_names[i], run_time);


        err = clEnqueueReadBuffer(
            commands, d_f_c, CL_TRUE, 0,
            sizeof(float) * size, h_f_c,
            0, NULL, NULL);
        checkError(err, "Reading back buffer d_f_c");
        err = clEnqueueReadBuffer(
            commands, d_i_c, CL_TRUE, 0,
            sizeof(int) * size, h_i_c,
            0, NULL, NULL);
        checkError(err, "Reading back buffer d_i_c");


        // this for loop is for athenticating the results of the OpenCL to assert it is all correct
        for (int j = 0; j < size-4; j+=5){
            if (h_i_a[j] + h_i_b[j] != h_i_c[j]) goto fail;
            if (h_i_a[j+1] - h_i_b[j+1] != h_i_c[j+1]) goto fail;
            if (h_i_a[j+2] * h_i_b[j+2] != h_i_c[j+2]) goto fail;
            if (h_i_a[j+3] / h_i_b[j+3] != h_i_c[j+3]) goto fail;
            if (h_i_a[j+4] % h_i_b[j+4] != h_i_c[j+4]) goto fail;
            // tolerance may be an issue here
            // if (h_f_a[j] + h_f_b[j] != h_f_c[j]) goto fail;
            // if (h_f_a[j+1] - h_f_b[j+1] != h_f_c[j+1]) goto fail;
            // if (h_f_a[j+2] * h_f_b[j+2] != h_f_c[j+2]) goto fail;
            // if (h_f_a[j+3] / h_f_b[j+3] != h_f_c[j+3]) goto fail;
            // if (fmod(h_f_a[j+4], h_f_b[j+4]) != h_f_c[j+4]) goto fail;
            continue;
        fail:
            printf("kernel did not caclulate arithmetic within tolerance\n");
            printf("%d + %d = %d\n%d - %d = %d\n%d * %d = %d\n%d / %d = %d\n%d %% %d = %d\n"
            ,h_i_a[j], h_i_b[j], h_i_c[j], h_i_a[j+1], h_i_b[j+1], h_i_c[j+1], h_i_a[j+2], 
            h_i_b[j+2], h_i_c[j+2], h_i_a[j+3], h_i_b[j+3], h_i_c[j+3], h_i_a[j+4], h_i_b[j+4],
            h_i_c[j+4]);
            break;
        }

        clReleaseMemObject(d_i_a);
        clReleaseMemObject(d_i_b);
        clReleaseMemObject(d_i_c);
        clReleaseMemObject(d_f_a);
        clReleaseMemObject(d_f_b);
        clReleaseMemObject(d_f_c);
        clReleaseKernel(kernel);

    }


    fill_random_floats(h_f_a, size, 1.0, 10.0);
    

    // This for loop runs and compared the exponential operations on all OpenCL devices
    // Strangely, both llvm and V3D supported exp10 whereas the cpu on its own returned 0.0
    for (int i = 0; i < num_devices; i++){
        printf("\nChecking support for exponenetial operations for device %s\n", device_names[i]);
        set_active_device(i);


        cl_kernel kernel = clCreateKernel(program, "exp_support", &err);
        checkError(err, "Creating kernel");

        d_f_a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * size, h_f_a, &err);
        checkError(err, "Creating buffer d_f_a");
        d_f_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * size, NULL, &err);
        checkError(err, "Creating buffer d_f_c");


        // setting the arguments for the kernel
        err =  clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_f_a);
        err |=  clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_f_c);
        checkError(err, "Setting kernel arguments");


        start_time = wtime();

        // We simply need to know if it succeeds
        const size_t global[1] = {size}; // we will chop off the last points, as this is more about testing operation support
        err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, global, NULL, 0, NULL, NULL); 
        checkError(err, "Enqueuing kernel");


        err = clFinish(commands);
        checkError(err, "Waiting for commands to finish");

        run_time = wtime() - start_time;

        printf("Exponential Kernel on %s took %f\n", device_names[i], run_time);


        err = clEnqueueReadBuffer(
            commands, d_f_c, CL_TRUE, 0,
            sizeof(float) * size, h_f_c,
            0, NULL, NULL);
        checkError(err, "Reading back buffer d_f_c");

        for (int j = 0; j < size; j++)
        {
            float expected = exp(h_f_a[j]) + exp2(h_f_a[j]);
            if (fabs(expected - h_f_c[j]) > 0.01f)
            {
                printf("expected %f\n", expected);
                printf("for index = %d: %f + %f != %f\n", j, 
                    exp(h_f_a[j]), exp2(h_f_a[j]), h_f_c[j]);
                break;
            }
        }


        clReleaseMemObject(d_f_a);
        clReleaseMemObject(d_f_c);
        clReleaseKernel(kernel);
    }



    // This for loop runs and compared the log operations on all OpenCL devices
    for (int i = 0; i < num_devices; i++){
        printf("\nChecking support for logarithmic operations for device %s\n", device_names[i]);
        set_active_device(i);


        cl_kernel kernel = clCreateKernel(program, "log_support", &err);
        checkError(err, "Creating kernel");

        d_f_a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * size, h_f_a, &err);
        checkError(err, "Creating buffer d_f_a");
        d_f_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * size, NULL, &err);
        checkError(err, "Creating buffer d_f_c");


        // setting the arguments for the kernel
        err =  clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_f_a);
        err |=  clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_f_c);
        checkError(err, "Setting kernel arguments");


        start_time = wtime();

        // We simply need to know if it succeeds
        const size_t global[1] = {size}; // we will chop off the last points, as this is more about testing operation support
        err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, global, NULL, 0, NULL, NULL); 
        checkError(err, "Enqueuing kernel");


        err = clFinish(commands);
        checkError(err, "Waiting for commands to finish");

        run_time = wtime() - start_time;

        printf("Logarithmic Kernel on %s took %f\n", device_names[i], run_time);


        err = clEnqueueReadBuffer(
            commands, d_f_c, CL_TRUE, 0,
            sizeof(float) * size, h_f_c,
            0, NULL, NULL);
        checkError(err, "Reading back buffer d_f_c");

        for (int j = 0; j < size; j++)
        {
            float adjusted = h_f_a[j] + 1e-6f;
            float expected = log(adjusted) + log2(adjusted) + log10(adjusted);
            if (fabs(expected - h_f_c[j]) > 0.01f)
            {
                printf("expected %f\n", expected);
                printf("for index = %d: %f + %f + %f != %f\n", j, 
                    log(adjusted), log2(adjusted), log10(adjusted), h_f_c[j]);
                break;
            }
        }


        
        clReleaseMemObject(d_f_a);
        clReleaseMemObject(d_f_c);
        clReleaseKernel(kernel);
    }

    fill_random_floats(h_f_b, size, 1.0, 10.0);

    // This for loop runs and compared the power operations on all OpenCL devices
    // to the best of my knowledge, it is wildly inaccurate, especially V3D
    for (int i = 0; i < num_devices; i++){
        printf("\nChecking support for power operations for device %s\n", device_names[i]);
        set_active_device(i);
    
    
        cl_kernel kernel = clCreateKernel(program, "pow_support", &err);
        checkError(err, "Creating kernel");
    
        d_f_a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * size, h_f_a, &err);
        checkError(err, "Creating buffer d_f_a");
        d_f_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * size, h_f_b, &err);
        checkError(err, "Creating buffer d_f_b");
        d_f_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * size, NULL, &err);
        checkError(err, "Creating buffer d_f_c");
    
    
        // setting the arguments for the kernel
        err =  clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_f_a);
        err |=  clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_f_b);
        err |=  clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_f_c);
        checkError(err, "Setting kernel arguments");
    
    
        start_time = wtime();
    
        // We simply need to know if it succeeds
        const size_t global[1] = {size}; // we will chop off the last points, as this is more about testing operation support
        err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, global, NULL, 0, NULL, NULL); 
        checkError(err, "Enqueuing kernel");
    
    
        err = clFinish(commands);
        checkError(err, "Waiting for commands to finish");
    
        run_time = wtime() - start_time;
    
        printf("Exponential Kernel on %s took %f\n", device_names[i], run_time);
    
    
        err = clEnqueueReadBuffer(
            commands, d_f_c, CL_TRUE, 0,
            sizeof(float) * size, h_f_c,
            0, NULL, NULL);
        checkError(err, "Reading back buffer d_f_c");
    
        for (int j = 0; j < size; j++)
        {
            float expected = pow(h_f_a[j], h_f_b[j]);
            if (fabs(expected - h_f_c[j]) > 0.01f)
            {
                printf("expected %f\n", expected);
                printf("for index = %d: %f ^ %f != %f\n", j, 
                    h_f_a[j], h_f_b[j], h_f_c[j]);
                break;
            }
        }
    
            
        clReleaseMemObject(d_f_a);
        clReleaseMemObject(d_f_b);
        clReleaseMemObject(d_f_c);
        clReleaseKernel(kernel);
    }

    // This loop runs and compares the sqrt operations on all OpenCL devices
    for (int i = 0; i < num_devices; i++){
        printf("\nChecking support for square root operations for device %s\n", device_names[i]);
        set_active_device(i);


        cl_kernel kernel = clCreateKernel(program, "square_support", &err);
        checkError(err, "Creating kernel");

        d_f_a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * size, h_f_a, &err);
        checkError(err, "Creating buffer d_f_a");
        d_f_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * size, NULL, &err);
        checkError(err, "Creating buffer d_f_c");


        // setting the arguments for the kernel
        err =  clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_f_a);
        err |=  clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_f_c);
        checkError(err, "Setting kernel arguments");


        start_time = wtime();

        // We simply need to know if it succeeds
        const size_t global[1] = {size}; // we will chop off the last points, as this is more about testing operation support
        err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, global, NULL, 0, NULL, NULL); 
        checkError(err, "Enqueuing kernel");


        err = clFinish(commands);
        checkError(err, "Waiting for commands to finish");

        run_time = wtime() - start_time;

        printf("Square root Kernel on %s took %f\n", device_names[i], run_time);


        err = clEnqueueReadBuffer(
            commands, d_f_c, CL_TRUE, 0,
            sizeof(float) * size, h_f_c,
            0, NULL, NULL);
        checkError(err, "Reading back buffer d_f_c");

        for (int j = 0; j < size; j++)
        {
            float expected =  sqrt(h_f_a[j]);
            if (fabs(expected - h_f_c[j]) > 0.01f)
            {
                printf("expected %f\n", expected);
                printf("for index = %d: %f != %f\n", j, 
                    sqrt(h_f_a[j]), h_f_c[j]);
                break;
            }
        }


        
        clReleaseMemObject(d_f_a);
        clReleaseMemObject(d_f_c);
        clReleaseKernel(kernel);
    }

    fill_random_floats(h_f_a, size, -10.0, 10.0);

    // This for loop runs and compared rounding ops on all OpenCL devices
    for (int i = 0; i < num_devices; i++){
        printf("\nChecking support for rounding operations for device %s\n", device_names[i]);
        set_active_device(i);


        cl_kernel kernel = clCreateKernel(program, "round_support", &err);
        checkError(err, "Creating kernel");

        d_f_a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * size, h_f_a, &err);
        checkError(err, "Creating buffer d_f_a");
        d_f_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * size, NULL, &err);
        checkError(err, "Creating buffer d_f_c");


        // setting the arguments for the kernel
        err =  clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_f_a);
        err |=  clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_f_c);
        checkError(err, "Setting kernel arguments");


        start_time = wtime();

        // We simply need to know if it succeeds
        const size_t global[1] = {size}; // we will chop off the last points, as this is more about testing operation support
        err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, global, NULL, 0, NULL, NULL); 
        checkError(err, "Enqueuing kernel");


        err = clFinish(commands);
        checkError(err, "Waiting for commands to finish");

        run_time = wtime() - start_time;

        printf("Round Kernel on %s took %f\n", device_names[i], run_time);


        err = clEnqueueReadBuffer(
            commands, d_f_c, CL_TRUE, 0,
            sizeof(float) * size, h_f_c,
            0, NULL, NULL);
        checkError(err, "Reading back buffer d_f_c");

        for (int j = 0; j < size; j++)
        {
            float expected =  floor(h_f_a[j]) + ceil(h_f_a[j]) + round(h_f_a[j]) + trunc(h_f_a[j]) + rint(h_f_a[j]);
            if (fabs(expected - h_f_c[j]) > 0.01f)
            {
                printf("expected %f\n", expected);
                printf("Got: %f\n", h_f_c[j]);
                break;
            }
        }


        clReleaseMemObject(d_f_a);
        clReleaseMemObject(d_f_c);
        clReleaseKernel(kernel);
    }

    // This loop runs and compares absolute, min, and max operations on all OpenCL devices
    for (int i = 0; i < num_devices; i++){
        printf("\nChecking support for abs, min, and max operations for device %s\n", device_names[i]);
        set_active_device(i);
    
    
        cl_kernel kernel = clCreateKernel(program, "abs_support", &err);
        checkError(err, "Creating kernel");
    
        d_f_a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * size, h_f_a, &err);
        checkError(err, "Creating buffer d_f_a");
        d_f_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * size, h_f_b, &err);
        checkError(err, "Creating buffer d_f_b");
        d_f_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * size, NULL, &err);
        checkError(err, "Creating buffer d_f_c");
    
    
        // setting the arguments for the kernel
        err =  clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_f_a);
        err |=  clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_f_b);
        err |=  clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_f_c);
        checkError(err, "Setting kernel arguments");
    
    
        start_time = wtime();
    
        // We simply need to know if it succeeds
        const size_t global[1] = {size}; // we will chop off the last points, as this is more about testing operation support
        err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, global, NULL, 0, NULL, NULL); 
        checkError(err, "Enqueuing kernel");
    
    
        err = clFinish(commands);
        checkError(err, "Waiting for commands to finish");
    
        run_time = wtime() - start_time;
    
        printf("Abs, min, max Kernel on %s took %f\n", device_names[i], run_time);
    
    
        err = clEnqueueReadBuffer(
            commands, d_f_c, CL_TRUE, 0,
            sizeof(float) * size, h_f_c,
            0, NULL, NULL);
        checkError(err, "Reading back buffer d_f_c");
    
        for (int j = 0; j < size; j++)
        {
            float expected = fabs(h_f_a[j]) + fmin(h_f_a[j], h_f_b[j]) + fmax(h_f_a[j], h_f_b[j]);
            if (fabs(expected - h_f_c[j]) > 0.01f)
            {
                printf("expected %f\n", expected);
                printf("got %f\n", h_f_c[j]);
                break;
            }
        }
    
            
        clReleaseMemObject(d_f_a);
        clReleaseMemObject(d_f_b);
        clReleaseMemObject(d_f_c);
        clReleaseKernel(kernel);
    }


    // This loop runs and compares interpolation operations on all OpenCL devices
    for (int i = 0; i < num_devices; i++){
        printf("\nChecking support for interpolation operations for device %s\n", device_names[i]);
        set_active_device(i);
    
    
        cl_kernel kernel = clCreateKernel(program, "interpolation_support", &err);
        checkError(err, "Creating kernel");
    
        d_f_a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * size, h_f_a, &err);
        checkError(err, "Creating buffer d_f_a");
        d_f_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * size, h_f_b, &err);
        checkError(err, "Creating buffer d_f_b");
        d_f_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * size, NULL, &err);
        checkError(err, "Creating buffer d_f_c");
    
    
        // setting the arguments for the kernel
        err =  clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_f_a);
        err |=  clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_f_b);
        err |=  clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_f_c);
        checkError(err, "Setting kernel arguments");
    
    
        start_time = wtime();
    
        // We simply need to know if it succeeds
        const size_t global[1] = {size}; // we will chop off the last points, as this is more about testing operation support
        err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, global, NULL, 0, NULL, NULL); 
        checkError(err, "Enqueuing kernel");
    
    
        err = clFinish(commands);
        checkError(err, "Waiting for commands to finish");
    
        run_time = wtime() - start_time;
    
        printf("Abs, min, max Kernel on %s took %f\n", device_names[i], run_time);
    
    
        err = clEnqueueReadBuffer(
            commands, d_f_c, CL_TRUE, 0,
            sizeof(float) * size, h_f_c,
            0, NULL, NULL);
        checkError(err, "Reading back buffer d_f_c");

        // Have not verified correctness but it does return something whereas C 
        // does not have built in mix step and smoothstep functions.
    
            
        clReleaseMemObject(d_f_a);
        clReleaseMemObject(d_f_b);
        clReleaseMemObject(d_f_c);
        clReleaseKernel(kernel);
    }


    fill_random_floats(h_f_a, size, 1.0, 10.0);

    // This for loop runs and compared fast math ops on all OpenCL devices
    for (int i = 0; i < num_devices; i++){
        printf("\nChecking support for fast math operations for device %s\n", device_names[i]);
        set_active_device(i);


        cl_kernel kernel = clCreateKernel(program, "fast_math_support", &err);
        checkError(err, "Creating kernel");

        d_f_a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * size, h_f_a, &err);
        checkError(err, "Creating buffer d_f_a");
        d_f_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * size, NULL, &err);
        checkError(err, "Creating buffer d_f_c");


        // setting the arguments for the kernel
        err =  clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_f_a);
        err |=  clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_f_c);
        checkError(err, "Setting kernel arguments");


        start_time = wtime();

        // We simply need to know if it succeeds
        const size_t global[1] = {size}; // we will chop off the last points, as this is more about testing operation support
        err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, global, NULL, 0, NULL, NULL); 
        checkError(err, "Enqueuing kernel");


        err = clFinish(commands);
        checkError(err, "Waiting for commands to finish");

        run_time = wtime() - start_time;

        printf("Round Kernel on %s took %f\n", device_names[i], run_time);


        err = clEnqueueReadBuffer(
            commands, d_f_c, CL_TRUE, 0,
            sizeof(float) * size, h_f_c,
            0, NULL, NULL);
        checkError(err, "Reading back buffer d_f_c");

        for (int j = 0; j < size; j++)
        {
            float expected =  sin(h_f_a[j]) + exp(h_f_a[j]) + log(h_f_a[j]);
            // this is not super accurate
            if (fabs(expected - h_f_c[j]) > 0.5f)
            {
                printf("expected %f\n", expected);
                printf("Got: %f\n", h_f_c[j]);
                break;
            }
        }


        clReleaseMemObject(d_f_a);
        clReleaseMemObject(d_f_c);
        clReleaseKernel(kernel);
    }


    free(h_f_a);
    free(h_f_b);
    free(h_f_c);
    free(h_i_a);
    free(h_i_b);
    free(h_i_c);
}




int main(int argc, char *argv[]) {

    const char *files[] = {"instructions.cl"};

    // create a seed
    srand((unsigned int)time(NULL));
    


    cl_int num_kernels;
    size_t sizeofnumkernels;
    int device_index = 0;

    // if we specify a device to use in the command line
    // depreciated line of code, from when a single device was chosen instead of all of them being used
    parseArguments(argc, argv, &device_index);


    // should be two on the Raspberry Pi
    num_devices = getDeviceList(devices);


    for (int i = 0; i < num_devices; i++){

        getDeviceName(devices[i], device_names[i]);
        printf("\nFound OpenCL device: %s\n", device_names[i]);


        // Collecting as much information on the devices as possible
        char str[1024];
        size_t sz;
        cl_uint u32;
        cl_ulong u64;
        cl_bool boolean;
        cl_device_type dtype;
        cl_uint dims;
        size_t sizes[3];



        // Vendor & Version
        clGetDeviceInfo(devices[i], CL_DEVICE_VENDOR, sizeof(str), str, &sz);
        printf("  Vendor             : %s\n", str);
        clGetDeviceInfo(devices[i], CL_DEVICE_VENDOR_ID, sizeof(u32), &u32, NULL);
        printf("  Vendor ID          : %u\n", u32);
        clGetDeviceInfo(devices[i], CL_DEVICE_PROFILE, sizeof(str), str, &sz);
        printf("  Profile            : %s\n", str);
        clGetDeviceInfo(devices[i], CL_DEVICE_VERSION, sizeof(str), str, &sz);
        printf("  Device Version     : %s\n", str);
        clGetDeviceInfo(devices[i], CL_DRIVER_VERSION, sizeof(str), str, &sz);
        printf("  Driver Version     : %s\n", str);

        // Type & Availability
        clGetDeviceInfo(devices[i], CL_DEVICE_TYPE, sizeof(dtype), &dtype, NULL);
        printf("  Device Type        : %s\n",
               (dtype & CL_DEVICE_TYPE_CPU)     ? "CPU " :
               (dtype & CL_DEVICE_TYPE_GPU)     ? "GPU " :
               (dtype & CL_DEVICE_TYPE_ACCELERATOR) ? "ACCEL" :
                                                    "UNKNOWN");
        clGetDeviceInfo(devices[i], CL_DEVICE_AVAILABLE, sizeof(boolean), &boolean, NULL);
        printf("  Available          : %s\n", boolean ? "Yes" : "No");
        clGetDeviceInfo(devices[i], CL_DEVICE_COMPILER_AVAILABLE, sizeof(boolean), &boolean, NULL);
        printf("  Compiler Available : %s\n", boolean ? "Yes" : "No");

        // Compute units & clock
        clGetDeviceInfo(devices[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(u32), &u32, NULL);
        printf("  Compute Units      : %u\n", u32);
        clGetDeviceInfo(devices[i], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(u32), &u32, NULL);
        printf("  Max Clock (MHz)    : %u\n", u32);

        // Memory
        clGetDeviceInfo(devices[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(u64), &u64, NULL);
        printf("  Global Mem Size    : %llu MB\n", (unsigned long long)(u64 >> 20));
        clGetDeviceInfo(devices[i], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(u64), &u64, NULL);
        printf("  Local Mem Size     : %llu KB\n", (unsigned long long)(u64 >> 10));
        clGetDeviceInfo(devices[i], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(u64), &u64, NULL);
        printf("  Max Mem Alloc Size : %llu MB\n", (unsigned long long)(u64 >> 20));
        clGetDeviceInfo(devices[i], CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(u64), &u64, NULL);
        printf("  Global Cache Size  : %llu KB\n", (unsigned long long)(u64 >> 10));
        clGetDeviceInfo(devices[i], CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, sizeof(u32), &u32, NULL);
        printf("  Global Cache Type  : %s\n",
               u32 == CL_NONE    ? "None" :
               u32 == CL_READ_ONLY_CACHE ? "Read-Only" :
               u32 == CL_READ_WRITE_CACHE ? "Read-Write" : "Unknown");

        // Work-group and work-item sizes
        clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(dims), &dims, NULL);
        printf("  Work Item Dims     : %u\n", dims);
        clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(sizes), sizes, NULL);
        printf("  Max Work Item Size : %zu × %zu × %zu\n",
               sizes[0], sizes[1], sizes[2]);
        clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(sz), &sz, NULL);
        printf("  Max Work Group Size: %zu\n", sz);

        // Address width & profiling
        clGetDeviceInfo(devices[i], CL_DEVICE_ADDRESS_BITS, sizeof(u32), &u32, NULL);
        printf("  Address Bits       : %u\n", u32);
        clGetDeviceInfo(devices[i], CL_DEVICE_PROFILING_TIMER_RESOLUTION, sizeof(sz), &sz, NULL);
        printf("  Timer Resolution   : %zu ns\n", sz);

        // Vector widths
        clGetDeviceInfo(devices[i], CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR,  sizeof(u32), &u32, NULL);
        printf("  Preferred Vec Width (char)  : %u\n", u32);
        clGetDeviceInfo(devices[i], CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT,   sizeof(u32), &u32, NULL);
        printf("  Preferred Vec Width (int)   : %u\n", u32);
        clGetDeviceInfo(devices[i], CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, sizeof(u32), &u32, NULL);
        printf("  Preferred Vec Width (float) : %u\n", u32);
        clGetDeviceInfo(devices[i], CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE,sizeof(u32), &u32, NULL);
        printf("  Preferred Vec Width (double): %u\n", u32);

        // Image support
        clGetDeviceInfo(devices[i], CL_DEVICE_IMAGE_SUPPORT, sizeof(boolean), &boolean, NULL);
        printf("  Image Support      : %s\n", boolean ? "Yes" : "No");
        if (boolean) {
            clGetDeviceInfo(devices[i], CL_DEVICE_MAX_READ_IMAGE_ARGS, sizeof(u32), &u32, NULL);
            printf("  Max Read Image Args: %u\n", u32);
            clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WRITE_IMAGE_ARGS,sizeof(u32), &u32, NULL);
            printf("  Max Write Image Args: %u\n", u32);
            clGetDeviceInfo(devices[i], CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(u32), &u32, NULL);
            printf("  Image2D Max Width  : %u\n", u32);
            clGetDeviceInfo(devices[i], CL_DEVICE_IMAGE2D_MAX_HEIGHT,sizeof(u32), &u32, NULL);
            printf("  Image2D Max Height : %u\n", u32);
        }

        // Extensions & queue props
        clGetDeviceInfo(devices[i], CL_DEVICE_EXTENSIONS, sizeof(str), str, &sz);
        printf("  Extensions         : %s\n", str);
        clGetDeviceInfo(devices[i], CL_DEVICE_QUEUE_PROPERTIES, sizeof(u64), &u64, NULL);
        printf("  Queue Properties   : %s%s%s\n",
               (u64 & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) ? "OUT-OF-ORDER " : "",
               (u64 & CL_QUEUE_PROFILING_ENABLE)            ? "PROFILING "       : "");




        // Create a context which encompases specific device
        contexts[i] = clCreateContext(0, 1, &devices[i], NULL, NULL, &err);
        checkError(err, "Creating context");

        // Create a command queue for specific device
        commands_lst[i] = clCreateCommandQueue(contexts[i], devices[i], 0, &err);
        checkError(err, "Creating command queue");

        // build the programs for this specific device
        programs[i] = build_program_from_files(contexts[i], &devices[i], 1, files, 1, NULL);
    }


    cl_int ret = clGetProgramInfo(programs[0], CL_PROGRAM_NUM_KERNELS,
                (size_t)(sizeof(num_kernels)),
                (void *)&num_kernels,
                &sizeofnumkernels);

    printf("This program contains %d kernels\n", num_kernels);

    // As of now, both of these functions will not work on the Raspberry Pi, neither bool or char are supported
    // on the V3D
    //test_bool_support();
    //test_char_support();


    // all of these tests should work
    //test_int_support();
    //test_float_support();
    //test_size_t_support();
    test_operation_support();
    


cleanup:

    for (int i = 0; i < num_devices; i++) {
        clReleaseProgram(programs[i]);
        clReleaseCommandQueue(commands_lst[i]);
        clReleaseContext(contexts[i]);
    }


    return EXIT_SUCCESS;
}