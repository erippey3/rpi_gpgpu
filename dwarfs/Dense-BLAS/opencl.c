#include "opencl.h"
#include "throughput.h"
#include <CL/cl.h>
#include <err_code.h>
#include <device_picker.h>
#include <cl_utils.h>
#include <stdio.h>
#include <wtime.h>

cl_int err;
cl_device_id device;
cl_context context;
cl_command_queue commands;
cl_program program;

const char *program_files[] = {"BLAS.cl"};


/* returns dot product of v1 and v2*/
float opencl_dot(vector v1, vector v2){
    printf("just use openmp\n");
}

/* scales x by alpha and adds y in place*/
void opencl_axpy(float alpha, vector x, vector y){
    cl_kernel kernel = clCreateKernel(program, "axpy_kernel", &err);
    checkError(err, "opencl axpy Creating Kernel");

    cl_mem d_x = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                sizeof(float) * x.length, x.data, &err);
    checkError(err, "opencl axpy creating buffer d_x");
    cl_mem d_y = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                sizeof(float) * x.length, x.data, &err);
    checkError(err, "opencl axpy creating buffer d_y");

    err = clSetKernelArg(kernel, 0, sizeof(float), &alpha);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_x);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_y);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &x.length);
    checkError(err, "opencl axpy setting kernel arguments");

    size_t global_work_size = x.length;
    cl_event kernel_event;

    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL,
                                 &global_work_size, NULL, 0, NULL, &kernel_event);
    checkError(err, "opencl axpy enqueueing kernel");

    // Wait for kernel to finish
    clWaitForEvents(1, &kernel_event);

    // Read result back
    err = clEnqueueReadBuffer(commands, d_x, CL_TRUE, 0,
                              sizeof(float) * x.length, x.data, 0, NULL, NULL);
    checkError(err, "opencl axpy reading buffer back");

    clReleaseEvent(kernel_event);
    clReleaseMemObject(d_x);
    clReleaseMemObject(d_y);
    clReleaseKernel(kernel);
}

/* scales x by alpha in place */
void opencl_scal(float alpha, vector x) {
    cl_kernel kernel = clCreateKernel(program, "scal_kernel", &err);
    checkError(err, "opencl scal Creating Kernel");

    cl_mem d_x = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                sizeof(float) * x.length, x.data, &err);
    checkError(err, "opencl scal creating buffer d_x");

    err = clSetKernelArg(kernel, 0, sizeof(float), &alpha);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_x);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &x.length);
    checkError(err, "opencl scal setting kernel arguments");

    size_t global_work_size = x.length;
    cl_event kernel_event;

    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL,
                                 &global_work_size, NULL, 0, NULL, &kernel_event);
    checkError(err, "opencl scal enqueueing kernel");

    // Wait for kernel to finish
    clWaitForEvents(1, &kernel_event);

    // Read result back
    err = clEnqueueReadBuffer(commands, d_x, CL_TRUE, 0,
                              sizeof(float) * x.length, x.data, 0, NULL, NULL);
    checkError(err, "opencl scal reading buffer back");

    clReleaseEvent(kernel_event);
    clReleaseMemObject(d_x);
    clReleaseKernel(kernel);
}


void opencl_throughput_test(benchmark *b){
    unsigned long seed1 = (unsigned long)(rand() * 100000);
    unsigned long seed2 = (unsigned long)(rand() * 100000);
    vector x = rand_vector(LENGTH, &seed1, stdout);
    vector y = rand_vector(LENGTH, &seed2, stdout);
    float alpha = 2.0f;

    double start, elapsed;
    float result;

    // Test AXPY
    b->test_name = "AXPY";
    start = wtime();
    for (int i = 0; i < NUM_TRIALS; i++) {
        opencl_axpy(alpha, x, y);
    }
    elapsed = wtime() - start;
    double axpy_flops = LENGTH * 2.0 * NUM_TRIALS;
    printf("AXPY: %.2f GFLOPS\n", (axpy_flops / elapsed) / 1e9);

    // Test SCAL
    b->test_name = "SCAL";
    start = wtime();
    for (int i = 0; i < NUM_TRIALS; i++) {
        opencl_scal(alpha, x);
    }
    elapsed = wtime() - start;
    double scal_flops = LENGTH * 1.0 * NUM_TRIALS; // one multiply per element
    printf("SCAL: %.2f GFLOPS\n", (scal_flops / elapsed) / 1e9);

    free(x.data);
    free(y.data);
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