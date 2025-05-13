#include "opencl.h"
#include <err_code.h>
#include <cl_utils.h>
#include <device_picker.h>
#include <wtime.h>
#include <math.h>


cl_int err;
cl_device_id device;
cl_context context;
cl_command_queue commands;
cl_program program;

const char *program_files[] = {"SpMV.cl"};




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

#define TILE_SIZE 8
vector *opencl_csr_spmv(const csr_matrix * m, const vector * v, benchmark *b){

    cl_mem d_Ap, d_Aj, d_Ax, d_x, d_y;
    double start_time;



    // creating memory buffers on device of choice
    d_Ap = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(int) * (m->num_rows + 1), m->Ap, &err);
    checkError(err, "Creating buffer d_Ap");

    d_Aj = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(int) * (m->num_nonzeros), m->Aj, &err);
    checkError(err, "Creating buffer d_Aj");        

    d_Ax = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * (m->num_nonzeros), m->Ax, &err);
    checkError(err, "Creating buffer d_Ax");

    d_x = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * (v->length), v->data, &err);
    checkError(err, "Creating buffer d_x");

    d_y = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * (m->num_rows),
        NULL, &err); 
    checkError(err, "Creating buffer d_y");

    // creating kernel from program file
    cl_kernel kernel = clCreateKernel(program, "csr_tiled", &err);
    checkError(err, "Creating kernel csr");

    

    // setting arguments of kernel
    err = clSetKernelArg(kernel, 0, sizeof(int), &m->num_rows);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_Ap);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_Aj);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_Ax);
    err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_x);
    err |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &d_y);
    checkError(err, "Setting kernel args");


    if (b)
        start_time = wtime();


    const size_t global[1] = {ceil((double)m->num_rows/TILE_SIZE)};
    const size_t local[1] = {64}; // wait off on this
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, global, NULL, 0, NULL, NULL);
    checkError(err, "Enqueueing kernel");

    err = clFinish(commands);

    checkError(err, "Waiting for kernel to finish");
    if (b) 
    {
        double delta_time = wtime() - start_time;
        add_runtime(b, delta_time);
    }


    vector *results = (vector *) malloc (sizeof(vector));
    results->length = m->num_rows;
    results->data = (float *) malloc(sizeof(float) * results->length);


    err = clEnqueueReadBuffer(commands, d_y, CL_TRUE, 0, sizeof(float) * results->length,
        results->data, 0, NULL, NULL);
    
    checkError(err, "Reaing back d_y");

    clReleaseMemObject(d_Ap);
    clReleaseMemObject(d_Aj);
    clReleaseMemObject(d_Ax);
    clReleaseMemObject(d_x);
    clReleaseMemObject(d_y);
    clReleaseKernel(kernel);

    return results;
}


vector *opencl_coo_spmv(const coo_matrix *, const vector *, benchmark *b){
    return NULL;
}

void shutdown_cl(FILE *stream){
    clReleaseProgram(program);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);
}