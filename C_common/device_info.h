#ifndef COMMON_DEVICE_INFO_H
#define COMMON_DEVICE_INFO_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
//
//  define VERBOSE if you want to print info about work groups sizes
//#define  VERBOSE 1
#ifdef VERBOSE
     extern int err_code(cl_int);
#endif

int output_device_info(cl_device_id device_id);

char* get_cpu_name();

char* get_gpu_name();

#endif //COMMON_DEVICE_INFO_H