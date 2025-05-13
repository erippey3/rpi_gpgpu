//------------------------------------------------------------------------------
//
// Name:     device_info()
//
// Purpose:  Function to output key parameters about the input OpenCL device.
//
//
// RETURN:   The OCL_SUCESS or the error code from one of the OCL function
//           calls internal to this function
//
// HISTORY:  Written by Tim Mattson, June 2010
//
//------------------------------------------------------------------------------
#include "device_info.h"

#define MAX_INFO_LEN 512

int output_device_info(cl_device_id device_id)
{
    int err;                            // error code returned from OpenCL calls
    cl_device_type device_type;         // Parameter defining the type of the compute device
    cl_uint comp_units;                 // the max number of compute units on a device
    cl_char vendor_name[1024] = {0};    // string to hold vendor name for compute device
    cl_char device_name[1024] = {0};    // string to hold name of compute device
#ifdef VERBOSE
    cl_uint          max_work_itm_dims;
    size_t           max_wrkgrp_size;
    size_t          *max_loc_size;
#endif


    err = clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_name), &device_name, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to access device name!\n");
        return EXIT_FAILURE;
    }
    printf(" \n Device is  %s ",device_name);

    err = clGetDeviceInfo(device_id, CL_DEVICE_TYPE, sizeof(device_type), &device_type, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to access device type information!\n");
        return EXIT_FAILURE;
    }
    if(device_type  == CL_DEVICE_TYPE_GPU)
       printf(" GPU from ");

    else if (device_type == CL_DEVICE_TYPE_CPU)
       printf("\n CPU from ");

    else 
       printf("\n non  CPU or GPU processor from ");

    err = clGetDeviceInfo(device_id, CL_DEVICE_VENDOR, sizeof(vendor_name), &vendor_name, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to access device vendor name!\n");
        return EXIT_FAILURE;
    }
    printf(" %s ",vendor_name);

    err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &comp_units, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to access device number of compute units !\n");
        return EXIT_FAILURE;
    }
    printf(" with a max of %d compute units \n",comp_units);

#ifdef VERBOSE
//
// Optionally print information about work group sizes
//
    err = clGetDeviceInfo( device_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), 
                               &max_work_itm_dims, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to get device Info (CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS)!\n",
                                                                            err_code(err));
        return EXIT_FAILURE;
    }
    
    max_loc_size = (size_t*)malloc(max_work_itm_dims * sizeof(size_t));
    if(max_loc_size == NULL){
       printf(" malloc failed\n");
       return EXIT_FAILURE;
    }
    err = clGetDeviceInfo( device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, max_work_itm_dims* sizeof(size_t), 
                               max_loc_size, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to get device Info (CL_DEVICE_MAX_WORK_ITEM_SIZES)!\n",err_code(err));
        return EXIT_FAILURE;
    }
    err = clGetDeviceInfo( device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), 
                               &max_wrkgrp_size, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to get device Info (CL_DEVICE_MAX_WORK_GROUP_SIZE)!\n",err_code(err));
        return EXIT_FAILURE;
    }
   printf("work group, work item information");
   printf("\n max loc dim ");
   for(int i=0; i< max_work_itm_dims; i++)
     printf(" %d ",(int)(*(max_loc_size+i)));
   printf("\n");
   printf(" Max work group size = %d\n",(int)max_wrkgrp_size);
#endif

    return CL_SUCCESS;

}

 
#ifdef __linux__

char* get_cpu_name() {
    static char cpu_name[MAX_INFO_LEN] = "Unknown";

    FILE *fp = fopen("/proc/cpuinfo", "r");
    if (fp) {
        char line[512];
        while (fgets(line, sizeof(line), fp)) {
            if (strncmp(line, "model name", 10) == 0) {
                char *colon = strchr(line, ':');
                if (colon) {
                    strncpy(cpu_name, colon + 2, MAX_INFO_LEN - 1);
                    cpu_name[strcspn(cpu_name, "\n")] = '\0'; // Remove newline
                }
                break;
            }
        }
        fclose(fp);
    } else {
        // Fallback to lscpu if /proc/cpuinfo is unavailable
        fp = popen("lscpu | grep 'Model name' | cut -d: -f2", "r");
        if (fp && fgets(cpu_name, MAX_INFO_LEN, fp)) {
            cpu_name[strcspn(cpu_name, "\n")] = '\0';
        }
        if (fp) pclose(fp);
    }

    return cpu_name;
}

char* get_gpu_name() {
    static char gpu_name[MAX_INFO_LEN] = "Unknown";

    FILE *fp = popen("lspci | grep -i 'vga\\|3d\\|display' | cut -d: -f3", "r");
    if (fp && fgets(gpu_name, MAX_INFO_LEN, fp)) {
        gpu_name[strcspn(gpu_name, "\n")] = '\0';
    }
    if (fp) pclose(fp);

    return gpu_name;
}

#else // Non-Linux fallback

char* get_cpu_name() {
    return "Unknown (Non-Linux)";
}

char* get_gpu_name() {
    return "Unknown (Non-Linux)";
}

#endif