#ifndef CL_UTILS_H
#define CL_UTILS_H

#include <CL/cl.h>

/**
 * @brief Builds an OpenCL program from one or more source files.
 *
 * @param context       OpenCL context.
 * @param devices       Array of OpenCL device IDs.
 * @param num_devices   Number of devices in the array.
 * @param filenames     Array of paths to source files.
 * @param num_files     Number of source files.
 * @param build_flags   Optional build flags (e.g., "-cl-fast-relaxed-math").
 * @return              Compiled OpenCL program, or NULL on failure.
 */
cl_program build_program_from_files(cl_context context,
                                    const cl_device_id *devices,
                                    cl_uint num_devices,
                                    const char **filenames,
                                    cl_uint num_files,
                                    const char *build_flags);



extern void* char_new_array(const size_t N,const char* error_msg);
extern void* int_new_array(const size_t N,const char* error_msg);
extern void* long_new_array(const size_t N,const char* error_msg);
extern void* float_new_array(const size_t N,const char* error_msg);
extern void* float_array_realloc(void* ptr,const size_t N,const char* error_msg);

#endif // CL_UTILS_H
