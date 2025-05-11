#include <cl_utils.h>
#include <err_code.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

cl_program build_program_from_files(cl_context context,
                                    const cl_device_id *devices,
                                    cl_uint num_devices,
                                    const char **filenames,
                                    cl_uint num_files,
                                    const char *build_flags) {
    cl_int err;
    size_t total_length = 0;
    char **sources = malloc(num_files * sizeof(char *));
    size_t *lengths = malloc(num_files * sizeof(size_t));

    if (!sources || !lengths) {
        fprintf(stderr, "Memory allocation failure.\n");
        return NULL;
    }

    for (cl_uint i = 0; i < num_files; i++) {
        FILE *f = fopen(filenames[i], "r");
        if (!f) {
            perror(filenames[i]);
            goto cleanup;
        }

        fseek(f, 0, SEEK_END);
        size_t len = ftell(f);
        rewind(f);

        sources[i] = malloc(len + 1);
        if (!sources[i]) {
            fclose(f);
            fprintf(stderr, "Memory allocation failure for file: %s\n", filenames[i]);
            goto cleanup;
        }

        fread(sources[i], 1, len, f);
        sources[i][len] = '\0';
        lengths[i] = len;
        fclose(f);

        total_length += len;
    }

    cl_program program = clCreateProgramWithSource(context, num_files,
                                                   (const char **)sources, lengths, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clCreateProgramWithSource failed with error %d\n", err);
        program = NULL;
        goto cleanup;
    }

    err = clBuildProgram(program, num_devices, devices, build_flags, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clBuildProgram failed with error %d\n", err);

        // Print build log for each device
        for (cl_uint i = 0; i < num_devices; i++) {
            size_t log_size;
            clGetProgramBuildInfo(program, devices[i], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
            char *log = malloc(log_size + 1);
            if (log) {
                clGetProgramBuildInfo(program, devices[i], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
                log[log_size] = '\0';
                fprintf(stderr, "Build log for device %u:\n%s\n", i, log);
                free(log);
            }
        }

        clReleaseProgram(program);
        program = NULL;
    }

cleanup:
    for (cl_uint i = 0; i < num_files; i++) {
        free(sources[i]);
    }
    free(sources);
    free(lengths);

    return program;
}






void* char_new_array(const size_t N,const char* error_msg)
{
	void* ptr;
	int err;
	ptr = malloc(N * sizeof(char));
	check(ptr != NULL,error_msg);
	return ptr;
}

void* int_new_array(const size_t N,const char* error_msg)
{
	void* ptr;
	int err;
	ptr = malloc(N * sizeof(int));
	check(ptr != NULL,error_msg);
	return ptr;
}

void* long_new_array(const size_t N,const char* error_msg)
{
	void* ptr;
	int err;
	ptr = malloc(N * sizeof(long));
	check(ptr != NULL,error_msg);
	return ptr;
}

void* float_new_array(const size_t N,const char* error_msg)
{
	void* ptr;
	int err;
	ptr = malloc(N * sizeof(float));
	check(ptr != NULL,error_msg);
	return ptr;
}

void* float_array_realloc(void* ptr,const size_t N,const char* error_msg)
{
	int err;
	ptr = realloc(ptr,N * sizeof(float));
	check(ptr != NULL,error_msg);
	return ptr;
}