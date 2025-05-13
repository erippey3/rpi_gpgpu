
#ifndef COMMON_ERR_CODE_H
#define COMMON_ERR_CODE_H
/*----------------------------------------------------------------------------
 *
 * Name:     err_code()
 *
 * Purpose:  Function to output descriptions of errors for an input error code
 *           and quit a program on an error with a user message
 *
 *
 * RETURN:   echoes the input error code / echos user message and exits
 *
 * HISTORY:  Written by Tim Mattson, June 2010
 *           This version automatically produced by genErrCode.py
 *           script written by Tom Deakin, August 2013
 *           Modified by Bruce Merry, March 2014
 *           Updated by Tom Deakin, October 2014
 *               Included the checkError function written by
 *               James Price and Simon McIntosh-Smith
 *
 *----------------------------------------------------------------------------
 */
#include <CL/cl.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>


const char *err_code (cl_int err_in);

void check_error(cl_int err, const char *operation, char *filename, int line);

void check(int b,const char* msg);

// https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/

#define REL_EPSILON 1e-5f
#define ABS_EPSILON 1e-8f
bool AlmostEqualRelative(float A, float B);


#define checkError(E, S) check_error(E,S,__FILE__,__LINE__)


#define MINIMUM(i,j) ((i)<(j) ? (i) : (j))

#define CHKERR(err, str) \
    if (err != CL_SUCCESS) \
    { \
        fprintf(stdout, "CL Error %d: %s\n", err, str); \
        exit(1); \
    }

#endif