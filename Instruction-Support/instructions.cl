// unsupported OpenCL extension
//#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Supported Data Type
__kernel void bool_support(
    __global bool *a, __global bool *b, __global bool *c
){
    uint i = get_global_id(0);
    c[i] = a[i] | b[i];
}


// Supported Data Type
__kernel void char_support(
    __global char *a, __global char *b, __global char *c
){
    uint i = get_global_id(0);
    c[i] = a[i] + b[i];
}

// Supported Data Type
__kernel void uchar_support(
    __global uchar *a, __global uchar *b, __global uchar *c
){
    uint i = get_global_id(0);
    c[i] = a[i] + b[i];
}

// Unsupported Data Type V3D
// Supported Data Type llvmpipe
// __kernel void short_support(
//     __global short *a, __global short *b, __global short *c
// ){
//     uint i = get_global_id(0);
//     c[i] = a[i] + b[i];
// }


// Unsupported Data Type V3D
// Supported Data Type llvmpipe
// __kernel void ushort_support(
//     __global ushort *a, __global ushort *b, __global ushort *c
// ){
//     uint i = get_global_id(0);
//     c[i] = a[i] + b[i];
// }


// Supported Data Type
__kernel void int_support(
    __global int *a, __global int *b, __global int *c
){
    uint i = get_global_id(0);
    c[i] = a[i] + b[i];
}

// Supported Data Type 
__kernel void uint_support(
    __global uint *a, __global uint *b, __global uint *c
){
    uint i = get_global_id(0);
    c[i] = a[i] + b[i];
}


// Unsupported data type V3D
// Supported Data Type llvmpipe
// __kernel void long_support(
//     __global long *a, __global long *b, __global long *c
// ){
//     uint i = get_global_id(0);
//     c[i] = a[i] + b[i];
// }


// Unsupported data type V3D
// Supported Data Type llvmpipe
// __kernel void ulong_support(
//     __global ulong *a, __global ulong *b, __global ulong *c
// ){
//     uint i = get_global_id(0);
//     c[i] = a[i] + b[i];
// }


// Issue with clvk
// __kernel void half_support(
//     __global half *a, __global half *b, __global half *c
// ){
//     uint i = get_global_id(0);
//     c[i] = a[i] + b[i];
// }

// Supported Data Type
__kernel void size_t_support(
    __global size_t *a, __global size_t *b, __global size_t *c
){
    uint i = get_global_id(0);
    c[i] = a[i] + b[i];
}


// Supported Data Type
__kernel void float_support(
    __global float *a, __global float *b, __global float *c
){
    uint i = get_global_id(0);
    c[i] = a[i] + b[i];
}


__kernel void arithmetic_support(
    __global float *f_a, __global float *f_b, __global float*f_c,
    __global int *i_a, __global int *i_b, __global int *i_c
) {
    uint gid = get_global_id(0);
    uint base = gid * 5;

    // int operations
    i_c[base]   = i_a[base]   + i_b[base];
    i_c[base+1] = i_a[base+1] - i_b[base+1];
    i_c[base+2] = i_a[base+2] * i_b[base+2];
    i_c[base+3] = i_a[base+3] / i_b[base+3];
    i_c[base+4] = i_a[base+4] % i_b[base+4];


    // float operations
    f_c[base]   = f_a[base]   + f_b[base];
    f_c[base+1] = f_a[base+1] - f_b[base+1];
    f_c[base+2] = f_a[base+2] * f_b[base+2];
    f_c[base+3] = f_a[base+3] / f_b[base+3];
    //f_c[base+4] = fmod(f_a[base+4], f_b[base+4]); // this absolutely kills performance
}

// Unsupported operations V3D
// Supported operations llvmpipe
// __kernel void trig_support(__global float* out, __global float* in) {
//     uint i = get_global_id(0);
//     float x = in[i];

//     out[i] = sin(x) + cos(x) + tan(x) + asin(x) + acos(x) + atan(x);
// }


// Supported operations
__kernel void exp_support(__global float* in, __global float* out) {
    uint i = get_global_id(0);
    float x = in[i];

    out[i] = exp(x) + exp2(x);
}

// Supported operations
__kernel void log_support(__global float* in, __global float* out) {
    uint i = get_global_id(0);
    float x = in[i] + 1e-6f; // avoid log(0)

    out[i] = log(x) + log2(x) + log10(x);
}

// Supported operations
__kernel void pow_support(__global float* in1, __global float* in2, __global float* out) {
    uint i = get_global_id(0);
    float base = in1[i];
    float expn = in2[i];

    out[i] = pow(base, expn);
}


// Supported operations
__kernel void square_support(__global float* in, __global float* out) {
    uint i = get_global_id(0);
    float x = in[i];

    out[i] = sqrt(x); // cbrt also available
    // cbrt not supported on pi
}

// Supported operations
__kernel void round_support(__global float* in, __global float* out) {
    uint i = get_global_id(0);
    float x = in[i];

    out[i] = floor(x) + ceil(x) + round(x) + trunc(x) + rint(x);
}

__kernel void abs_support(__global float* in1, __global float* in2, __global float* out) {
    uint i = get_global_id(0);
    float a = in1[i];
    float b = in2[i];

    out[i] = fabs(a) + fmin(a, b) + fmax(a, b);
}


// Supported operations
__kernel void interpolation_support(__global float* in1, __global float* in2, __global float* out) {
    uint i = get_global_id(0);
    float a = in1[i];
    float b = in2[i];
    float t = 0.5f;

    out[i] = mix(a, b, t) + step(a, b) + smoothstep(0.0f, 1.0f, a);
}

// Supported operations
__kernel void fast_math_support(__global float* in, __global float* out) {
    uint i = get_global_id(0);
    float x = in[i];

    out[i] = native_sin(x) + native_exp(x) + native_log(x);
}


