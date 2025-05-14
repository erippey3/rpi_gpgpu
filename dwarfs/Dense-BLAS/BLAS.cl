__kernel void axpy_kernel(const float alpha, __global float *x, __global float *y, const int N) {
    int i = get_global_id(0);
    if (i < N)
        x[i] = x[i]*alpha + y[i];
}



__kernel void scal_kernel(const float alpha, __global float *x, const int N) {
    int i = get_global_id(0);
    if (i < N)
        x[i] *= alpha;
}