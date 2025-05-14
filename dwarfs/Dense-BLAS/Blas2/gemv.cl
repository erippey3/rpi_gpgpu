#define TILE 16

__kernel void gemv_blocked(
    const int M,
    const int N,
    __global const float* restrict A,
    __global const float* restrict x,
    __global float* restrict y,
    __local float* x_tile) // shared tile of x
{
    int row = get_global_id(0);
    int lid = get_local_id(0);
    int group_size = get_local_size(0);

    float sum = 0.0f;

    // Process input vector x in tiles
    for (int t = 0; t < N; t += TILE)
    {
        // Each thread loads part of x into shared memory
        if (t + lid < N)
            x_tile[lid] = x[t + lid];
        barrier(CLK_LOCAL_MEM_FENCE);

        // Each thread computes dot product using this tile
        for (int j = 0; j < TILE && t + j < N; j++)
            sum += A[row * N + t + j] * x_tile[j];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    y[row] = sum;
}
