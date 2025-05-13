/*
 */

void __kernel csr(const unsigned int num_rows,
		__global unsigned int * Ap, 
		__global unsigned int * Aj, 
		__global float * Ax, 
		__global float * x, 
		__global float * y)
{
	unsigned int row = get_global_id(0);

	if(row < num_rows)
	{     
		float sum = 0.0f;

		const unsigned int row_start = Ap[row];      	
		const unsigned int row_end = Ap[row+1];

		unsigned int jj = 0;
		for (jj = row_start; jj < row_end; jj++)
			sum += Ax[jj] * x[Aj[jj]];      

		y[row] = sum;
	}
}



// on the pi this is largely unsuccessful, it achieves similar if not slighly worse performance
// larger tile sizes see drops in performance
// accesses are no longer coalessed by threads within a group
#define TILE_SIZE 8
void __kernel csr_tiled(const unsigned int num_rows,
		__global unsigned int * Ap, 
		__global unsigned int * Aj, 
		__global float * Ax, 
		__global float * x, 
		__global float * y)
{
	unsigned int row = get_global_id(0) * TILE_SIZE;

	
	// also more branching
	for (int i = row; i < row + TILE_SIZE; i++)
	{
		if(i < num_rows)
		{     
			float sum = 0.0f;

			const unsigned int row_start = Ap[i];      	
			const unsigned int row_end = Ap[i+1];

			unsigned int jj = 0;
			for (jj = row_start; jj < row_end; jj++)
				sum += Ax[jj] * x[Aj[jj]];      

			y[i] = sum;
		}
	}
}
