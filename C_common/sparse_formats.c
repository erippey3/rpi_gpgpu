#include <stdlib.h>
#include <cl_utils.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <err_code.h>
#include "sparse_formats.h"
#include "ziggurat.h"



triplet* triplet_new_array(const size_t N) {
	//dispatch on location
	return (triplet*) malloc(N * sizeof(triplet));
}

int triplet_comparator(const void *v1, const void *v2)
{
	const triplet* t1 = (triplet*) v1;
	const triplet* t2 = (triplet*) v2;

	if(t1->i < t2->i)
		return -1;
	else if(t1->i > t2->i)
		return +1;
	else if(t1->j < t2->j)
		return -1;
	else if(t1->j > t2->j)
		return +1;
	else
		return 0;
}

int unsigned_int_comparator(const void* v1, const void* v2)
{
	const unsigned int int1 = *((unsigned int*) v1);
	const unsigned int int2 = *((unsigned int*) v2);

	if(int1 < int2)
		return -1;
	else if(int1 > int2)
		return +1;
	else
		return 0;
}

void write_csr(const csr_matrix* csr,const unsigned int num_csr,const char* file_path)
{
	FILE* fp;
	int i,j;
	fp = fopen(file_path,"w");
	check(fp != NULL,"sparse_formats.write_csr() - Cannot Open File");
	fprintf(fp,"%u\n\n",num_csr);

	for(j=0; j<num_csr; j++)
	{
		fprintf(fp,"%u\n%u\n%u\n%u\n%lf\n%lf\n%lf\n",csr[j].num_rows,csr[j].num_cols,csr[j].num_nonzeros,csr[j].density_ppm,csr[j].density_perc,csr[j].nz_per_row,csr[j].stddev);

		for(i=0; i<=csr[j].num_rows; i++)
			fprintf(fp,"%u ",csr[j].Ap[i]);
		fprintf(fp,"\n");

		for(i=0; i<csr[j].num_nonzeros; i++)
			fprintf(fp,"%u ",csr[j].Aj[i]);
		fprintf(fp,"\n");

		for(i=0; i<csr[j].num_nonzeros; i++)
			fprintf(fp,"%f ",csr[j].Ax[i]);
		fprintf(fp,"\n\n");
	}

	fclose(fp);
}

csr_matrix* read_csr(unsigned int* num_csr,const char* file_path)
{
	FILE* fp;
	int i,j,read_count;
	csr_matrix* csr;

	check(num_csr != NULL,"sparse_formats.read_csr() - ptr to num_csr is NULL!");

	fp = fopen(file_path,"r");
	check(fp != NULL,"sparse_formats.read_csr() - Cannot Open Input File");

	read_count = fscanf(fp,"%u\n\n",num_csr);
	check(read_count == 1,"sparse_formats.read_csr() - Input File Corrupted! Read count for num_csr differs from 1");
	csr = malloc(sizeof(struct csr_matrix)*(*num_csr));

	for(j=0; j<*num_csr; j++)
	{
		read_count = fscanf(fp,"%u\n%u\n%u\n%u\n%lf\n%lf\n%lf\n",&(csr[j].num_rows),&(csr[j].num_cols),&(csr[j].num_nonzeros),&(csr[j].density_ppm),&(csr[j].density_perc),&(csr[j].nz_per_row),&(csr[j].stddev));
		check(read_count == 7,"sparse_formats.read_csr() - Input File Corrupted! Read count for header info differs from 7");

		read_count = 0;
		csr[j].Ap = int_new_array(csr[j].num_rows+1,"sparse_formats.read_csr() - Heap Overflow! Cannot allocate space for csr.Ap");
		for(i=0; i<=csr[j].num_rows; i++)
			read_count += fscanf(fp,"%u ",csr[j].Ap+i);
		check(read_count == (csr[j].num_rows+1),"sparse_formats.read_csr() - Input File Corrupted! Read count for Ap differs from csr[j].num_rows+1");

		read_count = 0;
		csr[j].Aj = int_new_array(csr[j].num_nonzeros,"sparse_formats.read_csr() - Heap Overflow! Cannot allocate space for csr.Aj");
		for(i=0; i<csr[j].num_nonzeros; i++)
			read_count += fscanf(fp,"%u ",csr[j].Aj+i);
		check(read_count == (csr[j].num_nonzeros),"sparse_formats.read_csr() - Input File Corrupted! Read count for Aj differs from csr[j].num_nonzeros");

		read_count = 0;
		csr[j].Ax = float_new_array(csr[j].num_nonzeros,"sparse_formats.read_csr() - Heap Overflow! Cannot allocate space for csr.Ax");
		for(i=0; i<csr[j].num_nonzeros; i++)
			read_count += fscanf(fp,"%f ",csr[j].Ax+i);
		check(read_count == (csr[j].num_nonzeros),"sparse_formats.read_csr() - Input File Corrupted! Read count for Ax differs from csr[j].num_nonzeros");
	}

	fclose(fp);
	return csr;
}

void print_timestamp(FILE* stream)
{
	time_t rawtime;
	struct tm* timeinfo;

	time(&rawtime);
	timeinfo = localtime(&rawtime);
	fprintf(stream,"Current time: %s",asctime(timeinfo));
}

unsigned long gen_rand(const long LB, const long HB)
{
	int range = HB - LB + 1;
	check((HB >= 0 && LB >= 0 && range > 0),"sparse_formats.gen_rand() - Invalid Bound(s). Exiting...");
	return (rand() % range) + LB;
}

csr_matrix laplacian_5pt(const unsigned int N)
{

	csr_matrix csr;
	csr.num_rows = N*N;
	csr.num_cols = N*N;
	csr.num_nonzeros = 5*N*N - 4*N;

	csr.Ap = int_new_array(csr.num_rows+4,"sparse_formats.laplacian_5pt() - Heap Overflow! Cannot allocate space for csr.Ap");
	csr.Aj = int_new_array(csr.num_nonzeros,"sparse_formats.laplacian_5pt() - Heap Overflow! Cannot allocate space for csr.Aj");
	csr.Ax = float_new_array(csr.num_nonzeros,"sparse_formats.laplacian_5pt() - Heap Overflow! Cannot allocate space for csr.Ax");

	unsigned int nz = 0;
	unsigned int i = 0;
	unsigned int j = 0;
	unsigned int indx = 0;

	for(i = 0; i < N; i++){
		for(j = 0; j < N; j++){
			indx = N*i + j;

			if (i > 0){
				csr.Aj[nz] = indx - N;
				csr.Ax[nz] = -1;
				nz++;
			}

			if (j > 0){
				csr.Aj[nz] = indx - 1;
				csr.Ax[nz] = -1;
				nz++;
			}

			csr.Aj[nz] = indx;
			csr.Ax[nz] = 4;
			nz++;

			if (j < N - 1){
				csr.Aj[nz] = indx + 1;
				csr.Ax[nz] = -1;
				nz++;
			}

			if (i < N - 1){
				csr.Aj[nz] = indx + N;
				csr.Ax[nz] = -1;
				nz++;
			}

			csr.Ap[indx + 1] = nz;
		}
	}
	return csr;
}


int bin_search(const triplet* data, int size, const triplet* key)
{
	triplet* mid_triplet;
	int lo,hi,m;
	lo = 0;
	hi = size-1;
	while(lo <= hi) //binary search to determine if element exists and, if not, what is the proper index for insertion
	{
		m = lo + ((hi - lo)/2);
		if(triplet_comparator(key,&(data[m])) > 0)
			lo = m + 1;
		else if (triplet_comparator(key,&(data[m])) < 0)
			hi = m - 1;
		else
			return m;
	}
	return (-1*lo - 1);
}

coo_matrix rand_coo(const unsigned int N,const unsigned long density, FILE* log)
{
	fprintf(log,"Creating Random Coordinate List Matrix\n");
	coo_matrix coo;
	triplet tmp, *current_triplet, *mid_triplet;

	unsigned int ind;
	int m;

	coo.num_rows = N;
	coo.num_cols = N;
	coo.density_ppm = density;
	coo.num_nonzeros = (((double)(N*density))/1000000.0)*N;
	printf("NUM_nonzeros: %d\n",coo.num_nonzeros);

	coo.non_zero = triplet_new_array(coo.num_nonzeros);
	check(coo.non_zero != NULL,"sparse_formats.rand_coo_bin_insertion(): Heap Overflow - Cannot allocate memory for coo.non_zero\n");
	print_timestamp(log);
	fprintf(log,"Memory Allocated. Generating Data...\n");

	current_triplet = &(coo.non_zero[0]); //Generate random first element
	(current_triplet->i) = gen_rand(0,N-1);
	(current_triplet->j) = gen_rand(0,N-1);

	for(ind=1; ind<coo.num_nonzeros; ind++)
	{
		current_triplet = &(coo.non_zero[ind]); //element to be inserted
		(current_triplet->i) = gen_rand(0,N-1);
		(current_triplet->j) = gen_rand(0,N-1);

		m = bin_search(coo.non_zero,ind,current_triplet);
		if(m < 0)
		{
			m = -1*m - 1;
		}
		else
		{
			ind--;
			continue;
		}

		if(m < ind)
		{
			tmp = *current_triplet;
			memmove(coo.non_zero + m + 1,coo.non_zero+m,sizeof(triplet)*(ind-m));
			coo.non_zero[m] = tmp;
		}
	}

	for(ind=0; ind<coo.num_nonzeros; ind++)
	{
		current_triplet = &(coo.non_zero[ind]);
		(current_triplet->v) = 1.0 - 2.0 * (rand() / (2147483647 + 1.0));
		while((current_triplet->v) == 0.0)
			(current_triplet->v) = 1.0 - 2.0 * (rand() / (2147483647 + 1.0));
	}

	print_timestamp(log);
	fprintf(log,"Matrix Completed. Returning...\n");

	return coo;
}

void print_coo_metadata(const coo_matrix* coo, FILE* stream) {
	fprintf(stream,"\nCOO Matrix Metadata:\n\nNRows=%d\tNCols=%d\tNNZ=%d\tDensity (ppm)=%d\tDensity (fract)=%g\n\n",coo->num_rows,coo->num_cols,coo->num_nonzeros,coo->density_ppm,(double)(coo->density_ppm/1000000.0));
}

void print_csr_metadata(const csr_matrix* csr, FILE* stream) {
	fprintf(stream,"\nCSR Matrix Metadata:\n\nNRows=%lu\tNCols=%lu\tNNZ=%lu\tDensity=%lu ppm = %g%%\tAverage NZ/Row=%g\tStdDev NZ/Row=%g\n\n",csr->num_rows,csr->num_cols,csr->num_nonzeros,csr->density_ppm,csr->density_perc,csr->nz_per_row,csr->stddev);
}

void print_coo(const coo_matrix* coo, FILE* stream)
{
	unsigned int ind;
	fprintf(stream,"\nPrinting COO Matrix in COO Form:\n\nNRows=%d\nNCols=%d\nNNZ=%d\nDensity (ppm)=%d\nDensity (fract)=%g\n",coo->num_rows,coo->num_cols,coo->num_nonzeros,coo->density_ppm,(double)(coo->density_ppm/1000000.0));
	for(ind=0; ind<coo->num_nonzeros; ind++)
		fprintf(stream,"(%2d,%2d,%5.2f)\n",coo->non_zero[ind].i,coo->non_zero[ind].j,coo->non_zero[ind].v);
}

void print_coo_std(const coo_matrix* coo,FILE* stream)
{
	int ind,ind2,nz_count=0;
	float val;

	fprintf(stream,"\nPrinting COO Matrix in Standard Form:\n\nNRows=%d\nNCols=%d\nNNZ=%d\nDensity (ppm)=%d\nDensity (fract)=%g\n",coo->num_rows,coo->num_cols,coo->num_nonzeros,coo->density_ppm,(double)(coo->density_ppm/1000000.0));

	for(ind=0; ind<coo->num_rows; ind++)
	{
		fprintf(stream,"[");
		for(ind2=0; ind2<coo->num_cols; ind2++)
		{
			if(ind == coo->non_zero[nz_count].i && ind2 == coo->non_zero[nz_count].j)
				val = coo->non_zero[nz_count++].v;
			else
				val = 0.0;
			fprintf(stream,"%6.2f",val);
		}
		fprintf(stream,"]\n");
	}
}

void print_csr_arr_std(const csr_matrix* csr, const unsigned int num_csr, FILE* stream)
{
	unsigned int k;
	for(k=0; k<num_csr; k++)
		print_csr_std(&csr[k],stream);
}

void print_csr_std(const csr_matrix* csr,FILE* stream)
{
	int ind,ind2,nz_count=0,row_count=0,next_nz_row;
	float val,density;
	density = ((float)(csr->num_nonzeros))/(((float)(csr->num_rows))*((float)(csr->num_cols)));

	print_csr_metadata(csr,stream);

	while(csr->Ap[row_count+1] == nz_count)
		row_count++;

	for(ind=0; ind<csr->num_rows; ind++)
	{
		fprintf(stream,"[");
		for(ind2=0; ind2<csr->num_cols; ind2++)
		{
			if(ind == row_count && ind2 == csr->Aj[nz_count])
			{
				val = csr->Ax[nz_count++];
				while(csr->Ap[row_count+1] == nz_count)
					row_count++;
			}
			else
				val = 0.0;
			fprintf(stream,"%6.2f",val);
		}
		fprintf(stream,"]\n");
	}
	fprintf(stream,"\n");
}

csr_matrix coo_to_csr(const coo_matrix* coo,FILE* log)
{
	fprintf(log, "Converting matrix from coordinate list to compressed sparse row\n");
	int ind,row_count,newline_count;

	csr_matrix csr;
	csr.num_rows = coo->num_rows;
	csr.num_cols = coo->num_cols;
	csr.num_nonzeros = coo->num_nonzeros;

	csr.Ap = int_new_array(csr.num_rows+1,"sparse_formats.coo_to_csr() - Heap Overflow! Cannot allocate space for csr.Ap");
	csr.Aj = int_new_array(csr.num_nonzeros,"sparse_formats.coo_to_csr() - Heap Overflow! Cannot allocate space for csr.Aj");
	csr.Ax = float_new_array(csr.num_nonzeros,"sparse_formats.coo_to_csr() - Heap Overflow! Cannot allocate space for csr.Ax");

	print_timestamp(log);
	fprintf(log,"Memory Allocated. Copying column indices & values...\n");

	for(ind=0; ind<coo->num_nonzeros; ind++)
	{
		csr.Ax[ind] = coo->non_zero[ind].v;
		csr.Aj[ind] = coo->non_zero[ind].j;
	}

	print_timestamp(log);
	fprintf(log,"Calculating Row Pointers...\n");

	row_count = 0;
	ind = 0;
	while(row_count <= coo->non_zero[ind].i)
		csr.Ap[row_count++] = 0;

	for(ind=1; ind<coo->num_nonzeros; ind++)
	{
		newline_count = coo->non_zero[ind].i - coo->non_zero[ind-1].i;
		while(newline_count > 0)
		{
			csr.Ap[row_count++] = ind;
			newline_count--;
		}
	}
	csr.Ap[row_count] = csr.num_nonzeros;

	print_timestamp(log);
	fprintf(log,"Conversion Complete. Returning...\n");

	return csr;
}

csr_matrix rand_csr(const unsigned int N,const unsigned int density, const double normal_stddev,unsigned long* seed,FILE* log)
{
	fprintf(log,"Creating Random Condensed Sparse Row Matrix\n");
	unsigned int i,j,nnz_ith_row,nnz,update_interval,rand_col;
	double nnz_ith_row_double,nz_error,nz_per_row_doubled,high_bound;
	int kn[128];
	float fn[128],wn[128];
	char* used_cols;
	csr_matrix csr;

	csr.num_rows = N;
	csr.num_cols = N;
	csr.density_perc = (((double)(density))/10000.0);
	csr.nz_per_row = (((double)N)*((double)density))/1000000.0;
	csr.num_nonzeros = round(csr.nz_per_row*N);
	csr.stddev = normal_stddev * csr.nz_per_row; //scale normalized standard deviation by average NZ/row

	fprintf(log,"Average NZ/Row: %-8.3f\n",csr.nz_per_row);
	fprintf(log,"Standard Deviation: %-8.3f\n",csr.stddev);
	fprintf(log,"Target Density: %u ppm = %g%%\n",density,csr.density_perc);
	fprintf(log,"Approximate NUM_nonzeros: %d\n",csr.num_nonzeros);

	csr.Ap = int_new_array(csr.num_rows+1,"rand_csr() - Heap Overflow! Cannot Allocate Space for csr.Ap");
	csr.Aj = int_new_array(csr.num_nonzeros,"rand_csr() - Heap Overflow! Cannot Allocate Space for csr.Aj");

	csr.Ap[0] = 0;
	nnz = 0;
	nz_per_row_doubled = 2*csr.nz_per_row; //limit nnz_ith_row to double the average because negative values are rounded up to 0. This
	high_bound = MINIMUM(csr.num_cols,nz_per_row_doubled); //limitation ensures the distribution will be symmetric about the mean, albeit not truly normal.
	used_cols = malloc(csr.num_cols*sizeof(char));
	check(used_cols != NULL,"rand_csr() - Heap Overflow! Cannot allocate space for used_cols");

	r4_nor_setup(kn,fn,wn);
	srand(*seed);

	update_interval = round(csr.num_rows / 10.0);
	if(!update_interval) update_interval = csr.num_rows;

	for(i=0; i<csr.num_rows; i++)
	{
		if(i % update_interval == 0) fprintf(log,"\t%d of %d (%5.1f%%) Rows Generated. Continuing...\n",i,csr.num_rows,((double)(i))/csr.num_rows*100);

		nnz_ith_row_double = r4_nor(seed,kn,fn,wn); //random, normally-distributed value for # of nz elements in ith row, NORMALIZED
		nnz_ith_row_double *= csr.stddev; //scale by standard deviation
		nnz_ith_row_double += csr.nz_per_row; //add average nz/row
		if(nnz_ith_row_double < 0)
			nnz_ith_row = 0;
		else if(nnz_ith_row_double > high_bound)
			nnz_ith_row = high_bound;
		else
			nnz_ith_row = (unsigned int) round(nnz_ith_row_double);

		csr.Ap[i+1] = csr.Ap[i] + nnz_ith_row;
		if(csr.Ap[i+1] > csr.num_nonzeros)
			csr.Aj = realloc(csr.Aj,sizeof(unsigned int)*csr.Ap[i+1]);

		for(j=0; j<csr.num_cols; j++)
			used_cols[j] = 0;

		for(j=0; j<nnz_ith_row; j++)
		{
			rand_col = gen_rand(0,csr.num_cols - 1);
			if(used_cols[rand_col])
			{
				j--;
			}
			else
			{
				csr.Aj[csr.Ap[i]+j] = rand_col;
				used_cols[rand_col] = 1;
			}
		}
		qsort((&(csr.Aj[csr.Ap[i]])),nnz_ith_row,sizeof(unsigned int),unsigned_int_comparator);
	}

	nz_error = ((double)abs((signed int)(csr.num_nonzeros - csr.Ap[csr.num_rows]))) / ((double)csr.num_nonzeros);
	if(nz_error >= .05)
		fprintf(stderr,"WARNING: Actual NNZ differs from Theoretical NNZ by %5.2f%%!\n",nz_error*100);
	csr.num_nonzeros = csr.Ap[csr.num_rows];
	fprintf(log,"Actual NUM_nonzeros: %d\n",csr.num_nonzeros);
	csr.density_perc = (((double)csr.num_nonzeros)*100.0)/((double)csr.num_cols)/((double)csr.num_rows);
	csr.density_ppm = (unsigned int)round(csr.density_perc * 10000.0);
	fprintf(log,"Actual Density: %u ppm = %g%%\n",csr.density_ppm,csr.density_perc);

	free(used_cols);
	csr.Ax = float_new_array(csr.num_nonzeros,"rand_csr() - Heap Overflow! Cannot Allocate Space for csr.Ax");
	for(i=0; i<csr.num_nonzeros; i++)
	{
		csr.Ax[i] = 1.0 - 2.0 * (rand() / (2147483647 + 1.0));
		while(csr.Ax[i] == 0.0)
			csr.Ax[i] = 1.0 - 2.0 * (rand() / (2147483647 + 1.0));
	}

	return csr;
}

void free_csr(csr_matrix* csr,const unsigned int num_csr)
{
	int k;
	for(k=0; k<num_csr; k++)
	{
		free(csr[k].Ap);
		free(csr[k].Aj);
		free(csr[k].Ax);
	}
	free(csr);
}


std_matrix rand_std_matrix(const unsigned int N, const unsigned int density, FILE* log)
{
	fprintf(log,"Creating Random Standard Matrix\n");
	std_matrix mat;

	mat.num_rows = N;
	mat.num_cols = N;
	mat.density_ppm = density;
	mat.num_nonzeros = (((double)(N*density))/1000000.0)*N;
	printf("NUM_nonzeros: %d\n",mat.num_nonzeros);

	mat.matrix = calloc(N * N, sizeof(float));
	check(mat.matrix != NULL, "sparse_formats.rand_std_matrix_bin_insertion(): eap Overflow - Cannot allocate memory for cstd.matrix\n");
	print_timestamp(log);
	fprintf(log, "Memory Allocated. Generating Data");

	for (int i = 0; i < mat.num_nonzeros; i++) {
		int index = gen_rand(0, (N*N)-1);
		if (!mat.matrix[index])
		{
			mat.matrix[index] = 1.0 - 2.0 * (rand() / (2147483647 + 1.0));
			while (mat.matrix[index] == 0.0)
				mat.matrix[index] = 1.0 - 2.0 * (rand() / (2147483647 + 1.0));
		} 
		else 
		{
			i--;
		}
	}
	print_timestamp(log);
	fprintf(log,"Matrix Completed. Returning...\n");

	return mat;
}



void print_std_matrix(const std_matrix* mat, FILE* stream)
{
	fprintf(stream, "\nPrinting Dense Matrix in Standard Form:\n\nNRows=%d\nNCols=%d\nDensity=1.0 (Dense Matrix)\n\n", 
		mat->num_rows, mat->num_cols);

	for (int i = 0; i < mat->num_rows; i++) {
		fprintf(stream, "[");
		for (int j = 0; j < mat->num_cols; j++) {
			int index = i * mat->num_cols + j;
			fprintf(stream, "%6.2f", mat->matrix[index]);
		}
		fprintf(stream, "]\n");
	}
	fprintf(stream, "\n");
}


void free_std(std_matrix * std, const unsigned int num_std)
{
	for (int i = 0; i < num_std; i++)
	{
		free(std[i].matrix);
	}
	free(std);
}

csr_matrix std_to_csr(const std_matrix* std, FILE* log)
{
	fprintf(log, "Converting matrix from standard to compressed sparse row\n");
	csr_matrix csr;
	csr.num_rows = std->num_rows;
	csr.num_cols = std->num_cols;
	csr.num_nonzeros = std->num_nonzeros;
	csr.density_ppm = std->density_ppm;

	csr.Ap = int_new_array(csr.num_rows+1,"sparse_formats.coo_to_csr() - Heap Overflow! Cannot allocate space for csr.Ap");
	csr.Aj = int_new_array(csr.num_nonzeros,"sparse_formats.coo_to_csr() - Heap Overflow! Cannot allocate space for csr.Aj");
	csr.Ax = float_new_array(csr.num_nonzeros,"sparse_formats.coo_to_csr() - Heap Overflow! Cannot allocate space for csr.Ax");
	
	print_timestamp(log);
	fprintf(log,"Memory Allocated. Copying column indices, values & row pointers...\n");

	int num_inputs = 0;
	for (int i = 0; i < std->num_rows; i++){
		csr.Ap[i] = num_inputs;
		for (int j = 0; j < std->num_cols; j++) {
			int index = i*std->num_cols + j;
			if (std->matrix[index]) {
				csr.Aj[num_inputs] = j;
				csr.Ax[num_inputs] = std->matrix[index];
				num_inputs ++;
			}
		}
	}
	csr.Ap[csr.num_rows] = num_inputs;

	print_timestamp(log);
	fprintf(log,"Conversion Complete. Returning...\n");

	return csr;
}



void free_coo(coo_matrix * coo, const unsigned int num_coo){
	for (int i = 0; i < num_coo; i++){
		free(coo[i].non_zero);
	}
	free(coo);
}


std_matrix csr_to_std(const csr_matrix* csr, FILE *log)
{
	fprintf(log, "Converting matrix from compressed sparse row to standard\n");

	std_matrix std;
	std.num_rows = csr->num_rows;
	std.num_cols = csr->num_cols;
	std.num_nonzeros = csr->num_nonzeros;
	std.density_ppm = csr->density_ppm;

	size_t bytes_needed = (size_t)std.num_rows * (size_t)std.num_cols * sizeof(float);
	fprintf(log, "Attempting to allocate %.2f GiB\n", (double)bytes_needed / (1 << 30));

	std.matrix = calloc (std.num_rows * std.num_cols, sizeof(float));
	check(std.matrix != NULL, "sparse_formats.csr_to_std_matrix_bin_insertion(): eap Overflow - Cannot allocate memory for cstd.matrix\n");

	print_timestamp(log);
	fprintf(log,"Memory Allocated. Copying column indices, values & row pointers...\n");



	for (int i = 0; i < csr->num_rows; i++)
	{
		for (int j = csr->Ap[i]; j < csr->Ap[i+1]; j++)
		{
			int index = i*std.num_cols + csr->Aj[j];
			std.matrix[index] = csr->Ax[j];
		}
	}

	print_timestamp(log);
	fprintf(log,"Conversion Complete. Returning...\n");

	return std;
}



coo_matrix csr_to_coo(const csr_matrix* csr, FILE *log)
{
	fprintf(log, "Converting matix from compressed sparse row to coordinate list\n");

	coo_matrix coo;
	coo.num_rows = csr->num_rows;
	coo.num_cols = csr->num_cols;
	coo.num_nonzeros = csr->num_nonzeros;
	coo.density_ppm = (long) csr->density_ppm;

	coo.non_zero = triplet_new_array(coo.num_nonzeros);

	print_timestamp(log);
	fprintf(log,"Memory Allocated. Copying column indices, values & row pointers...\n");

	int num_inputs = 0;
	for (int i = 0; i < csr->num_rows; i++)
	{
		for (int j = csr->Ap[i]; j < csr->Ap[i+1]; j++)
		{
			triplet* curr = &coo.non_zero[num_inputs];
			curr->i = i;
			curr->j = csr->Aj[j];
			curr->v = csr->Ax[j];
			num_inputs++;
		}
	}

	print_timestamp(log);
	fprintf(log,"Conversion Complete. Returning...\n");

	return coo;
}


vector vector_new(unsigned int length)
{
	vector vec;

	vec.length = length;
	vec.data = calloc(length, sizeof(float));

	return vec;
}


vector rand_vector(const unsigned int length, unsigned long* seed, FILE* log) {
    vector v;
    v.length = length;
    v.data = malloc(sizeof(float) * length);
    check(v.data != NULL, "rand_vector() - Heap Overflow! Cannot allocate space for vector data");

    if (log) fprintf(log, "Generating random vector of length %u with seed %lu\n", length, *seed);

    srand(*seed);

    for (unsigned int i = 0; i < length; ++i) {
        // Random float in range [-1.0, 1.0), nonzero
        v.data[i] = 1.0f - 2.0f * (rand() / (2147483647.0f + 1.0f));
        while (v.data[i] == 0.0f)
            v.data[i] = 1.0f - 2.0f * (rand() / (2147483647.0f + 1.0f));
    }

    return v;
}

void print_vector(const vector* v, FILE* stream){
	fprintf(stream, "vector of length %d\n", v->length);
	for (int i = 0; i < v->length; i++){
		fprintf(stream, "%6.2f\n", v->data[i]);
	}
}

void free_vector(vector* v, const unsigned int num_vecs){
	for (int i = 0; i < num_vecs; i++)
	{
		free(v[i].data);
	}
	free(v);
}



bool vector_is_equal(vector *v1, vector *v2, FILE *stream) {
    if (v1->length != v2->length) {
        fprintf(stream, "vector_is_equal(), v1 and v2 do not have the same length\n");
        return false;
    }

    int mismatch_index = -1;

    #pragma omp parallel for shared(mismatch_index) schedule(static)
    for (int i = 0; i < v1->length; i++) {
        if (mismatch_index != -1) continue; // early exit hint

        if (!AlmostEqualRelative(v1->data[i], v2->data[i])) {
            #pragma omp critical
            {
                if (mismatch_index == -1) {
                    mismatch_index = i;
                }
            }
        }
    }

    if (mismatch_index != -1) {
        fprintf(stream, "vector_is_equal(), at index %d, v1 (%6.2f) dne v2 (%6.2f)\n",
                mismatch_index, v1->data[mismatch_index], v2->data[mismatch_index]);
        return false;
    }

    return true;
}




coo_matrix load_matrix_market_to_coo(const char* filename, FILE* stream) {
    coo_matrix coo = {0}; // Ensure default initialization
    FILE* f = fopen(filename, "r");
    if (!f) {
        perror("Could not open Matrix Market file");
        return coo;
    }

    char line[1024];
    // Skip comments
    do {
        if (!fgets(line, sizeof(line), f)) {
            fprintf(stderr, "Failed to read Matrix Market header.\n");
            fclose(f);
            return coo;
        }
    } while (line[0] == '%');

    // Read dimensions
    unsigned int rows, cols, nnz;
    if (sscanf(line, "%u %u %u", &rows, &cols, &nnz) != 3) {
        fprintf(stderr, "%s: Invalid matrix size line.\n", filename);
        fclose(f);
        return coo;
    }

    coo.num_rows = rows;
    coo.num_cols = cols;
    coo.num_nonzeros = nnz;
    coo.non_zero = triplet_new_array(nnz);

    unsigned int i, j;
    double v;
	for (unsigned int idx = 0; idx < nnz; ++idx) {
		if (!fgets(line, sizeof(line), f)) {
			fprintf(stderr, "%s: Failed to read line %u of matrix.\n", filename, idx + 1);
			free(coo.non_zero);
			coo.non_zero = NULL;
			fclose(f);
			return coo;
		}

		int i, j;
		double v;
		int parsed = sscanf(line, "%d %d %lf", &i, &j, &v);
		if (parsed == 2) {
			v = 1.0; // Implicit binary entry
		} else if (parsed != 3) {
			fprintf(stderr, "%s: Invalid line %u: \"%s\"\n", filename, idx + 1, line);
			free(coo.non_zero);
			coo.non_zero = NULL;
			fclose(f);
			return coo;
		}

		coo.non_zero[idx].i = i - 1;
		coo.non_zero[idx].j = j - 1;
		coo.non_zero[idx].v = (float)v;
	}

    fclose(f);

	qsort(coo.non_zero, coo.num_nonzeros, sizeof(triplet), triplet_comparator);

    unsigned long long total = (unsigned long long)rows * cols;
    coo.density_ppm = (unsigned int)(((unsigned long long)nnz * 1000000ULL) / total);

    if (stream) {
        fprintf(stream, "%s: Loaded %u x %u matrix with %u nonzeros (%u ppm)\n",
                filename, coo.num_rows, coo.num_cols, coo.num_nonzeros, coo.density_ppm);
    }

    return coo;
}