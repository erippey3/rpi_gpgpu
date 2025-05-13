#include "openmp.h"
#include "err_code.h"
#include "sparse_formats.h"
#include "wtime.h"
#include <omp.h>


vector *openmp_csr_spmv(const csr_matrix * m, const vector * v, benchmark *b){
    check(m->num_cols == v->length, "spmv_serial.serial_csr_spmv(): matrix rows does not equal vector length\n");
    double start_time;
    vector * result = (vector *) malloc(sizeof(vector));
    *result = vector_new(m->num_rows);

    if (b)
        start_time = wtime();

    #pragma omp parallel for schedule(static) 
    for (int i = 0; i < m->num_rows; i++) {
        // don't unroll or else you will have to synchronize accesses to result
        float sum = 0;
        for (int j = m->Ap[i]; j < m->Ap[i+1]; j++){
            sum += m->Ax[j] * v->data[m->Aj[j]];
        }
        result->data[i] = sum;
    }

    if (b) 
    {
        double delta_time = wtime() - start_time;
        add_runtime(b, delta_time);
    }

    return result;
}



vector *openmp_coo_spmv(const coo_matrix * m, const vector * v, benchmark *b){
    check(m->num_cols == v->length, "spmv_serial.serial_coo_spmv(): matrix rows does not equal vector length\n");
    double start_time;
    vector * result = (vector *) malloc(sizeof(vector));
    *result = vector_new(m->num_rows);

    if (b)
        start_time = wtime();


    #pragma omp parallel
    {
        int curr_row = 0;
        float sum = 0;

        #pragma omp for
        for (int i = 0; i < m->num_nonzeros; i++) {
            triplet curr = m->non_zero[i];
            
            if (curr.i == curr_row)
                sum += curr.v * v->data[curr.j];
            else {
                #pragma omp atomic
                result->data[curr_row] += sum;

                sum = curr.v * v->data[curr.j];
                curr_row = curr.i;
            }
        }


        #pragma omp atomic
        result->data[curr_row] += sum;
    }

    if (b) 
    {
        double delta_time = wtime() - start_time;
        add_runtime(b, delta_time);
    }

    return result;
}