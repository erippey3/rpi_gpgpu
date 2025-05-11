#include "serial.h"
#include "err_code.h"
#include "wtime.h"

vector *serial_csr_spmv(const csr_matrix * m, const vector * v, benchmark *b){
    check(m->num_cols == v->length, "spmv_serial.serial_csr_spmv(): matrix rows does not equal vector length\n");
    double start_time;
    vector * result = (vector *) malloc(sizeof(vector));
    *result = vector_new(m->num_rows);

    if (b)
        start_time = wtime();

    for (int i = 0; i < m->num_rows; i++) {
        for (int j = m->Ap[i]; j < m->Ap[i+1]; j++){
            result->data[i] += m->Ax[j] * v->data[m->Aj[j]];
        }
    }

    if (b) 
    {
        double delta_time = wtime() - start_time;
        add_runtime(b, delta_time);
    }

    return result;
}




vector *serial_std_spmv(const std_matrix * m, const vector * v, benchmark *b){
    check(m->num_cols == v->length, "spmv_serial.serial_std_spmv(): matrix rows does not equal vector length\n");

}



vector *serial_coo_spmv(const coo_matrix * m, const vector * v, benchmark *b){
    check(m->num_cols == v->length, "spmv_serial.serial_coo_spmv(): matrix rows does not equal vector length\n");
}