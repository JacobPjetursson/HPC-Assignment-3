#include "datatools.h"
#include <stdio.h>
#include <stdlib.h>

void swap(double **m1, double **m2) {
    double *tmp;
    tmp = *m1;
    *m1 = *m2;
    *m2 = tmp;
}

void print_matrix(double * M, int n){
    printf("\n");
    for (int i = 0; i <n ; ++i) {
        for (int j = 0; j <n ; ++j) {
            printf("%1.2f\t",M[i*n+j]);
        }
        printf("\n");
    }
}
