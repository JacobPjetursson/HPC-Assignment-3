#include "init.h"
#include "datatools.h"


double *generateF(int N, double gridspacing) {
    double *f;
    cudaMallocHost((void**)&f,N*N* sizeof(double));
    double constant = gridspacing * gridspacing * 200;
    int i, j;
    for (i = 0; i < N; ++i) {
        for (j = 0; j < N; ++j) {
            f[i*N+j] = 0;
            if (gridspacing * j >= 1 && gridspacing * j <= 1 + 1.0 / 3 && gridspacing * i >= 1.0 / 3 &&
                gridspacing * i <= 2.0 / 3) {
                f[i*N+j] = constant;
            }
        }
    }

    return f;
}

double *generateU(int N) {
    double *u;
    cudaMallocHost((void**)&u,(N + 2)*(N + 2)* sizeof(double));
    int i, j;
    for (i = 0; i < N + 2; ++i) {
        for (j = 0; j < N + 2; ++j) {
            u[i*(N+2)+j] = (j == 0 || j == N + 1 || i == N + 1) ? 20.0 : 0.0;
        }
    }

    return u;
}
