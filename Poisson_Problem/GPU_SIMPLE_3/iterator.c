#include <math.h>
#include "iterator.h"
#include <stdio.h>
#include <omp.h>

__global__ void jacobiIteration(double *u, double *uold, double *f, int N) {
    int i, j;
    for (i = 1; i < N + 1; ++i) {
        for (j = 1; j < N + 1; ++j) {
            u[i*(N+2)+j] = 0.25 * (uold[i*(N+2)+j-1] + uold[i*(N+2)+j+1] + uold[(i-1)*(N+2)+j] + uold[(i+1)*(N+2)+j] + f[(i-1)*(N)+j-1]);
        }
    }
}

__global__ void jacobiIteration_per_elem(double *u, double *uold, double *f, int N) {
    int i = blockDim.y * blockIdx.y + threadIdx.y + 1;
    int j = blockDim.x * blockIdx.x + threadIdx.x + 1;
    if (i*(N+2)+j < (N + 2) * (N + 2)){
		u[i*(N+2)+j] = 0.25 * (uold[i*(N+2)+j-1] + uold[i*(N+2)+j+1] + uold[(i-1)*(N+2)+j] + uold[(i+1)*(N+2)+j] + f[(i-1)*(N)+j-1]);
    }
}

__global__ void jacobiIteration_per_elem_2(double *u, double *uold, double *uold_2, double *f, int N, int device) {
    int i,j;

    if (device == 0){
        j = blockDim.x * blockIdx.x + threadIdx.x + 1;
        i = blockDim.y * blockIdx.y + threadIdx.y + 1;
        if (i == N / 2){
            u[i*(N+2)+j] = 0.25 * (uold[i*(N+2)+j-1] + uold[i*(N+2)+j+1] + uold[(i-1)*(N+2)+j] + uold_2[j] + f[(i-1)*(N)+j-1]);
        } else {
            u[i*(N+2)+j] = 0.25 * (uold[i*(N+2)+j-1] + uold[i*(N+2)+j+1] + uold[(i-1)*(N+2)+j] + uold[(i+1)*(N+2)+j] + f[(i-1)*(N)+j-1]);
        }
    } else if (device == 1) {
        j = blockDim.x * blockIdx.x + threadIdx.x + 1;
        i = blockDim.y * blockIdx.y + threadIdx.y;        
        if (i == 0){
            u[i*(N+2)+j] = 0.25 * (uold[i*(N+2)+j-1] + uold[i*(N+2)+j+1] + uold_2[(N/2)*(N+2)+j] + uold[(i+1)*(N+2)+j] + f[(i-1+N/2)*(N)+j-1]);
        } else {
            u[i*(N+2)+j] = 0.25 * (uold[i*(N+2)+j-1] + uold[i*(N+2)+j+1] + uold[(i-1)*(N+2)+j] + uold[(i+1)*(N+2)+j] + f[(i-1+N/2)*(N)+j-1]);
        }
    }
    
}




