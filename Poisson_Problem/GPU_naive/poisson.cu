#include <stdio.h>
#include <float.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include "iterator.h"
#include "init.h"
#include "datatools.h"
#include <math.h>
#include <omp.h>


int jacobi(int N, int kmax, double **u, double **f) {
    double *uold;
    int size_N = N *  N * sizeof(double);
    int size_N2 = (N+2)*(N+2)*sizeof(double);

    int k = 0;

    cudaMallocHost((void**) &uold, size_N2);
    for (int i = 0; i <(N+2)*(N+2) ; ++i) {
        uold[i]=(*u)[i];
    }
    double *u_d;
    double *uold_d;
    double *f_d;

    cudaMalloc((void**) &f_d,size_N);
    cudaMalloc((void**) &uold_d, size_N2);
    cudaMalloc((void**) &u_d, size_N2);
    cudaMemcpy(f_d, (*f), size_N, cudaMemcpyHostToDevice);
    cudaMemcpy(u_d, (*u), size_N2,cudaMemcpyHostToDevice);
    cudaMemcpy(uold_d, uold, size_N2,cudaMemcpyHostToDevice);

    int g = N / 32;

    dim3 dimGrid(g, g, 1);
    dim3 dimBlock(32, 32, 1);

    while (k < kmax) {
        swap(&u_d, &uold_d);
        k += 1;
        //jacobiIteration<<< 1, 1>>>(u_d, uold_d, f_d, N);
        jacobiIteration_per_elem<<< dimGrid, dimBlock>>>(u_d, uold_d, f_d, N);
        cudaDeviceSynchronize();
    }

    cudaMemcpy((*u), u_d, size_N2, cudaMemcpyDeviceToHost);

    cudaFree(uold_d);
    cudaFree(u_d);
    cudaFree(f_d);
    cudaFreeHost(uold);
    return k;
}

int main(int argc, char *argv[]) {

    double *dummy_d;
    cudaMalloc((void**) &dummy_d, 0);

    int N;
    int kmax;
    char *funcType;
    double mflops, memory;
    long ts, te;
    struct timeval timecheck;
    int iterations;

    N = 32;
    kmax = 100;
    funcType = "jacobi";

    // command line arguments for the three sizes above
    if (argc >= 2)
        funcType = argv[1];

    if (argc >= 3)
        N = atoi(argv[2]);

    if (argc >= 4)
        kmax = atoi(argv[3]);


    double gridspacing = (double) 2 / (N + 1);

    double *f;
    double *u;
    f = generateF(N, gridspacing);
    u = generateU(N);

    gettimeofday(&timecheck, NULL);
    ts = (long) timecheck.tv_sec * 1000 + (long) timecheck.tv_usec / 1000;
    if (strcmp(funcType, "jacobi") == 0) {
        memory = ((N + 2) * (N + 2) * 2 + (N * N)) * sizeof(double);
        iterations = jacobi(N, kmax, &u, &f);
    } else {
        printf("First parameter should be either jacobi or gauss");
        exit(1);
    }

    // Get elapsed time
    gettimeofday(&timecheck, NULL);
    te = (long) timecheck.tv_sec * 1000 + (long) timecheck.tv_usec / 1000;
    double elapsed = (double) (te - ts) / 1000;

    mflops = 1.0e-06 * iterations * (N * N * FLOP);
    mflops /= elapsed;
    memory /= 1024.0; // KB

    printf("%f\t", mflops); // MFLOP/S
    printf("%.3f\t", elapsed); // Time spent
    printf("%f\t", memory); // Mem footprint
    printf("%d\t", iterations); // iterations
    printf("%d\n", N); // N
    //print_matrix(u, N + 2);
    return 0;
}
