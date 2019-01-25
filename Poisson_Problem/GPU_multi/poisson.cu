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

    int elem_f = N * N;
    int size_N = elem_f * sizeof(double);

    int elem_u = (N + 2) * (N + 2);
    int size_N2 = elem_u*sizeof(double);


    int k = 0;

    cudaMallocHost((void**) &uold, size_N2);
    for (int i = 0; i <(N+2)*(N+2) ; ++i) {
        uold[i]=(*u)[i];
    }
    double *f_d1;
    double *f_d2;

    double *u_d1;
    double *uold_d1;

    double *u_d2;
    double *uold_d2;

    cudaSetDevice(0);
    cudaDeviceEnablePeerAccess(1, 0); 
    cudaMalloc((void**) &f_d1,size_N);
    cudaMalloc((void**) &uold_d1, size_N2 / 2);
    cudaMalloc((void**) &u_d1, size_N2 / 2);

    cudaMemcpy(f_d1, (*f), size_N, cudaMemcpyHostToDevice);
    cudaMemcpy(u_d1, (*u), size_N2 / 2, cudaMemcpyHostToDevice);
    cudaMemcpy(uold_d1, uold, size_N2 / 2, cudaMemcpyHostToDevice);

    cudaSetDevice(1);
    cudaDeviceEnablePeerAccess(0, 0); 
    cudaMalloc((void**) &f_d2, size_N);
    cudaMalloc((void**) &uold_d2, size_N2 / 2);
    cudaMalloc((void**) &u_d2, size_N2 / 2);

    cudaMemcpy(f_d2, (*f), size_N, cudaMemcpyHostToDevice);
    cudaMemcpy(u_d2, (*u) + elem_u / 2, size_N2 / 2, cudaMemcpyHostToDevice);
    cudaMemcpy(uold_d2, uold + elem_u / 2, size_N2 / 2, cudaMemcpyHostToDevice);

    int g = N / 32;
    int g2 = N / 64;

    dim3 dimGrid(g, g2, 1);
    dim3 dimBlock(32, 32, 1);

    while (k < kmax) {
        swap(&u_d1, &uold_d1);
        swap(&u_d2, &uold_d2);
        k += 1;
        //jacobiIteration<<< 1, 1>>>(u_d, uold_d, f_d, N);
        cudaSetDevice(0);
        jacobiIteration_per_elem_2<<< dimGrid, dimBlock>>>(u_d1, uold_d1, uold_d2, f_d1, N, 0);
        cudaSetDevice(1);
        jacobiIteration_per_elem_2<<< dimGrid, dimBlock>>>(u_d2, uold_d2, uold_d1, f_d2, N, 1);

        //cudaSetDevice(0);
        //cudaDeviceSynchronize();
        cudaSetDevice(1);
        cudaStreamSynchronize(0);
    }

    cudaSetDevice(0);
    cudaMemcpy((*u), u_d1, size_N2 / 2, cudaMemcpyDeviceToHost);
    cudaFree(uold_d1);
    cudaFree(u_d1);
    cudaFree(f_d1);

    cudaSetDevice(1);
    cudaMemcpy((*u) + elem_u / 2, u_d2, size_N2 / 2, cudaMemcpyDeviceToHost);
    cudaFree(uold_d2);
    cudaFree(u_d2);
    cudaFree(f_d2);
    

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
