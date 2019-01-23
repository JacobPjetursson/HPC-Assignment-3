void matmult_nat(int m, int n, int k, double *A, double *B, double *C);
void matmult_lib(int m, int n, int k, double *A, double *B, double *C);
void matmult_gpu1(int m, int n, int k, double *A, double *B, double *C);
__global__ void matmult_gpu1_kernel(int m, int n, int k, double *A, double *B, double *C);
void matmult_gpu2(int m, int n, int k, double *A, double *B, double *C);
__global__ void matmult_gpu2_kernel(int m, int n, int k, double *A, double *B, double *C);
void matmult_gpu3(int m, int n, int k, double *A, double *B, double *C);
__global__ void matmult_gpu3_kernel(int m, int n, int k, double *A, double *B, double *C);
void matmult_gpu4(int m, int n, int k, double *A, double *B, double *C);
__global__ void matmult_gpu4_kernel(int m, int n, int k, double *A, double *B, double *C);
void matmult_gpu5(int m, int n, int k, double *A, double *B, double *C);
__global__ void matmult_gpu5_kernel(int m, int n, int k, double *A, double *B, double *C);
void matmult_gpu6(int m, int n, int k, double *A, double *B, double *C);
void matmult_gpulib(int m, int n, int k, double *A, double *B, double *C);


#ifndef C_MATMULT_H
#define C_MATMULT_H

#endif //C_MATMULT_H
