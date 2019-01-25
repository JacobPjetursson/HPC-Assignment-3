extern "C" {
#include "matmult.h"
#include <cblas.h>
}
#include <helper_cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <cublas_v2.h>
#include <omp.h>

// Block size when blocking for registers
#define bsx 1
#define bsy 16
// Thread block size for shared memory
#define BLOCK_SIZE 16
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

extern "C" {
    void matmult_lib(int m, int n, int k, double *A, double *B, double *C) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, A, k, B, n, 0, C, n);
    }
}

void matmult_gpu1(int m, int n, int k, double *A, double *B, double *C) {
    double *d_A, *d_B, *d_C;
    for(int i = 0; i < m * n; i++)
        C[i] = 0;

    cudaMalloc( (void **)&d_A, m * k * sizeof(double));
    cudaMalloc( (void **)&d_B, n * k * sizeof(double));
    cudaMalloc( (void **)&d_C, m * n * sizeof(double));
    cudaMemcpy(d_A, A, m * k * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n * k * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, m * n * sizeof(double), cudaMemcpyHostToDevice);

    matmult_gpu1_kernel<<<1, 1>>>(m, n, k, d_A, d_B, d_C);
    cudaDeviceSynchronize();

    cudaMemcpy(C, d_C, m * n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);


}

__global__
void matmult_gpu1_kernel(int m, int n, int k, double *A, double *B, double *C) {
    int i, j, l;
    for (i = 0; i < m; ++i) {
        for (l = 0; l < k; ++l) {
            for (j = 0; j < n; ++j) {
                C[i * n + j] += A[i * k + l] * B[l * n + j];
            }
        }
    }
}

void matmult_gpu2(int m, int n, int k, double *A, double *B, double *C) {
    double *d_A, *d_B, *d_C;
    for(int i = 0; i < m * n; i++)
        C[i] = 0;
    cudaMalloc( (void **)&d_A, m * k * sizeof(double));
    cudaMalloc( (void **)&d_B, n * k * sizeof(double));
    cudaMalloc( (void **)&d_C, m * n * sizeof(double));
    cudaMemcpy(d_A, A, m * k * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n * k * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, m * n * sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = 16; // NUM_THREADS_IN_BLOCK
    int gridN = (int)ceil((double)n / blockSize);
    int gridM = (int)ceil((double)m / blockSize);
    dim3 dimGrid(gridN,gridM,1);
    dim3 dimBlock(blockSize, blockSize, 1);

    matmult_gpu2_kernel<<<dimGrid, dimBlock>>>(m, n, k, d_A, d_B, d_C);
    cudaDeviceSynchronize();
    cudaMemcpy(C, d_C, m * n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

__global__ void matmult_gpu2_kernel(int m, int n, int k, double *A, double *B, double *C) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= n || row >= m)
        return;
    int l;
    double C_reg = 0.0;
    for (l = 0; l < k; ++l) {
        C_reg += A[row * k + l] * B[l * n + col];
    }
    C[row * n + col] = C_reg;

}

void matmult_gpu3(int m, int n, int k, double *A, double *B, double *C) {
    double *d_A, *d_B, *d_C;
    for(int i = 0; i < m * n; i++)
        C[i] = 0;
    cudaMalloc( (void **)&d_A, m * k * sizeof(double));
    cudaMalloc( (void **)&d_B, n * k * sizeof(double));
    cudaMalloc( (void **)&d_C, m * n * sizeof(double));
    cudaMemcpy(d_A, A, m * k * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n * k * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, m * n * sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = 16; // NUM_THREADS_IN_BLOCK
    int gridN = (int)ceil((double)n / blockSize);
    int gridM = (int)ceil((double)m / blockSize * 0.5);
    dim3 dimGrid(gridN,gridM,1);
    dim3 dimBlock(blockSize, blockSize, 1);

    matmult_gpu3_kernel<<<dimGrid, dimBlock>>>(m, n, k, d_A, d_B, d_C);
    cudaDeviceSynchronize();
    cudaMemcpy(C, d_C, m * n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

__global__ void matmult_gpu3_kernel(int m, int n, int k, double *A, double *B, double *C) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
    if (col >= n || row >= m)
        return;
    int l;
    double C_reg[2] = {0.0, 0.0};

    if (row >= m - 1) {
        for (l = 0; l < k; ++l) {
            C_reg[0] += A[row * k + l] * B[l * n + col];
        }
        C[row * n + col] = C_reg[0];

    } else {
        for (l = 0; l < k; ++l) {
            C_reg[0] += A[row * k + l] * B[l * n + col];
            C_reg[1] += A[(row+1) * k + l] * B[l * n + col];
        }
        C[row * n + col] = C_reg[0];
        C[(row+1) * n + col] = C_reg[1];
    }

}

void matmult_gpu4(int m, int n, int k, double *A, double *B, double *C) {
    cudaSetDevice(1);
    double *d_A, *d_B, *d_C;
    for(int i = 0; i < m * n; i++)
        C[i] = 0;
    cudaMalloc( (void **)&d_A, m * k * sizeof(double));
    cudaMalloc( (void **)&d_B, n * k * sizeof(double));
    cudaMalloc( (void **)&d_C, m * n * sizeof(double));
    cudaMemcpy(d_A, A, m * k * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n * k * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, m * n * sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = 16; // NUM_THREADS_IN_BLOCK
    int gridN = (int)ceil((double)n / blockSize / bsx);
    int gridM = (int)ceil((double)m / blockSize / bsy);
    dim3 dimGrid(gridN,gridM,1);
    dim3 dimBlock(blockSize, blockSize, 1);

    matmult_gpu4_kernel<<<dimGrid, dimBlock>>>(m, n, k, d_A, d_B, d_C);
    cudaDeviceSynchronize();
    cudaMemcpy(C, d_C, m * n * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

__global__ void matmult_gpu4_kernel(int m, int n, int k, double *A, double *B, double *C) {
    int col = (blockIdx.x * blockDim.x + threadIdx.x) * bsx;
    int row = (blockIdx.y * blockDim.y + threadIdx.y) * bsy;
    if (col >= n || row >= m)
        return;

    double C_reg[bsx * bsy];
    int l, i, j;
    for (i = 0; i < bsx * bsy; i++)
        C_reg[i] = 0.0;

    int loopJ = MIN(bsx, (n-col));
    int loopI = MIN(bsy, (m-row));
    for (j = 0; j < loopJ; j++) {
        for (i = 0; i < loopI; i++) {
            for (l = 0; l < k; ++l) {
                C_reg[i * bsx + j] += A[(row+i) * k + l] * B[l * n + col + j];
            }
            C[(row+i) * n + col+j] = C_reg[i * bsx + j];
        }
    }

}

void matmult_gpu5(int m, int n, int k, double *A, double *B, double *C)
{
    double *d_A, *d_B, *d_C;
    for(int i = 0; i < m * n; i++)
        C[i] = 0;
    cudaMalloc( (void **)&d_A, m * k * sizeof(double));
    cudaMalloc( (void **)&d_B, n * k * sizeof(double));
    cudaMalloc( (void **)&d_C, m * n * sizeof(double));
    cudaMemcpy(d_A, A, m * k * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n * k * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, m * n * sizeof(double), cudaMemcpyHostToDevice);

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(n / dimBlock.x, m / dimBlock.y);
    matmult_gpu5_kernel<<<dimGrid, dimBlock>>>(m, n, k, d_A, d_B, d_C);

    cudaDeviceSynchronize();
    cudaMemcpy(C, d_C, m * n * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

__global__ void matmult_gpu5_kernel(int m, int n, int k, double *A, double *B, double *C)
{
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    double *Csub = &C[n * BLOCK_SIZE * blockRow + BLOCK_SIZE * blockCol];
    double Cvalue = 0;

    int row = threadIdx.y;
    int col = threadIdx.x;

    for (int l = 0; l < (k / BLOCK_SIZE); ++l) {
        double *Asub = &A[k * BLOCK_SIZE * blockRow + BLOCK_SIZE * l];
        double *Bsub = &B[n * BLOCK_SIZE * l + BLOCK_SIZE * blockCol];

        __shared__ double As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE];

        As[row][col] = Asub[row * k + col];
        Bs[row][col] = Bsub[row * n + col];

        __syncthreads();

        for (int i = 0; i < BLOCK_SIZE; ++i)
            Cvalue += As[row][i] * Bs[i][col];
        __syncthreads();
    }

    Csub[row * m + col] = Cvalue;
}

void matmult_gpulib(int m, int n, int k, double *A, double *B, double *C)
{
    double *d_A, *d_B, *d_C;
    for(int i = 0; i < m * n; i++)
        C[i] = 0;
    cudaMalloc( (void **)&d_A, m * k * sizeof(double));
    cudaMalloc( (void **)&d_B, n * k * sizeof(double));
    cudaMalloc( (void **)&d_C, m * n * sizeof(double));
    cudaMemcpy(d_A, A, m * k * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n * k * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, m * n * sizeof(double), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);
    double alpha = 1.0;
    double beta = 0.0;
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_B, n, d_A, k, &beta, d_C, n);

    cudaMemcpy(C, d_C, m * n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
}
