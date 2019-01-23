extern "C" {
#include "matmult.h"
#include <cblas.h>
}
#include <helper_cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define bsx 1
#define bsy 16
// Thread block size for shared memory
#define BLOCK_SIZE 16

extern "C" {
    void matmult_lib(int m, int n, int k, double *A, double *B, double *C) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, A, k, B, n, 0, C, n);
        for (int x = 0; x < m * n; x++)
            printf("%f\n", C[x]);
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

    for (l = 0; l < k; ++l) {
        C[row * n + col] += A[row * k + l] * B[l * n + col];
    }

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
    if (row >= m - 1) {
        for (l = 0; l < k; ++l) {
            C[row * n + col] += A[row * k + l] * B[l * n + col];
        }
    } else {
        for (l = 0; l < k; ++l) {
            C[row * n + col] += A[row * k + l] * B[l * n + col];
            C[(row+1) * n + col] += A[(row+1) * k + l] * B[l * n + col];
        }
    }

    /*
    int col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= n || row >= m)
        return;
    int l;
    if (col >= n - 1) {
        for (l = 0; l < k; ++l) {
            C[row * n + col] += A[row * k + l] * B[l * n + col];
        }
    } else {
        for (l = 0; l < k; ++l) {
            C[row * n + col] += A[row * k + l] * B[l * n + col];
            C[row * n + col + 1] += A[row * k + l] * B[l * n + col + 1];
        }
    }
     */
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
    //for (int i = 0; i < m * n; i++)
    //    printf("%f\n", C[i]);
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

    if (row >= (m - bsy + 1) && col >= (n - bsx + 1)) {
        for (j = 0; j < (n - col); j++) {
            for (i = 0; i < (m - row); i++) {
                for (l = 0; l < k; ++l) {
                    C_reg[i * (n - col) + j] += A[(row+i) * k + l] * B[l * n + (col+j)];
                }
                C[(row+i) * n + (col+j)] = C_reg[i * (n - col) + j];
            }
        }
    } else if (row >= m - bsy + 1) {
        for (i = 0; i < (m - row); i++) {
            for (l = 0; l < k; ++l) {
                C_reg[i] += A[(row+i) * k + l] * B[l * n + col];
            }
            C[(row+i) * n + col] = C_reg[i];
        }

    } else if (col >= n - bsx + 1) {
        for (j = 0; j < (n - col); j++) {
            for (l = 0; l < k; ++l) {
                C_reg[j] += A[row * k + l] * B[l * n + col + j];
            }
            C[row * n + col + j] = C_reg[j];
        }
    }
    else {
        for (j = 0; j < bsx; j++) {
            for (i = 0; i < bsy; i++) {
                for (l = 0; l < k; ++l) {
                    C_reg[i * bsx + j] += A[(row+i) * k + l] * B[l * n + col + j];
                }
                C[(row+i) * n + col+j] = C_reg[i * bsx + j];
            }
        }
    }
}

__device__ double* GetSubMatrix_ours(double *A, int row, int col, int width)
{
    double *Asub = &A[width * BLOCK_SIZE * row
                            + BLOCK_SIZE * col];
    return Asub;
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

    double *Csub = GetSubMatrix_ours(C, blockRow, blockCol, n);
    double Cvalue = 0;

    int row = threadIdx.y;
    int col = threadIdx.x;

    for (int l = 0; l < (k / BLOCK_SIZE); ++l) {
        double *Asub = GetSubMatrix_ours(A, blockRow, l, k);
        double *Bsub = GetSubMatrix_ours(B, l, blockCol, n);

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
