<p align="center"> <h1 align="center">2D Matrix Multiplication on the GPU</h1> </p>

In this exercise, you'll implement matrix multiplication using CUDA to calculate the product of two N×NN×N matrices on the GPU.


***Problem Overview***

Matrix multiplication involves multiplying each element of a row in matrix A by each element of a column in matrix B, then summing the results to find each element in the resulting matrix C.

Given two matrices A and B, each of size N×N:
```math
C[i][j]=\sum_{k=0}^{N−1}A[i][k]×B[k][j]
```
In CUDA, each thread will compute one element of the resulting matrix C, leveraging parallel computation to perform this operation efficiently.



***Task 1: Complete the CUDA Kernel***

- Implement the matrixMulGPU CUDA kernel that performs matrix multiplication of A and B to populate the matrix C.
- In the kernel, each thread should calculate the value of a single element C[i][j] by summing over the corresponding elements in A and B.

**Hints:**

- Use threadIdx and blockIdx to identify the unique row and column assigned to each thread.
- Each thread should iterate through the elements in row i of A and column j of B to compute the sum.


***Task 2: Configure CUDA Grid and Block Dimensions***

- Define a 2D grid of threads, where each thread corresponds to an element in the result matrix C.
- Set up appropriate dim3 variables for threads_per_block and number_of_blocks to process the entire N×N matrix.

**Hints:**

- Consider setting threads_per_block as 16×16 (or another power of 2), creating blocks of 256 threads.
- number_of_blocks should be set such that the grid covers all N×N elements.


***Provided Code Template***

The following code is mostly complete. Your task is to fill in the matrixMulGPU kernel function and the grid/block dimensions.

```
#include <stdio.h>

#define N 64

// Kernel to perform matrix multiplication on the GPU
__global__ void matrixMulGPU(int *a, int *b, int *c) {
  // FIX ME: Identify row and column for each thread and calculate the element of C
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < N && col < N) {
    int value = 0;
    for (int k = 0; k < N; ++k) {
      value += a[row * N + k] * b[k * N + col];
    }
    c[row * N + col] = value;
  }
}

// CPU function to perform matrix multiplication (for verification)
void matrixMulCPU(int *a, int *b, int *c) {
  int val = 0;
  for(int row = 0; row < N; ++row) {
    for(int col = 0; col < N; ++col) {
      val = 0;
      for (int k = 0; k < N; ++k)
        val += a[row * N + k] * b[k * N + col];
      c[row * N + col] = val;
    }
  }
}

int main() {
  int *a, *b, *c_cpu, *c_gpu;
  int size = N * N * sizeof(int);

  // Allocate unified memory accessible from CPU and GPU
  cudaMallocManaged(&a, size);
  cudaMallocManaged(&b, size);
  cudaMallocManaged(&c_cpu, size);
  cudaMallocManaged(&c_gpu, size);

  // Initialize matrices
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {
      a[row * N + col] = row;
      b[row * N + col] = col + 2;
      c_cpu[row * N + col] = 0;
      c_gpu[row * N + col] = 0;
    }
  }

  // FIX ME: Set up the grid and block dimensions
  dim3 threads_per_block(16, 16);
  dim3 number_of_blocks((N + threads_per_block.x - 1) / threads_per_block.x,
                        (N + threads_per_block.y - 1) / threads_per_block.y);

  // Launch kernel
  matrixMulGPU<<<number_of_blocks, threads_per_block>>>(a, b, c_gpu);

  // Synchronize to ensure completion
  cudaDeviceSynchronize();

  // Run CPU version for verification
  matrixMulCPU(a, b, c_cpu);

  // Verify that the GPU result matches the CPU result
  bool error = false;
  for (int row = 0; row < N && !error; ++row) {
    for (int col = 0; col < N && !error; ++col) {
      if (c_cpu[row * N + col] != c_gpu[row * N + col]) {
        printf("Error at c[%d][%d]\n", row, col);
        error = true;
        break;
      }
    }
  }
  if (!error) {
    printf("Success! GPU matrix multiplication is correct.\n");
  }

  // Free allocated memory
  cudaFree(a);
  cudaFree(b);
  cudaFree(c_cpu);
  cudaFree(c_gpu);

  return 0;
}
```
