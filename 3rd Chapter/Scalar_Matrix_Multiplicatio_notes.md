<p align="center"> <h1 align="center">Steps to Implement Scalar-Matrix Multiplication in CUDA C</h1> </p>

***Understand the Problem***

Given:

- A scalar ss.
- A matrix A with dimensions N×M

We need to compute a new matrix C, where each element of C is calculated as:
```math
C[i][j]=s×A[i][j]
```

***Define Kernel to Perform Scalar Multiplication***

CUDA uses a special function called a kernel to execute code on the GPU. For our scalar-matrix multiplication, we’ll define a kernel function that:

- Receives the matrix A and the scalar s.
- Computes each element of the output matrix CC as s×A[i][j].


***Write the Code***

Here’s a complete CUDA C code example for scalar-matrix multiplication.

```sh
#include <stdio.h>
#include <cuda_runtime.h>

// Define the matrix dimensions (N x M)
#define N 1024  // Number of rows
#define M 1024  // Number of columns

// Kernel function to perform scalar multiplication on each element of the matrix
__global__ void scalarMultiply(float *matrix, float scalar, int rows, int cols) {
    // Calculate the row and column index of the element
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the thread is within matrix bounds
    if (row < rows && col < cols) {
        // Perform scalar multiplication
        int idx = row * cols + col;
        matrix[idx] *= scalar;
    }
}

int main() {
    // Scalar value for multiplication
    float scalar = 5.0f;

    // Allocate memory for the matrix on the host (CPU)
    size_t size = N * M * sizeof(float);
    float *h_matrix = (float*) malloc(size);

    // Initialize the matrix with some values
    for (int i = 0; i < N * M; i++) {
        h_matrix[i] = 1.0f; // Just for simplicity, set all elements to 1.0
    }

    // Allocate memory on the device (GPU)
    float *d_matrix;
    cudaMalloc(&d_matrix, size);

    // Copy the matrix from host memory to device memory
    cudaMemcpy(d_matrix, h_matrix, size, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 threadsPerBlock(16, 16);  // Block of 16x16 threads
    dim3 blocksPerGrid((M + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the scalar multiplication kernel on the GPU
    scalarMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_matrix, scalar, N, M);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Copy the result back to the host
    cudaMemcpy(h_matrix, d_matrix, size, cudaMemcpyDeviceToHost);

    // Verify the results
    bool success = true;
    for (int i = 0; i < N * M; i++) {
        if (h_matrix[i] != 5.0f) {  // Each element should now be 5.0
            printf("Mismatch at index %d: %f\n", i, h_matrix[i]);
            success = false;
            break;
        }
    }
    if (success) printf("Scalar multiplication successful!\n");

    // Free device and host memory
    cudaFree(d_matrix);
    free(h_matrix);

    return 0;
}
```

***Explanation of the Code***

 **Kernel Function (scalarMultiply):**
- The kernel function scalarMultiply performs the scalar-matrix multiplication. It calculates a unique index for each thread based on blockIdx, threadIdx, blockDim, and uses this index to multiply each element in the matrix by the scalar.

   **Setting up Host and Device Memory:**
  - h_matrix is allocated on the CPU, and values are initialized.
  - cudaMalloc allocates memory for d_matrix on the GPU.
  - cudaMemcpy transfers data from the CPU to the GPU.

**Grid and Block Configuration:**
  - We define threadsPerBlock as 16×1616×16, creating a 2D block of threads.
- blocksPerGrid calculates the number of blocks needed to cover the matrix dimensions.

**Kernel Launch:**
- scalarMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_matrix, scalar, N, M); launches the kernel on the GPU.

**Copying Data Back:**
- After GPU execution, cudaMemcpy copies the results from d_matrix back to h_matrix on the host.

 **Verification and Cleanup:**
- A verification loop checks if each element was correctly multiplied by the scalar. Then, both GPU and CPU memory are freed.
