<p align="center"> <h1 align="center">Vector Summation on the GPU</h1> </p>

In vector summation, we add corresponding elements of two arrays (vectors) and store the result in a third array. With GPU parallelism, each element of the result array can be computed simultaneously, making this task a perfect fit for CUDA.

***CUDA-Specific Code Structure for Vector Summation***

To implement vector summation on the GPU:

- Allocate memory for the vectors on both the CPU and GPU.
- Transfer data from CPU memory to GPU memory.
- Launch the kernel to perform parallel summation on the GPU.
- Transfer the result from GPU back to CPU memory.
- Free allocated memory on the GPU.

***Vector Summation Code Example***

The following code demonstrates these steps in CUDA C:

```
#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel for vector summation
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // Compute global thread index
    if (i < N) {
        C[i] = A[i] + B[i];  // Perform element-wise addition
    }
}

int main() {
    int N = 1 << 20;  // Define the size of the vectors (e.g., 2^20 elements)
    size_t size = N * sizeof(float);

    // Allocate memory on the host (CPU)
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // Initialize vectors A and B on the host
    for (int i = 0; i < N; ++i) {
        h_A[i] = i * 0.5f;
        h_B[i] = i * 0.3f;
    }

    // Allocate memory on the device (GPU)
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch the vector addition kernel with N threads
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy the result back to the host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify results
    for (int i = 0; i < 10; ++i) {
        printf("h_C[%d] = %f\n", i, h_C[i]);
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
```
**Kernel Function Definition: Vector Addition**

```
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // Compute global thread index
    if (i < N) {
        C[i] = A[i] + B[i];  // Perform element-wise addition
    }
}
```
- __global__ designates vectorAdd as a kernel function, which is executed on the GPU but called from the CPU.
- blockIdx.x, blockDim.x, and threadIdx.x together calculate the unique thread index i. Each thread computes a single element of the result array C.


**Host Code: Memory Allocation and Kernel Launch**

```
int N = 1 << 20;
size_t size = N * sizeof(float);
```
We define the number of elements, N, and calculate size in bytes for memory allocation.

```
float *h_A = (float *)malloc(size);
float *h_B = (float *)malloc(size);
float *h_C = (float *)malloc(size);
```
h_A, h_B, and h_C are pointers to host memory where we store vectors A, B, and C, respectively.

```
for (int i = 0; i < N; ++i) {
    h_A[i] = i * 0.5f;
    h_B[i] = i * 0.3f;
}
```
This loop initializes vectors h_A and h_B on the host.

**Device Memory Allocation and Data Transfer**

```float *d_A, *d_B, *d_C;
cudaMalloc((void **)&d_A, size);
cudaMalloc((void **)&d_B, size);
cudaMalloc((void **)&d_C, size);
```
Here, d_A, d_B, and d_C are pointers to device memory. cudaMalloc allocates memory for these vectors on the GPU.

```
cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
```
cudaMemcpy copies data from the host to the device (cudaMemcpyHostToDevice), transferring h_A and h_B to d_A and d_B.

**Launching the kernel**
```
int threadsPerBlock = 256;
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
```
- threadsPerBlock and blocksPerGrid are chosen to ensure that there are enough threads to handle N elements.
- We launch the kernel with blocksPerGrid blocks and threadsPerBlock threads per block, where each thread computes one element of C.

**Copying Results Back to Host and Cleanup**
```
cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
```
After kernel execution, we copy d_C (GPU result) to h_C (CPU) for verification.

```
for (int i = 0; i < 10; ++i) {
    printf("h_C[%d] = %f\n", i, h_C[i]);
}
```
We print the first 10 results to verify correctness.

```
cudaFree(d_A);
cudaFree(d_B);
cudaFree(d_C);
```
cudaFree releases GPU memory for d_A, d_B, and d_C.

```
free(h_A);
free(h_B);
free(h_C);
```
Finally, we free host memory allocated for h_A, h_B, and h_C.
