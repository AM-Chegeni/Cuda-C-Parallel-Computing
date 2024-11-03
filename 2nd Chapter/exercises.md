<p align="center"> <h1 align="center">Vector Initialization and Summation with Managed Memory and Prefetching</h1> </p>

Implement and understand managed memory allocation and memory prefetching in CUDA to simplify data management across CPU and GPU. You will initialize two vectors with specific values and sum them in parallel using CUDA, leveraging managed memory.

**Instructions:**

- ***Set up managed memory:*** Allocate memory for three vectors AA, BB, and CC of size NN using cudaMallocManaged. Managed memory allows seamless memory access from both the CPU and GPU.
- ***Initialize values in vectors AA and BB:*** Write a kernel function initWith that initializes all elements of a vector to a given value, e.g., 3 for AA and 4 for BB.
- ***Prefetch memory:*** Use cudaMemPrefetchAsync to move each vector to the GPU, preparing them for processing.
- ***Perform vector addition:*** Write a kernel addVectorsInto that adds elements from AA and BB and stores the result in CC.
- ***Verify the result:*** Write a host function checkElementsAre that checks if all elements of CC have the expected value (7). Print a success message if the values are correct.
- ***Clean up memory:*** Free the memory for vectors AA, BB, and CC using cudaFree.

**Code Skeleton:**

Here’s a starting code structure to guide your implementation:

```sh

#include <stdio.h>

__global__ void initWith(float num, float *a, int N) {
  // Your code for initializing the vector here
}

__global__ void addVectorsInto(float *result, float *a, float *b, int N) {
  // Your code for vector addition here
}

void checkElementsAre(float target, float *vector, int N) {
  // Your code for result verification here
}

int main() {
  // Device and memory setup, kernel launches, error checking, and cleanup code here
}
```














<p align="center"> <h1 align="center">Vector Summation with Multiple Kernel Launches</h1> </p>

Implement vector summation with multiple kernel launches to compute the final result iteratively. This exercise will help you understand how to manage kernel launches and synchronization in CUDA.

**Instructions:**

- Initialize a vector AA on the CPU with random values.
- Allocate GPU memory for AA and a single scalar result sumsum.
- Write a kernel function that performs a partial sum of the vector elements in parallel, storing the results back in the vector. In each iteration, the kernel will sum adjacent pairs of elements, reducing the size of the vector by half.
- Iteratively launch the kernel until there is only one element left in the vector (i.e., the final sum).
- Copy the result from GPU memory to the CPU and print it.

**Hints:**

- This exercise is a variation on a parallel reduction technique. Each kernel launch reduces the size of the input by half.
- Within each kernel call, you could use code similar to:

```sh
  __global__ void partialSum(float *A, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = 1;

    while (stride < N) {
        if (i % (2 * stride) == 0 && i + stride < N) {
            A[i] += A[i + stride];
        }
        stride *= 2;
        __syncthreads();
    }
  }

```


<p align="center"> <h1 align="center">Vector Summation with Thrust Library</h1> </p>

 Implement vector summation using Thrust, a high-level parallel library for CUDA. This exercise will show you how to simplify parallel programming in CUDA using Thrust’s built-in functions.
 
**Instructions:**

  - Initialize vectors AA and BB with random values on the CPU.
- Transfer the vectors to the GPU using thrust::device_vector.
   - Perform the vector addition using Thrust’s transform function.
   - Calculate the total sum of vector CC (result vector) using Thrust’s reduce function to demonstrate how both element-wise summation and full summation can be handled with Thrust.
   - Print the final sum from the result vector.

**Hints:**

  - Thrust’s device_vector automatically handles memory allocation and data transfer between CPU and GPU, simplifying code.
- Code snippet to perform the element-wise addition:

```sh
thrust::device_vector<float> A(N, 1.0f);  // Initialize A with N elements
thrust::device_vector<float> B(N, 2.0f);  // Initialize B with N elements
thrust::device_vector<float> C(N);

// Element-wise addition using Thrust
thrust::transform(A.begin(), A.end(), B.begin(), C.begin(), thrust::plus<float>());

// Calculate the total sum of vector C
float sum = thrust::reduce(C.begin(), C.end(), 0.0f, thrust::plus<float>());
```
