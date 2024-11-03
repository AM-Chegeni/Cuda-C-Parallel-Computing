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

```
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

```
thrust::device_vector<float> A(N, 1.0f);  // Initialize A with N elements
thrust::device_vector<float> B(N, 2.0f);  // Initialize B with N elements
thrust::device_vector<float> C(N);

// Element-wise addition using Thrust
thrust::transform(A.begin(), A.end(), B.begin(), C.begin(), thrust::plus<float>());

// Calculate the total sum of vector C
float sum = thrust::reduce(C.begin(), C.end(), 0.0f, thrust::plus<float>());
```
