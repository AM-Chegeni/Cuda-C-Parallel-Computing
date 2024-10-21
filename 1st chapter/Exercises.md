<p align="center">
 <h1 align="center">Exercise 1: Exploring __device__ and __global__ Functions</h1>
</p>

You can create multiple __device__ functions and call them from a __global__ function. Additionally, you can call the __global__ function from a host function, and then invoke that host function in the main function. Try modifying the __device__ functions to perform different tasks and observe their outputs.

```sh
#include <stdio.h>

__device__ void DeviceFunction1(int value)
{
    printf("Device Function 1: Value is %d\n", value);
}

__device__ void DeviceFunction2(int value)
{
    printf("Device Function 2: Value is %d\n", value * 2);
}

__global__ void kernel(int input)
{
    DeviceFunction1(input);
    DeviceFunction2(input);
}

void hostFunction(int input)
{
    kernel<<<1, 1>>>(input);
    cudaDeviceSynchronize();
}

int main()
{
    int input = 5;
    hostFunction(input);
    return 0;
}
```

***Task: Modify the DeviceFunction1 and DeviceFunction2 to perform arithmetic operations and observe how the outputs change.***


<p align="center">
 <h1 align="center">Exercise 2: Understanding Thread and Block Configuration</h1>
</p>

In this exercise, you will call a __global__ function with different thread and block configurations to see how they affect the output. Try to use various configurations like <<<2, 4>>> or <<<1, 8>>> to print "hello world" multiple times and analyze the output.

```sh
#include <stdio.h>

__global__ void kernel()
{
    printf("Hello World from Thread %d in Block %d\n", threadIdx.x, blockIdx.x);
}

void hostFunction()
{
    kernel<<<2, 4>>>(); // Change configurations and observe
    cudaDeviceSynchronize();
}

int main()
{
    hostFunction();
    return 0;
}
```

***Task: Change the kernel launch configuration and observe how many times "Hello World" gets printed. Discuss the concept of threads and blocks in CUDA.***



<p align="center">
 <h1 align="center">Exercise 3: SIMT Behavior with Multiple Threads</h1>
</p>

In this exercise, you will create a __global__ function that uses multiple threads to calculate squares of numbers concurrently. You can observe the behavior of the threads and how they interact with each other.

```sh
#include <stdio.h>

__global__ void kernel(int* input, int* output)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    output[idx] = input[idx] * input[idx];
}

void hostFunction(int* input, int* output, int size)
{
    int* d_input;
    int* d_output;
    cudaMalloc(&d_input, size * sizeof(int));
    cudaMalloc(&d_output, size * sizeof(int));
    cudaMemcpy(d_input, input, size * sizeof(int), cudaMemcpyHostToDevice);

    kernel<<<1, size>>>(d_input, d_output);
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
}

int main()
{
    const int size = 5;
    int input[size] = {1, 2, 3, 4, 5};
    int output[size];

    hostFunction(input, output, size);

    for (int i = 0; i < size; i++)
    {
        printf("Square of %d is %d\n", input[i], output[i]);
    }
    
    return 0;
}
```

***Task: Modify the input array and observe how the output changes. Discuss the significance of SIMT (Single Instruction, Multiple Threads) in CUDA, especially how multiple threads can work on different data simultaneously.***

