In GPU-based parallel programming, we transfer data from the CPU to the GPU, where the actual computation takes place, by using CUDA C/C++. This allows us to harness the immense computational power of GPU cores for faster processing.

## At this stage, you might be wondering about two key points:

- What exactly is CUDA?
- How do we transfer data between the CPU and GPU, and how can we fully utilize the GPU’s parallel processing capabilities?

**In this chapter, we’ll cover these fundamental concepts and walk through a simple "Hello World" program in CUDA C to help you get started with GPU programming.**

<p align="center">
 <h1 align="center">CUDA?</h1>
</p>

CUDA (Compute Unified Device Architecture) is a parallel computing platform created by NVIDIA. It enables developers to harness the power of the GPU (Graphics Processing Unit) for high-performance computational tasks. By using programming languages like C and C++, CUDA allows you to leverage the GPU's numerous cores to accelerate computations, making it an essential tool for tasks that require significant processing power, such as scientific simulations, machine learning, and real-time rendering.

As the name implies, CUDA C/C++ is a blend of standard C (or C++) with CUDA extensions. Therefore, a specialized compiler is required to process both C/C++ code and CUDA-specific code. NVIDIA provides NVCC (NVIDIA CUDA Compiler) to meet this need. NVCC is capable of compiling both traditional C/C++ binaries and CUDA binaries, ensuring that the code runs efficiently on both the CPU and GPU, seamlessly managing the communication between them during execution.

## How NVCC works:

- ***Code Analysis:*** NVCC begins by analyzing the source code to identify the sections written in CUDA C/C++. It differentiates between the code intended for the host (CPU) and the device (GPU), determining which parts of the program will run on each.

- ***Separation of Host and Device Code:*** Once identified, NVCC separates the host code, which will run on the CPU, from the device code, which will be executed on the GPU. It handles both code segments separately during compilation, ensuring the correct instructions are generated for the respective architectures.

- ***Compilation and Optimization:*** NVCC compiles the host code using a standard CPU compiler such as GCC (on Linux) or MSVC (on Windows). Simultaneously, it compiles the device code using the CUDA compiler provided by NVIDIA. The device code is optimized specifically for NVIDIA GPUs, leveraging their architecture to maximize performance.

- ***GPU-specific Code Generation:*** NVCC generates GPU-specific machine code in the form of PTX (Parallel Thread Execution). PTX is an intermediate representation of the device code, not directly executable on the GPU but serves as a platform-independent code for further processing.

- ***PTX Translation and Optimization:*** The NVIDIA GPU driver translates the PTX code into GPU-specific machine code, known as SASS (Scalable Assembly). During this step, additional architecture-specific optimizations are performed to enhance performance on the target GPU.

- ***Linking and Final Binary Generation:*** After compiling both the host and device code, NVCC links them together. The final step involves combining these components into an executable binary file, which can be run on the target GPU system. The executable contains the CPU instructions for the host and optimized GPU instructions for the device.

***This entire process ensures that the code is properly compiled for both CPU and GPU architectures, enabling efficient execution on systems with NVIDIA GPUs.***


<p align="center">
 <h1 align="center">How do we transfer data from the CPU to the GPU and make use of GPU cores?</h1>
</p>

First, we write code in C or C++ to gather data and store it in the CPU’s memory. From there, we call a kernel (a function written in CUDA that runs on the GPU) to transfer the data from CPU memory to GPU memory for processing. Once the computation is complete, we copy the results back from the GPU to the CPU to display or print the output. CPU and GPU each have their own separate memory spaces and they cannot directly access each other's memory, so data must be transferred between them via the PCI bus (Peripheral Component Interconnect). This copying process ensures that both the CPU and GPU have access to the necessary data for computation.

<p align="center">
 <h1 align="center">Hello GPU World !</h1>
</p>

```sh
#include <stdio.h>


__global__ void kernel()
{

    printf("hello GPU world");
}

int main()
{
    kernel<<<1,1>>>();
    cudaDeviceSynchronize();

    return 0;
}
```
***Lets go through the different parts of code and explain them in details: ***

### CUDA-specific function: __global__ void kernel():
```sh
__global__ void kernel()
{
    printf("hello GPU world");
}
```
__global__: This is a CUDA keyword that specifies that the function kernel is a kernel function, meaning it will be executed on the GPU, not the CPU. The __global__ qualifier tells CUDA that this function can be called from the host (CPU) but will run on the device (GPU).

#### void kernel():
This is the definition of the kernel function. It doesn't return anything (void). Inside this function, we are using printf to output the text "hello world". However, this printf will be executed by the GPU, not by the CPU. In CUDA, you can use printf inside a kernel for debugging or printing messages from the GPU side.

### main function: Host-side code:
```sh
int main()
{
    kernel<<<1,1>>>();
    cudaDeviceSynchronize();

    return 0;
}
```

#### kernel<<<1,1>>>();
Kernel launch syntax: kernel<<<1, 1>>>(); is the syntax for launching a kernel function in CUDA. The <<<1, 1>>> is a special syntax that specifies how many blocks and threads will be used to execute this kernel on the GPU. The first 1 specifies the number of blocks (a group of threads) to be used for execution. In this case, there is just 1 block. The second 1 specifies the number of threads per block. In this case, we are using 1 thread within that block. This means that the kernel will be executed by exactly one thread on the GPU in one block.

#### cudaDeviceSynchronize();
This is a CUDA runtime API function. It ensures that the host (CPU) waits for the device (GPU) to complete all preceding kernel launches before proceeding. CUDA kernel launches are asynchronous by default, meaning the host program does not wait for the GPU to finish executing the kernel. By calling cudaDeviceSynchronize(), we force the CPU to wait for the GPU to complete the kernel() execution, ensuring that the "hello world" message gets printed before the program terminates.
