# CUDA and Parallel Computing Projects

## Section 1: CUDA and Python on GPU

### 1. Matrix Manipulation Toolkit
**Goal**: Implement a toolkit that performs matrix-matrix multiplication, matrix transposition, and scalar matrix multiplication using CUDA. Benchmark its performance against a CPU-based implementation.  
**Data**: Generate random matrices of varying sizes (e.g., 512x512, 1024x1024, 2048x2048) using NumPy.  
**Key Concepts**: Memory coalescing, shared memory, tiling, and benchmarking techniques.  

---

### 2. Image Convolution and Edge Detection
**Goal**: Implement a CUDA-based image processing pipeline for convolution and edge detection (e.g., Sobel filter).  
**Data**: Use standard grayscale images (e.g., 512x512) from datasets like MNIST or CIFAR grayscale.  
**Key Concepts**: Thread synchronization, memory coalescing, and stencil operations.  

---

### 3. Heat Diffusion Simulation
**Goal**: Solve a 2D heat diffusion equation using CUDA-based stencil operations.  
**Data**: Initialize a grid (e.g., 1024x1024) with random heat values and use boundary conditions.  
**Key Concepts**: Thread indexing, grid/block allocation, and transparent scalability.  

---

### 4. Parallel Reduction for Statistics
**Goal**: Compute the mean, variance, and other statistics of a large dataset using CUDA parallel reduction.  
**Data**: Generate large datasets of random floating-point numbers (e.g., arrays of size 10⁶ or 10⁸).  
**Key Concepts**: Reduction, warp scheduling, and latency hiding.  

---

### 5. Physics Simulation: N-body Problem
**Goal**: Simulate gravitational interactions among particles (stars or galaxies) using CUDA for parallel computation.  
**Data**: Initialize particles with random positions, velocities, and masses in 3D space.  
**Key Concepts**: Thread synchronization, device memory management, and memory hierarchy.  

---

### 6. Tiled Matrix Multiplication
**Goal**: Implement a tiled matrix multiplication algorithm and compare performance with a naive approach.  
**Data**: Generate random matrices of increasing sizes (e.g., 1024x1024, 2048x2048).  
**Key Concepts**: Memory sharing, tiling, and scalability.  

---

## Section 2: OpenACC and OpenMP

### 1. Monte Carlo Simulation for Pi Calculation
**Goal**: Use OpenMP and OpenACC to parallelize a Monte Carlo method to estimate the value of Pi.  
**Data**: Randomly generate 10⁶ points inside a unit square.  
**Key Concepts**: Parallel regions, thread scheduling, and reducing overhead.  

---

### 2. Wave Equation Simulation
**Goal**: Simulate the propagation of waves in a 2D medium using OpenACC and compare it with a sequential CPU implementation.  
**Data**: Initialize a 2D grid with a disturbance at the center and define boundary conditions.  
**Key Concepts**: Data regions, loop optimizations, and efficient memory usage.  

---

### 4. Numerical Integration
**Goal**: Use OpenACC to parallelize a numerical integration method (e.g., Simpson's rule or trapezoidal rule).  
**Data**: Generate functions to integrate (e.g., polynomials, sine functions, or more complex expressions).  
**Key Concepts**: Loop scheduling, thread management, and speedup evaluation.  

---

### 5. Particle System Simulation
**Goal**: Simulate a simple particle system (e.g., gas molecules in a box) using OpenMP for CPU and OpenACC for GPU acceleration.  
**Data**: Initialize particles with random positions and velocities.  
**Key Concepts**: Work distribution, thread collaboration, and memory optimizations.  

---

## Combining Sections 1 and 2

### 1. Hybrid Parallel Matrix Multiplication
**Goal**: Develop a hybrid approach where CUDA handles the core computation (e.g., matrix multiplication) and OpenMP/OpenACC handles preprocessing or smaller tasks.  
**Data**: Randomly generate matrices of varying sizes.  
**Key Concepts**: Efficient use of GPU and CPU, data sharing between host and device.  

---

### 2. Hybrid N-body Simulation
**Goal**: Offload the heavy computation (force calculation) to CUDA, while OpenMP handles task scheduling and initial setup.  
**Data**: Randomly initialize positions, velocities, and masses.  
**Key Concepts**: Device coordination, multi-level parallelism, and performance analysis.  

---

### 3. Real-Time Image Processing
**Goal**: Combine OpenACC for preprocessing (e.g., resizing or filtering) and CUDA for detailed processing (e.g., edge detection, convolution).  
**Data**: Use a dataset of images (e.g., from ImageNet or custom images).  
**Key Concepts**: Coordinating host and device, memory optimization, and real-time benchmarks. 


---

### 4. Estimating \( \pi \) Using Numerical Integration
**Goal**: This project explores how to compute the integral:

\[
\int_0^1 \frac{4}{1+x^2} \, dx = \pi
\]

using numerical methods such as the trapezoidal rule or Riemann sums and parallelize it with CUDA, OpenACC, and OpenMP. The goal is to analyze the performance of these parallelization techniques and compare their speedups and accuracies.
  
**Data**: ## Data

- No pre-existing data is required.
- Generate a sequence of evenly spaced \( x \)-values in \([0, 1]\).
- The function \( f(x) = \frac{4}{1+x^2} \) is predefined.
 
**Key Concepts**: ### CUDA:
- Thread indexing for assigning subintervals.
- Efficient memory management and reduction operations.

### OpenMP:
- Use `#pragma omp parallel for` to divide work across threads.
- Manage shared and private variables for efficient computation.

### OpenACC:
- Use `#pragma acc parallel loop` to accelerate computation on the GPU.
- Handle data movement between host and device memory.

