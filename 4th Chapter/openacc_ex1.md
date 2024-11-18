<p align="center"> <h1 align="center">Quantum Harmonic Oscillator Energy Transitions</h1> </p>

***Understand the Problem***

The quantum harmonic oscillator has discrete energy states represented by a Hamiltonian matrix $H$. For a given state vector $\psi$, the new state after a time evolution or operation is given by:
```math
\psi_{new}​=H⋅\psi
```

In this exercise, you'll:

  - Represent the Hamiltonian as a matrix $H$.
- Multiply HH with multiple state vectors $\psi_i$​ in parallel using OpenACC.


***Steps to Solve***

1. Define the Hamiltonian Matrix
   - Use a symmetric matrix for the harmonic oscillator: 
```math
H_{ij} = 
\begin{cases} 
\hbar \omega \left(i + \frac{1}{2}\right), & \text{if } i = j \\ 
0, & \text{otherwise}
\end{cases}
```
Here:

  - $\hbar$ is the reduced Planck's constant.
 - $\omega$ is the angular frequency.
- The matrix is diagonal because the quantum harmonic oscillator has discrete energy states without direct coupling.


2. Generate State Vectors
   Generate M random normalized state vectors $\psi_i$​ , each of size N. The normalization condition is:

```math
\|\psi_i\| = \sqrt{\sum_{j=1}^N \psi_{ij}^2} = 1
```

3. Perform Matrix-Vector Multiplication

 For each state vector $\psi_i$​​, calculate:

```math
\psi_{\text{new}, i} = \sum_{j=1}^N H_{ij} \psi_{ij}
```
 This operation needs to be parallelized using OpenACC for multiple $\psi_i$​​

4. Implementation in C with OpenACC

```sh

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 100  // Size of the Hamiltonian
#define VECTORS 1000  // Number of state vectors

void generate_hamiltonian(double H[N][N], double hbar_omega) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == j) {
                H[i][j] = hbar_omega * (i + 0.5);
            } else {
                H[i][j] = 0.0;
            }
        }
    }
}

void generate_state_vectors(double psi[VECTORS][N]) {
    for (int v = 0; v < VECTORS; v++) {
        double norm = 0.0;
        for (int i = 0; i < N; i++) {
            psi[v][i] = (double)rand() / RAND_MAX;
            norm += psi[v][i] * psi[v][i];
        }
        // Normalize
        norm = sqrt(norm);
        for (int i = 0; i < N; i++) {
            psi[v][i] /= norm;
        }
    }
}

void matrix_vector_multiplication(double H[N][N], double psi[VECTORS][N], double psi_new[VECTORS][N]) {
    #pragma acc parallel loop collapse(2) copyin(H[0:N][0:N], psi[0:VECTORS][0:N]) copyout(psi_new[0:VECTORS][0:N])
    for (int v = 0; v < VECTORS; v++) {
        for (int i = 0; i < N; i++) {
            psi_new[v][i] = 0.0;
            for (int j = 0; j < N; j++) {
                psi_new[v][i] += H[i][j] * psi[v][j];
            }
        }
    }
}

int main() {
    double H[N][N], psi[VECTORS][N], psi_new[VECTORS][N];
    double hbar_omega = 1.0;  // Energy quantum

    // Initialize Hamiltonian and state vectors
    generate_hamiltonian(H, hbar_omega);
    generate_state_vectors(psi);

    // Perform matrix-vector multiplication
    matrix_vector_multiplication(H, psi, psi_new);

    // Print the first new state vector for verification
    printf("New state vector (first 5 elements):\n");
    for (int i = 0; i < 5; i++) {
        printf("%f ", psi_new[0][i]);
    }
    printf("\n");

    return 0;
}
```
