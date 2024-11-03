<p align="center"> <h1 align="center">N-body Simulation for Gravitational Systems</h1> </p>

This project involves simulating particles interacting under gravitational forces, such as stars in a galaxy, and observing their behavior over time.

***Goal:***

Simulate a system of particles where each particle exerts a gravitational force on every other particle. This simulation can represent celestial bodies, like stars or planets, under the influence of mutual gravitational attraction.

***Challenges:***

- **Computational Complexity:** Each particle interacts with every other particle, leading to an $O(N^2)$ complexity for N particles. CUDA is useful here to parallelize force calculations.
- **Accuracy and Stability:** Gravitational simulations can be sensitive to numerical errors, so we’ll need to carefully choose time steps and use stable algorithms like the Leapfrog integration method.

***CUDA Concepts Applied:***

 - **Parallel Vector Operations:** Each particle's force vector is calculated independently and can be parallelized across CUDA threads.
- **Reduction for Summing Forces:** Using efficient memory management for aggregating forces acting on each particle.

***Physics Background***

Newton's law of gravitation states that the gravitational force F between two particles with masses $m_1$​ and $m_2$ separated by a distance r is:
```math
F=G⋅\frac{m_1⋅m_2}{r^2}
```

where G is the gravitational constant. The force is a vector, pointing from one particle to the other, and affects each particle's acceleration.


***Steps for the N-Body Simulation***

- **Initialize:** Assign initial positions, velocities, and masses for all particles.
- **Compute Forces:** For each particle, calculate the gravitational force exerted by every other particle.
- **Update Velocities and Positions:** Use the computed forces to update each particle’s velocity and position.
- **Repeat:** Continue calculating forces and updating positions over successive time steps.
