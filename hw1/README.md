### Stochastic Control Homework 1

Source code for the first homework project of the course Stochastic Control in RWTH Aachen. 

Simulations implemented in Julia. 
Just run the `main.jl` file to get the simulation results for Policy Gradient and Stochastic Policy Gradient. 

Also, saves the plots of the movement of the robot for the optimal policy found by SPG. 

- `utils.jl` contains various auxilliary functions used inside the code base or for plotting, printing, and debugging. 
- `constants.jl` contains various constants used throughout the codebase, as well as $M$ and $N$ which affect the number of episodes used for each simulation.
- `optimal_policy.jld2` contains the optimal parametrization and the optimal control policy generated in our simulations. The Julia package `JLD2` must be installed; then the data can be loaded with the command `@load "optimal_policy.jld2"`. 
- The rest of the files should be self-explenatory. 