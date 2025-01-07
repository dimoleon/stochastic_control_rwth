using Distributions, LinearAlgebra, Plots, Random

include("constants.jl")
include("utils.jl")
include("softmax.jl")
include("policy_gradients.jl")


θ0 = uniform_theta_generator() 
# θ0 = random_theta_generator() 
# θ_pg, opt_policy_pg, rewards_pg, pol_diffs_pg = policy_gradient(softmax, softmax_log_grad, θ0, M, N, η=0.003)
θ_spg, opt_policy_spg, rewards_spg, pol_diffs_spg = stochastic_policy_gradient(softmax, softmax_log_grad, θ0, M, η=0.003)



trajectories = []
inits = []
for y ∈ 1:5 
    push!(inits, (y, 1))
end
for y ∈ [2,3,5]
    push!(inits, (y, 4))
end

for state ∈ inits
    ω = sample_episode(state, opt_policy_spg)
    push!(trajectories, ω)
end

# Function to plot the robot's trajectory with action probabilities
function plot_trajectory_with_probs(trajectory)
    grid = zeros(Int, MAXY, MAXZ)
    
    # Mark obstacles and goal states
    for (y, z) in obstacles
        grid[y, z] = -1
    end
    for (y, z) in goals
        grid[y, z] = 1
    end

    # Create the plot
    plot()
    heatmap!(1:MAXZ, 1:MAXY, grid; color=:viridis, aspect_ratio=:equal, alpha=0.5)
    
    # Plot the trajectory
    for (i, (state, action, action_probs)) in enumerate(trajectory)
        y, z = state
        scatter!([z], [y], color=:blue, label=false)
        # text!(z, y, string(round.(action_probs, digits=2)), color=:black, halign=:center)
        
        # Add arrow indicating movement direction
        if i > 1
            prev_y, prev_z = trajectory[i - 1][1]
            # arrow!([prev_z, z], [prev_y, y], lw=1, arrowhead=0.3, label=false)
        end
    end
    
    # Annotate plot
    title!("Robot Trajectory with Action Probabilities")
    xlabel!("Z-axis")
    ylabel!("Y-axis")
end

# # Plot all trajectories
# for trajectory in trajectories
#     plot_trajectory_with_probs(trajectory)
#     sleep(1)  # Pause between plots to view each trajectory
# end