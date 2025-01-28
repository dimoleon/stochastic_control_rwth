using LinearAlgebra
using Random
using Distributions
using Plots
using PlotlyJS

# plotly()
gr()

include("parameters.jl")
include("utils.jl")
include("particle_filter.jl")
include("source_seeking.jl")

dt = 0.7
N = 50 
T = 1000
σ1, ϵ1 = 0.1, 0.4
σ2, ϵ2 = 0.3, 1.0

x0a = [10.0, 0.0]
x0b = [0.0, -10.0]
x0 = copy(x0b)

θ0a = [0.0, 0.0]
θ0b = [-10.0, 0.0]
θ0 = copy(θ0b)
σ, ϵ = σ1, ϵ1
# σ, ϵ = σ2, ϵ2
# σ, ϵ = 0.1, 1.0



estimates, lamp_positions, measurements = 
    particle_filter(x0, N, T, dt, σ, ϵ)
robot_positions, velocities = source_seeking(estimates, θ0, dt)

# Plotting
x_ref_circle = [10*cosd(t) for t in 0:360]
y_ref_circle = [10*sind(t) for t in 0:360]

# plot(x_ref_circle, y_ref_circle, color=:black, lw=0.3, aspect_ratio=:equal, label="Reference circle")
# est_x = getindex.(estimates, 1)
# est_y = getindex.(estimates, 2) 
# l_x = getindex.(lamp_positions, 1) 
# l_y = getindex.(lamp_positions, 2) 

# plot!(l_x, l_y, label="Lamp Position", lw=2)
# plot!(est_x, est_y, label="PF Estimate", lw=2) 
# scatter!([l_x[1]], [l_y[1]], label="x0") 
# title!("Particle Filter (σ=$(σ), ε=$(ϵ), x₀=$(x0))")


# anim = @animate for t in 1:length(lamp_positions) 
#     @views lamp_x = getindex.(lamp_positions[1:t], 1)
#             lamp_y = getindex.(lamp_positions[1:t], 2)
#             robot_x = getindex.(robot_positions[1:t], 1)
#             robot_y = getindex.(robot_positions[1:t], 2)
#     plot(x_ref_circle, y_ref_circle, color=:black, lw=0.3, aspect_ratio=:equal, label="Reference circle")
#     plot!(lamp_x, lamp_y, label="Lamp Position", lw=2)
#     plot!(robot_x, robot_y, label="Robot Position", lw=2)
#     title!("Particle Filter (σ=$(σ), ε=$(ϵ), x₀=$(x0), θ=$(θ0))")
#     xlabel!("x1")
#     ylabel!("x2")
# end

# gif(anim, "d2.gif")

