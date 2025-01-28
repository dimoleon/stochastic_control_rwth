# consider passing the distribution as parameter to both dynamics and measurement function to avoid to creating new distribution variables on each iteration
function lamp_dynamics(x::Vector{Float64}, time::Float64, dt::Float64, laplace_var::Float64)
    x_new = x .+ 0.2*dt*[-sin(time/50), cos(time/50)]
    w = Laplace(0, laplace_var)
    laplace_noise = rand(w, 2)
    
    x_new .+= laplace_noise
    return x_new
end

function measurement(real_position::Vector{Float64}, epsilon::Float64)
    v = Uniform(0, epsilon)
    uniform_noise = rand(v, 2)

    estimate = real_position .+ uniform_noise
    return estimate
end

function weighted_mean(particles::Vector{Vector{Float64}}, weights::Vector{Float64})
    return sum(weights .* particles)
end

function robot_dynamics(curr_pos::Vector{Float64}, velocity::Vector{Float64}, dt::Float64)
    new_pos = curr_pos + velocity*dt
    return new_pos
end

function robot_policy(curr_pos::Vector{Float64}, lamp_estimate::Vector{Float64})
    velocity = lamp_estimate - curr_pos
    velocity /= max(norm(velocity), 10) 

    return velocity 
end