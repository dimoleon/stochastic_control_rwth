function particle_filter(x0::Vector{Float64}, num_particles::Int64, num_steps::Int64,
                    dt::Float64, σ::Float64, ϵ::Float64, N_tr::Float64 = num_particles/4)
    # initialization

    # particles initialization might need to be changed
    # particles = measurement.(fill(x0, num_particles), ϵ)
    particles = [[rand(Uniform(x0[1] - 4, x0[1] + 4)), 
                       rand(Uniform(x0[2] - 4, x0[2] + 4))] for _ in 1:num_particles]

    weights = fill(1/num_particles, num_particles) 
    lamp_positions = [x0] 
    estimates = [weighted_mean(particles, weights)] 
    measurements = [measurement(x0, ϵ)] 

    x_real = copy(x0)
    for t in 1:num_steps
        x_real = lamp_dynamics(x_real, t*dt, dt, σ)
        z = measurement(x_real, ϵ)

        push!(lamp_positions, x_real)
        push!(measurements, z)

        particles = lamp_dynamics.(particles, t*dt, dt, σ)

        # likelihoods = [(all(p - z .<= ϵ) ? 1.0 : 0.0) for p in particles]
        likelihoods = [exp(-norm(p .- z) / ϵ) for p in particles]
        weights = weights .* likelihoods
        weights /= sum(weights)

        # resampling 
        if N_tr * sum(weights.^2) > 1
            weight_distribution = Categorical(weights)
            particles = particles[rand(weight_distribution, num_particles)]
            weights .= 1 / num_particles
        end

        estimate = weighted_mean(particles, weights)
        push!(estimates, estimate) 
    end

    return estimates, lamp_positions, measurements
end