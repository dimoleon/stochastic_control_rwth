function source_seeking(estimates::Vector{Vector{Float64}}, init_pos::Vector{Float64}, dt::Float64)
    robot_positions = [init_pos]
    velocities = [] 

    robot_position = copy(init_pos)
    for estimate in estimates
        velocity = robot_policy(robot_position, estimate)
        robot_position = robot_dynamics(robot_position, velocity, dt)

        push!(velocities, velocity)
        push!(robot_positions, robot_position)
    end
    
    return robot_positions, velocities
end