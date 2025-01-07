function end_condition(state) 
    y, z = state 
    return z == MAXZ 
end

function valid_actions(state)
    y, z = state 
    if z == MAXZ 
        return []
    elseif y == MINY
        return [1, 2]
    elseif y == MAXY
        return [1, 3]
    end

    return actions[:] 
end

function calc_next_state(state, action) 
    y, z = state 
    # @assert z < MAXZ "Κοίτα το ζ, βρε ηλίθιε"

    z_new = z + 1 
    y_new = y

    if action == 3
        # @assert y > MINY "Κοίτα κάτω στο ψ, βρε ηλίθιε" 
        y_new -= 1
    elseif action == 2
        # @assert y < MAXY "Κοίτα πάνω στο ψ, βρε ηλίθιε" 
        y_new += 1 
    end 

    return (y_new, z_new) 
end

function action_reward(pre_action, curr_state)
    reward = 0 
    if curr_state ∈ goals
        reward += RGOAL
    end
    if curr_state ∈ obstacles
        reward += ROBST
    end

    if pre_action != 1
        reward += RDIAG
    end

    return reward
end

function create_policy(θ, policy_func)
    # policy = Dict((y, z) => policy_func(θ[y][z]) for y ∈ MINY:MAXY for z ∈ MINZ:MAXZ)
    policy = policy_func.(θ)

    return policy 
end

function select_action(policy, curr_state)
    weights = policy[curr_state[1], curr_state[2]] 
    dist = Categorical(weights)
    return actions[rand(dist)]
end

function sample_episode(state_initial, policy)
    if (end_condition(state_initial)) 
        # what if it starts on an end goal? 
        return [(state_initial, 0, 0)] 
    end

    # T = 0
    curr_state = state_initial
    curr_action = select_action(policy, curr_state)
    reward_0 = 0 
    path = [(curr_state, curr_action, reward_0)] 

    # (1 -> T)
    for time ∈ 1:T
        next_state = calc_next_state(curr_state, curr_action) 
        curr_reward = action_reward(curr_action, next_state) 
        curr_state = next_state

        if (end_condition(curr_state)) 
            push!(path, (curr_state, 0, curr_reward))
            break
        end

        curr_action = select_action(policy, curr_state)
        # println("Curr action: ", curr_action) 
        push!(path, (curr_state, curr_action, curr_reward))
    end

    return path 
end

function print_theta(θ) 
    for z ∈ MINZ:MAXZ
        for y ∈ MINY:MAXY
            val = round.(θ[y,z]; digits=2)
            println("($y, $z) => $val")
        end
    end
end

function random_theta_generator()

    # 0 does not allow particular action
    helper(y, z, action) = action ∈ valid_actions((y,z)) ? rand() : 0

    # theta = [[[helper(y, z, action) for action ∈ actions] for z ∈ MINZ:MAXZ] for y ∈ MINY:MAXY] 
    # theta = Dict((y, z) => [helper(y, z, action) for action ∈ actions] for (y, z) ∈ states)
    theta = [[helper(y, z, action) for action ∈ actions] for y ∈ states_y, z ∈ states_z]
    return theta
end

function uniform_theta_generator()

    # 0 does not allow particular action
    helper(y, z, action) = action ∈ valid_actions((y,z)) ? 1.0 : 0

    theta = [[helper(y, z, action) for action ∈ actions] for y ∈ states_y, z ∈ states_z]
    return theta
end

function policy_diff_max(policyA, policyB)
    # policies are matrices of 3 element vectors
    return norm(norm.(policyA - policyB))
end

function plot_queen(trajectory, policy)
    grid = zeros(Int, MAXY, MAXZ)
    
    for (y, z) in obstacles
        grid[y, z] = -1
    end
    for (y, z) in goals
        grid[y, z] = 1
    end

    plot()
    heatmap!(1:MAXZ, 1:MAXY, grid; color=:viridis, aspect_ratio=:equal, alpha=0.5)
    
    for (i, (state, action, reward)) in enumerate(trajectory)
        y, z = state
        
        if i > 1
            prev_y, prev_z = trajectory[i - 1][1]
            plot!([prev_z, z], [prev_y, y], arrow=true, lw=1, arrowhead=0.3, label=false, color=:red)
        end

        if end_condition(state)
            break
        end
        action_probs = policy[y, z][action]
        scatter!([z], [y], color=:blue, label=false)
        annotate!(z, y-1, text(string(round.(action_probs, digits=2)), color=:black, halign=:center, 7))
    end
    
    xlabel!("Z")
    ylabel!("Y")
end

function policy_episode_average(policy)
    rewards = []
    for x ∈ valid_init_states
        ω = sample_episode(x, policy)
        reward = sum([ω[t][3] for t ∈ eachindex(ω)])
        push!(rewards, reward) 
    end
    return mean(rewards)
end

# # Plot all trajectories
# for trajectory in trajectories
#     plot_trajectory_with_probs(trajectory)
#     sleep(1)  # Pause between plots to view each trajectory
# end