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
    policy = Dict(state => policy_func(θ[state]) for state ∈ states) 

    return policy 
end

function select_action(policy, curr_state)
    weights = policy[curr_state] 
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
            val = round.(θ[(y,z)]; digits=2)
            println("($y, $z) => $val")
            # @printf("(%d, %d) => %.2f", y, z, val)
        end
    end
end

function random_theta_generator()

    # 0 does not allow particular action
    helper(y, z, action) = action ∈ valid_actions((y,z)) ? rand() : 0

    # theta = [[[helper(y, z, action) for action ∈ actions] for z ∈ MINZ:MAXZ] for y ∈ MINY:MAXY] 
    theta = Dict((y, z) => [helper(y, z, action) for action ∈ actions] for (y, z) ∈ states)

    return theta
end

function uniform_theta_generator()

    # 0 does not allow particular action
    helper(y, z, action) = action ∈ valid_actions((y,z)) ? 1.0 : 0

    # theta = [[[helper(y, z, action) for action ∈ actions] for z ∈ MINZ:MAXZ] for y ∈ MINY:MAXY] 
    theta = Dict((y, z) => [helper(y, z, action) for action ∈ actions] for (y, z) ∈ states)

    return theta
end

function policy_diff_max(policyA, policyB)
    # returns infinite norm of the vector of diffs
    return norm([policyA[k] - policyB[k] for k ∈ keys(policyA)], Inf) 
end