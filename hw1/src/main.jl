include("softmax.jl")

using Distributions, LinearAlgebra

T = 10 
M = Int(1e5) 
N = Int(100)

MINY = 1
MINZ = 1
MAXY = 5 
MAXZ = 10 

RGOAL = 50
ROBST = -10 
RDIAG = -2 

goals = Set([(2, 10), (4, 10)])
obstacles = Set([(2, 2), (2, 4), 
                (3, 3), 
                (1, 4), (4, 4), 
                (3, 5), (5, 5), 
                (1, 6), (4, 6),
                (2, 7), (5, 7), 
                (1, 8), (3, 8), (4, 8), 
                (3, 9), (4, 9),
                (1, 10), (3, 10), (5, 10)]) 

actions = [1, 2, 3]
# 1 -> same
# 2 -> up 
# 3 -> down 
# 0 -> no action possible 

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

states = [(y, z) for z ∈ MINZ:MAXZ for y ∈ MINY:MAXY]

# valid_actions_table = Dict((y, z) => valid_actions((y, z)) for y ∈ MINY:MAXY for z ∈ MINZ:MAXZ)

function calc_next_state(state, action) 
    y, z = state 
    @assert z < MAXZ "Κοίτα το ζ, βρε ηλίθιε"

    z_new = z + 1 
    y_new = y

    if action == 3
        @assert y > MINY "Κοίτα κάτω στο ψ, βρε ηλίθιε" 
        y_new -= 1
    elseif action == 2
        @assert y < MAXY "Κοίτα πάνω στο ψ, βρε ηλίθιε" 
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

function random_theta_generator()

    # 0 does not allow particular action
    helper(y, z, action) = action ∈ valid_actions((y,z)) ? rand() : 0

    # theta = [[[helper(y, z, action) for action ∈ actions] for z ∈ MINZ:MAXZ] for y ∈ MINY:MAXY] 
    theta = Dict((y, z) => [helper(y, z, action) for action ∈ actions] for (y, z) ∈ states)

    return theta
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

function policy_gradient(state_initial, policy_func, grad_func, θ0, M, N, η = 0.1)
    updated_policy = create_policy(θ0, policy_func)
    θ = Dict(θ0) 
    for _ ∈ 1:M 
        # generate same structure zeros θ
        # g = Dict(state => [0, 0, 0] for state ∈ states)

        for j ∈ 1:N 
            ω = sample_episode(state_initial, updated_policy) 
            trajectory_reward = sum([ω[t][3] for t ∈ eachindex(ω)])
            for (state, action, _) ∈ ω
                # y, z = state
                # g[state] += trajectory_reward*grad_func(updated_policy, state, action)
                θ[state] += trajectory_reward*grad_func(updated_policy, state, action) * η / N
            end
        end
        
        updated_policy = create_policy(θ, policy_func) 
        # for k ∈ θ
        #     θ[k] += g[k] * η / N
        # end
    end

    return θ, updated_policy
end

function stochastic_policy_gradient(state_initial, policy_func, grad_func, θ0, M, η = 0.1)
    
end

θ0 = random_theta_generator() 
θ, opt_policy = policy_gradient((2, 1), softmax, softmax_log_grad, θ0, M, N)