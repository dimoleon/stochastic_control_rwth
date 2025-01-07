function policy_gradient(policy_func, grad_func, θ0, M, N; η = 0.003)
    sim_init_states = valid_init_states[:]
    pg_rewards = []
    policy_diffs = [] 
    updated_policy = create_policy(θ0, policy_func)
    θ = copy(θ0) 
    # for state_initial ∈ shuffle(valid_init_states)
    for state_initial ∈ sim_init_states
        for i ∈ 1:M 
            # state_initial = rand(states[1:5])
            rewards_tracker = zeros(N)
            for j ∈ 1:N 
                ω = sample_episode(state_initial, updated_policy) 
                trajectory_reward = sum([ω[t][3] for t ∈ eachindex(ω)])
                rewards_tracker[j] = trajectory_reward
                for (state, action, _) ∈ ω
                    y, z = state
                    θ[y, z] += trajectory_reward*grad_func(updated_policy, state, action) * η / N
                end
            end

            # push!(pg_rewards, mean(rewards_tracker))
            pg_rewards = vcat(pg_rewards, rewards_tracker)

            new_policy = create_policy(θ, policy_func) 
            for j ∈ 1:N
                policy_upd_diff = policy_diff_max(new_policy, updated_policy)
                push!(policy_diffs, policy_upd_diff)
            end

            updated_policy = new_policy
        end
    end

    return θ, updated_policy, pg_rewards, policy_diffs
end

function stochastic_policy_gradient(policy_func, grad_func, θ0, M; η = 0.01)
    sim_init_states = valid_init_states[:]
    spg_rewards = zeros(length(sim_init_states), M)
    policy_diffs = []
    updated_policy = create_policy(θ0, policy_func)
    θ = copy(θ0)
    # for state_initial ∈ shuffle(valid_init_states)
    for state_initial ∈ sim_init_states
        for i ∈ 1:M
            # state_initial = rand(states[1:5])
            ω = sample_episode(state_initial, updated_policy)
            trajectory_reward = sum([ω[t][3] for t ∈ eachindex(ω)])
            # push!(spg_rewards, trajectory_reward)
            spg_rewards[state_initial[1], i] = trajectory_reward
            for (state, action, _) ∈ ω
                y, z = state
                θ[y, z] += trajectory_reward*grad_func(updated_policy, state, action) * η
            end

            new_policy = create_policy(θ, policy_func) 
            push!(policy_diffs, policy_diff_max(new_policy, updated_policy))
            
            updated_policy = new_policy

            updated_policy = create_policy(θ, policy_func) 
        end
    end
    spg_rewards_mean = [mean([spg_rewards[x, m] for x ∈ eachindex(sim_init_states)]) for m ∈ 1:M]

    return θ, updated_policy, spg_rewards_mean, policy_diffs
end