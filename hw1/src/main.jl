using Distributions, LinearAlgebra, Plots, Random, JLD2

include("constants.jl")
include("utils.jl")
include("softmax.jl")
include("policy_gradients.jl")


θ0 = uniform_theta_generator() 
# θ0 = random_theta_generator() 
θ_pg, opt_policy_pg, rewards_pg, pol_diffs_pg = policy_gradient(softmax, softmax_log_grad, θ0, M, N, η=0.003)
θ_spg, opt_policy_spg, rewards_spg, pol_diffs_spg = stochastic_policy_gradient(softmax, softmax_log_grad, θ0, M, η=0.003)

# policies_avg_rewards = [] 
# for i ∈ 1:20 
#     θ_spg, opt_policy_spg, rewards_spg, pol_diffs_spg = stochastic_policy_gradient(softmax, softmax_log_grad, θ0, 1000, η=0.003)
#     avg_reward = policy_episode_average(opt_policy_spg)
#     push!(policies_avg_rewards, avg_reward)
# end



trajectories = []
inits = []
for y ∈ 1:5 
    push!(inits, (y, 1))
end
for y ∈ [2,3,5]
    push!(inits, (y, 4))
end

for state ∈ inits
    y, z = state
    ω = sample_episode(state, opt_policy_spg)
    # push!(trajectories, ω)
    plot_queen(ω, opt_policy_spg)
    savefig("$y _ $z")
end

