function softmax(vals)
    fake_exp(x) = x != 0 ? exp(x) : 0 
    exponed = fake_exp.(vals)
    s = sum(exponed) 

    return s == 0 ? exponed : exponed/s
end

function softmax_log_grad(policy, state, action) 
    if (action == 0) 
        return zeros(length(actions))
    end
    prob = -policy[state[1], state[2]]
    prob[action] += 1 

    return prob
end
