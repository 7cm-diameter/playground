# `ql.jl` requires Turing, Distributions, Plots, and StatsPlots
using Turing, Distributions, Plots, StatsPlots

# definitions of type alias
Action = Int64
Reward = Real
Probability = Real
QValue = Real

# definition of the environment
struct Bandit
    p::Array{Probability, 1}
end

function step(bandit::Bandit, action::Action)
    p = bandit.p[action]
    if p >= rand()
        return 1.
    end
    return 0.
end

# definition of the model
mutable struct Agent
    q::Array{QValue, 1}
    alpha::Real
    beta::Real
end

function from_env(alpha::Real, beta::Real, env::Bandit)
    n = length(env.p)
    q = zeros(n)
    return Agent(q, alpha, beta)
end

function softmax(agent::Agent)
    qmax = maximum(agent.q)
    qexp = exp.(agent.beta * (agent.q .- qmax))
    return qexp / sum(qexp)
end

function choose_action(agent::Agent)
    p = softmax(agent)
    return rand(Categorical(p))
end

function update(agent::Agent, action::Action, reward::Reward)
    agent.q[action] += agent.alpha * (reward - agent.q[action])
end

# definition of steps in the simulation
function step(agent::Agent, env::Bandit)
    action = choose_action(agent)
    reward = step(env, action)
    update(agent, action, reward)
    return action, reward
end

function run(agent::Agent, env::Bandit, trial::Int64)
    result = Array{Tuple{Action, Reward}, 1}(undef, trial)
    for i in 1:trial
        ar = step(agent, env)
        result[i] = ar
    end
    return result
end

# run simulation and fit q-learning model to the simulated data
bandit = Bandit([0.1, 0.5])
agent = from_env(0.1, 2., bandit)
result = run(agent, bandit, 1000)

@model QLearning(env::Bandit, result::Array{Tuple{Action, Reward}}) = begin
    T = length(result)
    alpha ~ Beta(1, 1)
    beta ~ Gamma(1., 100.)
    agent = from_env(alpha, beta, env)

    for t in 1:(T - 1)
        a, r = result[t]
        p = softmax(agent)
        result[t][1] ~ Categorical(p)
        update(agent, a, r)
    end
end

chain = sample(QLearning(bandit, result), NUTS(), 500)
plot(chain)
