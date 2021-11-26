using Distributions, Plots, Turing, DataFrames, MCMCChains, StatsPlots


mutable struct QLearningAgent
    q::Array{Real, 1}
    α::Real
    β::Real
    function QLearningAgent(α::Real, β::Real, K::Int64)
        q = Array{Real, 1}(zeros(K))
        new(q, α, β)
    end
end

function update(agent::QLearningAgent, action::Real, reward::Real)
    agent.q[action] += agent.α * (reward - agent.q[action])
end

function policy(agent::QLearningAgent)
    qmax = maximum(agent.q)
    qexp = exp.(agent.β .* (agent.q .- qmax))
    return qexp ./ sum(qexp)
end


struct Bandit
    p::Array{Real, 1}
end

function step(env::Bandit, action::Int64)
    k = length(env.p)
    rewards = env.p .>= rand(k)
    return rewards[action]
end


function step(agent::QLearningAgent, env::Bandit)
    p = policy(agent)
    action = rand(Categorical(p))
    reward = step(env, action)
    update(agent, action, reward)
    d = DataFrame(action = action, reward = reward)
    return hcat(d, DataFrame(agent.q', :auto))
end

function run(agent::QLearningAgent, env::Bandit, trial::Int64)
    return reduce(vcat, map(_ -> step(agent, env), 1:trial))
end

bandit = Bandit([0.2, 0.8])
k = length(bandit.p)
agent = QLearningAgent(0.05, 2., k)
result = run(agent, bandit, 500)

plot(result.x1, label="Q1")
plot!(result.x2, label="Q2")
scatter!(result.action .- 1,
         markersize=1,
         markershape=:vline,
         label="Response")


@model QLearningModel(actions::Array{Real, 1}, rewards::Array{Real, 1}, k::Int64) = begin
    T = length(actions)
    α ~ Beta(1, 1)
    β ~ Gamma(1, 100)
    agent = QLearningAgent(α, β, k)
    for t in 1:T
        action = actions[t]
        reward = rewards[t]
        p = policy(agent)
        actions[t] ~ Categorical(p)
        update(agent, action, reward)
    end
end

actions = Array{Real, 1}(result.action)
rewards = Array{Real, 1}(result.reward)
chains = sample(QLearningModel(actions, rewards, 2), NUTS(), 1000)
plot(chains)

chains = sample(QLearningModel(actions, rewards, 2), NUTS(), MCMCThreads(), 1000, 4)
plot(chains)
