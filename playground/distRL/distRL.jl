using Distributions, DataFrames, Plots


mutable struct DistributionalQAgent
    αp::Array{Real, 1}
    αn::Array{Real, 1}
    β::Real
    q::Array{Real, 1}
    function DistributionalQAgent(αp::Array{Real, 1}, αn::Array{Real, 1}, β::Real , n::Real)
        q = Array{Real, 1}(zeros(n))
        new(αp, αn, β, q)
    end
end

function generate_lrs(min::Real, max::Real, n::Real)
    quantile(Uniform(min, max), 0.:(1 / (n - 1)):1.)
end

function update(agent::DistributionalQAgent, reward::Real)
    δ = ifelse.(reward .- agent.q .<= 0, -1., 1.)
    agent.q .+= δ .* ifelse.(δ .<= 0, agent.αn, agent.αp)
end


struct Environment
    p::Array{Real, 1}
    value::Array{Real, 1}
end

function step(env::Environment)
    i = rand(Categorical(env.p))
    env.value[i]
end


function step(env::Environment, agent::DistributionalQAgent)
    reward = step(env)
    update(agent, reward)
    DataFrame(agent.q', :auto)
end


N = 50
T = 1000
WARMUP = 100


αp = Array{Real, 1}(generate_lrs(0.01, 0.1, 50))
αn = reverse(αp)
agent = DistributionalQAgent(αp, αn, 1., N)
env = Environment([0.5, 0.5], [1., 2.])


result = Array{Real, 2}(reduce(vcat, map(_ -> step(env, agent), 1:T)))
result_without_warmup = reshape(result[WARMUP+1:end, :], ((T - WARMUP) * N, 1))
p1 = plot(result, label="", xlabel="Trial", ylabel="Q-value")
p2 = histogram(result_without_warmup, label="", xlabel="Q-value", ylabel="counts")
plot(p1, p2)
