using Distributions, DataFrames, Plots, StatsPlots


mutable struct DistributionalQAgent
    αp::Array{Real, 1}
    αn::Array{Real, 1}
    q::Array{Real, 1}
    function DistributionalQAgent(αp::Array{Real, 1}, αn::Array{Real, 1}, n::Real)
        q = Array{Real, 1}(zeros(n))
        new(αp, αn, q)
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


# Simulation 1
N = 2
T = 300
WARMUP = 100

αp = Array{Real, 1}(generate_lrs(0.01, 0.1, N))
αn = reverse(αp)
agent = DistributionalQAgent(αp, αn, N)
env = Environment([0.5, 0.5], [0.5, 2.0])

result = Array{Real, 2}(reduce(vcat, map(_ -> step(env, agent), 1:T)))

hline([0.5, 2.], linestyle = :dash, label = "", color = :black)
plot!(result, label=["optimistic" "pessimistic"], xlabel="Trial", ylabel="State value for each learner", linewidth=1.75, legend = :bottomright, color = [1 2])

savefig("./simulation1.png")

# Simulation 2
N = 3
T = 300
WARMUP = 100

αp = Array{Real, 1}(generate_lrs(0.01, 0.1, N))
αn = reverse(αp)
agent = DistributionalQAgent(αp, αn, N)
env = Environment([1/3, 1/3, 1/3], [0.5, 1., 2.0])

result = Array{Real, 2}(reduce(vcat, map(_ -> step(env, agent), 1:T)))

hline([0.5, 1., 2.], linestyle = :dash, label = "", color = :black)
plot!(result, label = "", xlabel="Trial", ylabel="State value for each learner", linewidth=1.75, legend = :bottomright, color = [1 2 3])

savefig("./simulation2.png")

# Simulation 3
N = 100
T = 300
WARMUP = 100


αp = Array{Real, 1}(generate_lrs(0.01, 0.1, N))
αn = reverse(αp)
agent = DistributionalQAgent(αp, αn, N)
env = Environment([0.5, 0.5], [1., 3.])

result = Array{Real, 2}(reduce(vcat, map(_ -> step(env, agent), 1:T)))
result_without_warmup = reshape(result[WARMUP+1:end, :], ((T - WARMUP) * N, 1))

colors = palette(:viridis, N)
p1 = plot(result,
          label="", xlabel="Trial", ylabel="State value for each learner",
          linewidth=0.75, color = colors[1:end]')

p2 = density(result_without_warmup,
             label="", xlabel="State value", ylabel="Density",
             fillcolor = :red)

plot(p1, p2)

savefig("./simulation3.png")
# Simulation 4
env = Environment([0.25, 0.25, 0.5], [0., 1., 3.])

result = Array{Real, 2}(reduce(vcat, map(_ -> step(env, agent), 1:T)))
result_without_warmup = reshape(result[WARMUP+1:end, :], ((T - WARMUP) * N, 1))

p1 = plot(result,
          label="", xlabel="Trial", ylabel="State value for each learner",
          linewidth=0.75, color = colors[1:end]')

p2 = density(result_without_warmup,
             label="", xlabel="State values", ylabel="Density",
             fillcolor = :red)

plot(p1, p2)

savefig("./simulation4.png")
