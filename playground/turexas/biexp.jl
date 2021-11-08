using Distributions, Turing, Plots, StatsPlots, MCMCChains, DataFrames

function BiExponential(theta1::Real, theta2::Real, p::Real)
    return MixtureModel(Exponential[
                             Exponential(theta1),
                             Exponential(theta2),
                            ], [p, 1 - p])
end

function generate_samples(theta1::Real, theta2::Real, p::Real, n::Int64)
    m = BiExponential(theta1, theta2, p)
    rand(m, n)
end

@model bi_exponential(samples::Array{Real, 1}) = begin
    theta1 ~ Gamma(1, 10)
    theta2 ~ truncated(Normal(), theta1, Inf)
    p ~ Beta(1, 1)
    m = BiExponential(theta1, theta2, p)
    for i in 1:length(samples)
        samples[i] ~ m
    end
end

function svf(x::Array{Real, 1}, digits::Int64)
    x = round.(x, digits=digits)
    unqx = unique(x) |> sort
    counts = map(x_ -> sum(x .== x_), unqx)
    DataFrame(x = unqx, count = counts, svr = 1. .- cumsum(counts ./ sum(counts)))
end

function predict(chains::Chains, x::Array{Real, 1})
    theta1 = mean(chains["theta1"])
    theta2 = mean(chains["theta2"])
    p = mean(chains["p"])
    becdf = cdf(BiExponential(theta1, theta2, p), x)
    return 1. .- becdf
end

samples = Array{Real, 1}(generate_samples(0.1, 5, 0.9, 2000))
chains = sample(bi_exponential(samples), NUTS(), MCMCThreads(), 1000, 4)

svf_samples = svf(samples, 2)
pred_samples = predict(chains, Array{Real, 1}(svf_samples.x))
scatter(svf_samples.x, log10.(svf_samples.svr), markersize=2)
plot!(svf_samples.x, log10.(pred_samples))
ylims!(-3, 0)
