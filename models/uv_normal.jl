#
# Utility functions for clustering with univariate normal likelihood (mean and precision unknown)
#

function pdf_likelihood(M::NormalGamma, y::Float64, θ::Tuple{Float64,Float64})
  pdf(NormalCanon(θ[1]*θ[2],θ[2]), y)
end

function sample_posterior(M::NormalGamma, Y::Array{Float64,1})
  rand(posterior_canon(M,suffstats(Normal,Y)))
end
function sample_posterior(M::NormalGamma, y::Float64)
  rand(posterior_canon(M,suffstats(Normal,[y])))
end

function marginal_likelihood(M::NormalGamma, y::Float64)
  gamma(M.shape+1/2)/gamma(M.shape) * sqrt(M.nu/(M.nu+1)) * 1/sqrt(2*π) * M.rate^M.shape /
    (M.rate+M.nu/2/(M.nu+1)*(y-M.mu)^2)^(M.shape+1/2)
end

function clusterUVN(Y::Array{Float64,1};α::Float64=1.0, μ0::Float64=0.0, n0::Float64=1e-8, α0::Float64=1.0, β0::Float64=1.0, iters::Int64=5000)
  U = ConjugatePriors.NormalGamma(μ0,n0,α0,β0)
  DMM.DPCluster(Y,U,α,iters=iters)
end
