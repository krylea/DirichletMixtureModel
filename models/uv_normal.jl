#
# Utility functions for clustering with univariate normal likelihood (mean and precision unknown)
#

function suffstats(::Type{NormalCanon}, x::AbstractVector{Float64})
  suffstats(Normal, x)
end

function UnivariateNormalModel(p::NormalGamma)
  MixtureModel(p, NormalCanon, canon_uvn)
end
function UnivariateNormalModel()
  p=NormalGamma(0.0,1e-8,2.0,0.5)
  MixtureModel(p, NormalCanon, canon_uvn)
end

function canon_uvn(mu::Float64, tau::Float64)
  (mu*tau, tau)
end

function marginal_likelihood(p::NormalGamma, y::Float64)
  gamma(p.shape+1/2)/gamma(p.shape) * sqrt(p.nu/(p.nu+1)) * 1/sqrt(2*π) * p.rate^p.shape /
    (p.rate+p.nu/2/(p.nu+1)*(y-p.mu)^2)^(p.shape+1/2)
end

function clusterUVN(Y::Array{Float64,1};α::Float64=1.0, iters::Int64=5000)
  U = UnivariateNormalModel(NormalGamma())
  DMM.DPCluster(Y,U,α,iters=iters)
end
