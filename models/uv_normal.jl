#
# Utility functions for clustering with univariate normal likelihood (mean and precision unknown)
#

struct UnivariateNormalModel <: UnivariateConjugateModel
  prior::NormalGamma
end

function UnivariateNormalModel(ss::NormalStats)
  p=NormalGamma(ss.m,1e-8,2.0,0.5)
  UnivariateNormalModel(p)
end
function UnivariateNormalModel()
  p=NormalGamma(0.0,1e-8,2.0,0.5)
  UnivariateNormalModel(p)
end

function pdf_likelihood(model::UnivariateNormalModel, y::Float64, θ::Tuple{Float64,Float64})
  pdf(NormalCanon(θ[2]*θ[1], θ[2]), y)
end
function sample_posterior(model::UnivariateNormalModel, Y::Array{Float64,1})
  p=posterior_canon(model.prior,suffstats(Normal,Y))
  rand(p)
end
function sample_posterior(model::UnivariateNormalModel, y::Float64)
  p=posterior_canon(model.prior,suffstats(Normal,[y]))
  rand(p)
end
function marginal_likelihood(model::UnivariateNormalModel, y::Float64)
  p=model.prior
  gamma(p.shape+1/2)/gamma(p.shape) * sqrt(p.nu/(p.nu+1)) * 1/sqrt(2*π) * p.rate^p.shape /
    (p.rate+p.nu/2/(p.nu+1)*(y-p.mu)^2)^(p.shape+1/2)
end
