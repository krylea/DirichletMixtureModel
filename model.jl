
CONJUGATES=Dict(
    NormalCanon => NormalGamma,
    Exponential => Gamma,
    MvNormalCanon => NormalWishart
)


abstract type MixtureModel end

struct UnivariateMixtureModel <: MixtureModel
    prior::UnivariateDistribution
    likelihood::Type
    canon_transform::Nullable{Function}
    conjugate::Bool
end

struct MultivariateMixtureModel <: MixtureModel
    prior::Distribution
    likelihood::Type
    canon_transform::Nullable{Function}
    conjugate::Bool
end

function MixtureModel(prior::Distribution, likelihood::Type, canon_transform=Nullable{Function}())
    if typeof(prior) <: CONJUGATES[likelihood]
        if likelihood<:UnivariateDistribution
            return UnivariateMixtureModel(prior, likelihood, canon_transform, true)
        else
            return MultivariateMixtureModel(prior, likelihood, canon_transform, true)
        end
    else
        if likelihood<:UnivariateDistribution
            return UnivariateMixtureModel(prior, likelihood, canon_transform, false)
        else
            return MultivariateMixtureModel(prior, likelihood, canon_transform, false)
        end
    end
end

function pdf_likelihood(M::MixtureModel, y::Union{Float64,Array{Float64,1}}, θ::Tuple)
    if isnull(M.canon_transform)
        l = M.likelihood(θ...)
    else
        l = M.likelihood(get(M.canon_transform)(θ...)...)
    end
    pdf(l, y)
end

function sample_posterior(M::UnivariateMixtureModel, Y::Array{Float64,1})
    rand(posterior_canon(M.prior, suffstats(M.likelihood, Y)))
end
function sample_posterior(M::UnivariateMixtureModel, y::Float64)
    rand(posterior_canon(M.prior, suffstats(M.likelihood, [y])))
end
function sample_posterior(M::MultivariateMixtureModel, Y::Array{Float64,2})
  p=posterior_canon(M.prior,suffstats(M.likelihood,Y))
  rand(p)
end
function sample_posterior(M::MultivariateMixtureModel, y::Array{Float64,1})
  p=posterior_canon(M.prior,suffstats(M.likelihood,reshape(y,(length(y),1))))
  rand(p)
end

function marginal_likelihood(M::MixtureModel, y::Union{Float64, Array{Float64,1},Array{Float64,2}})
    marginal_likelihood(M.prior, y)
end

#   Fallback
function marginal_likelihood(D::Distribution, y::Union{Float64, Array{Float64,1}})
    error("No marginal likelihood function provided.")
end
