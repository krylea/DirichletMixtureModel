#until ConjugatePriors gets updated
function rand(nw::NormalWishart)
    Lam = rand(Wishart(nw.nu, nw.Tchol))
    Lsym = PDMat(Symmetric(inv(Lam) ./ nw.kappa))
    mu = rand(MvNormal(nw.mu, Lsym))
    return (mu, Lam)
end

function posterior_canon(prior::NormalWishart, ss::MvNormalStats)
    mu0 = prior.mu
    kappa0 = prior.kappa
    TC0 = prior.Tchol
    nu0 = prior.nu

    kappa = kappa0 + ss.tw
    nu = nu0 + ss.tw
    mu = (kappa0.*mu0 + ss.s) ./ kappa

    z = prior.zeromean ? ss.m : ss.m - mu0
    Lam = Symmetric(inv(inv(TC0) + ss.s2 + kappa0*ss.tw/kappa*(z*z')))

    return NormalWishart(mu, kappa, cholfact(Lam), nu)
end

function logpdf(nw::NormalWishart, x::Vector{T}, Lam::Matrix{T}) where T<:Real
    p = length(x)

    nu = nw.nu
    kappa = nw.kappa
    mu = nw.mu
    Tchol = nw.Tchol
    hnu = 0.5 * nu
    hp = 0.5 * p

    # Normalization
    logp = hp*(log(kappa) - Float64(log(2π)))
    logp -= hnu * logdet(Tchol)
    logp -= hnu * p * log(2.)
    logp -= logmvgamma(p, hnu)

    # Wishart (MvNormal contributes 0.5 as well)
    logp += (hnu - hp) * logdet(Lam)
    logp -= 0.5 * trace(Tchol \ Lam)

    # Normal
    z = nw.zeromean ? x : x - mu
    logp -= 0.5 * kappa * dot(z, Lam * z)

    return logp
end
pdf(nw::NormalWishart, x::Vector{T}, Lam::Matrix{S}) where T<:Real where S<:Real =
exp(logpdf(nw, x, Lam))



function suffstats(::Type{MvNormalCanon}, x::AbstractMatrix{Float64})
  suffstats(MvNormal, x)
end


function MultivariateNormalModel(p::NormalWishart)
  MixtureModel(p, MvNormalCanon, canon_mvn)
end
function MultivariateNormalModel(ss::MvNormalStats)
  p=NormalWishart(ss.m, 1e-8, ss.s2/ss.tw, Float64(length(ss.m)))
  MixtureModel(p, MvNormalCanon, canon_mvn)
end
function MultivariateNormalModel(d::Int64)
  p=NormalWishart(zeros(d), 1.0, eye(d), d*1.0)
  MixtureModel(p, MvNormalCanon, canon_mvn)
end
function MultivariateNormalModel()
  d=2
  p=NormalWishart(zeros(d), 1.0, eye(d), d*1.0)
  MixtureModel(p, MvNormalCanon, canon_mvn)
end


function canon_mvn(μ::Array{Float64, 1}, T::Array{Float64,2})
  (T*μ, T)
end


function marginal_likelihood(p::NormalWishart, y::Array{Float64,2})
  d=length(y)
  mu0 = p.mu
  kappa0 = p.kappa
  TC0 = p.Tchol
  nu0 = p.nu

  kappa = kappa0 + 1
  nu = nu0 + 1
  mu = (kappa0.*mu0 + y) ./ kappa

  Lam0 = TC0[:U]'*TC0[:U]
  Lam = Lam0 + kappa0/kappa*(mu0-y)*(mu0-y)'

  exp(-d/2*log(π) + logmvgamma(d,nu/2) - logmvgamma(d,nu0/2) + nu0/2*logdet(TC0) - nu/2*logdet(Lam) + d/2 * (log(kappa0) - log(kappa)))
end

function clusterMVN(Y::Array{Float64,2}; α::Float64=1.0, μ0::Array{Float64,1}=Array{Float64,1}(), κ0::Float64=1.0, T0::Array{Float64,2}=Array{Float64,2}(), ν0::Float64=NaN, iters::Int64=5000)
  D=size(Y,2)
  U = MultivariateNormalModel(D)
  DMM.DPCluster(Y,U,α,iters=iters)
end
