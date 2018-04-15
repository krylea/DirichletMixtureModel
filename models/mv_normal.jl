#until ConjugatePriors gets updated
function rand(nw::NormalWishart)
    Lam = rand(Wishart(nw.nu, nw.Tchol))
    Lsym = PDMat(Symmetric(inv(Lam) ./ nw.kappa))
    mu = rand(MvNormal(nw.mu, Lsym))
    return (mu, Lam)
end

function suffstats(::Type{MvNormalCanon}, x::AbstractMatrix{Float64})
  suffstats(MvNormal, x)
end


function MultivariateNormalModel(p::NormalWishart)
  MixtureModel(p, MvNormalCanon, canon_mvn)
end
function MultivariateNormalModel(d::Int64)
  p=NormalWishart(zeros(d), 1.0, eye(d), d-1.0)
  MixtureModel(p, MvNormalCanon, canon_mvn)
end
function MultivariateNormalModel()
  d=2
  p=NormalWishart(zeros(d), 1.0, eye(d), d-1.0)
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
