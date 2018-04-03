

function pdf_likelihood(M::NormalWishart, y::Float64, θ::Tuple{AbstractVector,AbstractMatrix})
  pdf(MvNormalCanon(θ[2]*θ[1], θ[2]), y)
end

function sample_posterior(M::NormalWishart, Y::Array{Float64,2})
  rand(posterior_canon(M,suffstats(MvNormal,Y)))
end
function sample_posterior(M::NormalWishart, y::Array{Float64,1})
  rand(posterior_canon(M,suffstats(MvNormal,reshape(y,(length(y),1)))))
end

function marginal_likelihood(M::NormalWishart, y::Array{Float64,1})
  d=length(y)
  mu0 = M.mu
  kappa0 = M.kappa
  TC0 = M.Tchol
  nu0 = M.nu

  kappa = kappa0 + 1
  nu = nu0 + 1
  mu = (kappa0.*mu0 + y) ./ kappa

  Lam0 = TC0[:U]'*TC0[:U]
  Lam = Lam0 + kappa0*1/kappa*(y*y')

  exp(d/2*log(π) + logmvgamma(d,nu/2) - logmvgamma(d,nu0/2) + nu/2*logdet(T0) - nu/2*logdet(Lam) + d/2 * (log(kappa0) - log(kappa)))
end

#I think this is wrong
'''
function marginal_likelihood(M::NormalWishart, y::Array{Float64,1})
  d=length(y)
  posterior = posterior_canon(M,reshape(y,(length(y),1)))
  dfT = posterior.nu-d+1
  covT = transpose(posterior.Tchol[:U])*posterior.Tchol[:U]*(posterior.kappa+1)/posterior.kappa/dfT
  return pdf(GenericMvTDist(dfT, posterior.mu, covT), y)
end
'''