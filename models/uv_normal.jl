

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
