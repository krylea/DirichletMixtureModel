

function pdf_likelihood(M::Gamma, y::Float64, θ::Tuple{Float64})
  pdf(Exponential(θ[1]), y)
end

function sample_posterior(M::Gamma, Y::Array{Float64,1})
  rand(posterior_canon(M,suffstats(Exponential,Y)))
end
function sample_posterior(M::NormalGamma, y::Float64)
  rand(posterior_canon(M,suffstats(Exponential,[y])))
end

function marginal_likelihood(M::Gamma, y::Float64)
  M.α/(M.θ)^(2*α+1) * (1+y*M.θ)^(M.α+1)
end


function clusterExp(Y::Array{Float64,1};α::Float64=1.0, α0::Float64=2.0, θ0::Float64=0.5, iters::Int64=5000)
  U = Gamma(α0,θ0)
  DMM.DPCluster(Y,U,α,iters=iters)
end
