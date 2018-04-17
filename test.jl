include("DMM.jl")

using Atom
using DMM

import Distributions:
        UnivariateDistribution,
        MultivariateDistribution,
        Exponential,
        Normal,
        MvNormal,
        rand


function generateSamples(::Type{T}, thetas::Tuple, numSamples::Array{Int64,1}; shuffled=true) where T <: UnivariateDistribution
    @assert length(thetas) == length(numSamples)
    M=length(thetas)
    N=sum(numSamples)
    data=zeros(Float64, N)
    n=0
    for i in 1:M
        n_i=numSamples[i]
        dist_i=T(thetas[i]...)
        for j in 1:n_i
            data[n+j]=rand(dist_i)
        end
        n+=n_i
    end
    if shuffled
        shuffle!(data)
    end
    return data
end
function generateSamples(::Type{T}, thetas::Tuple, numSamples::Array{Int64,1}, d::Int64; shuffled=true) where T <: MultivariateDistribution
    @assert length(thetas) == length(numSamples)
    M=length(thetas)
    N=sum(numSamples)
    data=zeros(Float64, d, N)
    n=0
    for i in 1:M
        n_i=numSamples[i]
        dist_i=T(thetas[i]...)
        for j in 1:n_i
            data[:,n+j]=rand(dist_i)
        end
        n+=n_i
    end
    if shuffled
        data=data[:,shuffle(1:end)]
    end
    return data
end

srand(1010)

#   Multivariate Test Code
T1 = [1. 0.; 0. 1.]
T2 = [1. 0.5; 0.5 1.]
μ1 = [5.,0.]
μ2 = [0.,-5.]

params = ((μ1, T1), (μ2, T2))
data = generateSamples(MvNormal, params, [100, 100], 2)

α=1.0

D=size(data,1)
U = DMM.MultivariateNormalModel(D)
s = DMM.DPCluster(data,U,1.0)

#=

#   Univariate Test Code

params = ((0.0, 2.0), (3.0, 0.5))
data = generateSamples(Normal, params, [100, 100], 2)

α=1.0

D=size(data,1)
U = DMM.MultivariateNormalModel(D)
s = DMM.DPCluster(data,U,1.0)
=#
