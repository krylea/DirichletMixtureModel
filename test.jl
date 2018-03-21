include("DMM.jl")

using Atom
using DMM

#Univariate test code
'''
gaussians = [(3.,3.), (-1.,0.5), (10,6)]

function samplegaussian(n, args)
    mu,sigma = args
    return randn(n)*sigma + mu
end

N=[100,400,100]
α=1.

μ0 = 0.
n0 = 1e-8
α0 = 1.
β0 = 1.

data=Float64[]
for i in 1:length(gaussians)
    append!(data,samplegaussian(N[i],gaussians[i]))
end

U = ConjugatePriors.NormalGamma(μ0,n0,α0,β0)

s = DMM.DPCluster(data,U,α)
'''

#Multivariate test code

cv1 = [1. 0.;0. 1.]
cv2 = [1. 0.5;0.5 1.]
mu1 = [100.; 0.]
mu2 = [0.; 100.]

mvn1 = Distributions.MvNormal(mu1,cv1)
mvn2 = Distributions.MvNormal(mu2,cv2)

mvns = [mvn1,mvn2]

N=100

data = Array{Float64}(N*length(mvns),length(mu1))
for i in 1:length(mvns)
    for j in 1:N
        data[N*(i-1)+j,:]=rand(mvns[i])
    end
end

d=2
nu0=2.
kappa0=1.e-8
T0 = [1. 0.; 0. 1.]
mu0=[0.;0.]

U=ConjugatePriors.NormalWishart(mu0,kappa0,T0,nu0)
s=DMM.DPCluster(data,U,α)
