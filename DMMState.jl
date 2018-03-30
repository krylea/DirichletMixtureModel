#
# Dirichlet Mixture Models
#
# By Kira Selby
#

#
# The code in this file provides utilities for storing and manipulating
# the current state of the Markov Chain for use in performing Gibbs Sampling
# from a Dirichlet Mixture Model (see DPCLuster.jl)
#

#
# Check if two vectors are equal (within relative error ϵ)
#
function isequalϵ(a, b, ϵ=1e-6)
  @assert length(a)==length(b)
  if length(a)==1
    return abs(a[1]-b[1])/min(a[1],b[1]) < ϵ
  else
    for i in length(a)
      if abs(a[i]-b[i])/min(a[i],b[i]) >= ϵ
        return false
      end
    end
    return true
  end
end

#
# Stores the current state of the clusters in the Markov Chain.
#

struct DMMState
  ϕ::Dict{Int64,Tuple}
  Y::Dict{Int64,AbstractArray{Float64}}
  n::Dict{Int64,Int64}
end

#
# Constructors to create an empty state, build a new state from an existing state,
# or create a new state from unlabelled data
#

function DMMState()
  return DMMState(Dict{Int64,Array{Float64,1}}(),Dict{Int64,Tuple}(), Dict{Int64,Int64}())
end

function DMMState(ϕ::Dict{Int64,Tuple}, n::Dict{Int64,Int64})
  return DMMState(ϕ,Dict{Int64,Array{Float64,1}}(), n)
end

function DMMState(Y::Array{Float64,1}, model::Distribution)
  N=length(Y)
  ϕ=Dict{Int64,Tuple}()
  Ynew=Dict{Int64,Array{Float64,1}}()
  n=Dict{Int64,Int64}()
  for i in 1:N
    ϕ[i] = sample_posterior(model,Y[i])
    Ynew[i] = [Y[i]]
    n[i] = 1
  end
  return DMMState(ϕ,Ynew,n)
end
function DMMState(Y::Array{Float64,2}, model::Distribution)
  N,d=size(Y)
  ϕ=Dict{Int64,Tuple{AbstractVector,AbstractMatrix}}()
  Ynew=Dict{Int64,Array{Float64,2}}()
  n=Dict{Int64,Int64}()
  for i in 1:N
    ϕ[i] = sample_posterior(model,Y[i,:])
    Ynew[i] = reshape(Y[i,:],(1,d))
    n[i] = 1
  end
  return DMMState(ϕ,Ynew,n)
end

#
# Add new data to the state (when the label is completely unknown)
#

function add!(state::DMMState, yi::Float64, ϕi::Array{Float64,1})
  added=false
  for (k,v) in state.Y
    if isequalϵ(ϕi,state.ϕ[i])
      addto!(state, yi, i)
      added=true
    end
  end
  if added == false
    addnew!(state,yi,ϕi)
  end
end

function addnew!(state::DMMState, yi::Float64, ϕi::Tuple)
  i = 1
  K = keys(state.n)
  while i in K
    i += 1
  end
  state.Y[i] = [yi]
  state.n[i] = 1
  state.ϕ[i] = ϕi
  return i
end

#
# Add new data to the state (when it is known that the data belongs to a new cluster)
#

function addnew!(state::DMMState, yi::Float64, ϕi::Tuple, i::Int64)
  state.Y[i] = [yi]
  state.n[i] = 1
  state.ϕ[i] = ϕi
end

#
# Add new data to the state (when the cluster label is known)
#

function addto!(state::DMMState, yi::Float64, i::Int64)
  @assert i in keys(state.ϕ)
  @assert i in keys(state.n)
  state.n[i]+=1
  if i in keys(state.Y)
    append!(state.Y[i], yi)
  else
    state.Y[i] = [yi]
  end
end

#
# Clean up state by removing empty clusters. Y is assumed to already be accurate.
#
function cleanup!(state::DMMState)
  for (k,v) in state.n
    if v==0
      delete!(state.n, k)
      delete!(state.ϕ, k)
    end
  end
end
