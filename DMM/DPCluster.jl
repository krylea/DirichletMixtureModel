#
# Dirichlet Mixture Models
#
# By Kira Selby
#

#
# This file contains the functions for performing the clustering algorithms
# themselves, using the utilities defined elsewhere. The clustering is done
# according to the algorithms in Neal (2000). To use the functions defined here,
# either select a distribution pair from DistributionPairs.jl or define your own,
# then call the functions with sample data and parameters η (model-specific) and
# α
#

#
# Clustering over class labels using Algorithm 2 from Neal(2000).
#

function DPCluster(Y::Array{Float64,1}, model::Distribution, α::Float64; iters::Int64=5000)
  # Initialize the clusters, returning c and phi
  state::DMMState = DMMState(Y,model)

  # Iterate
  for i in 1:iters
    # Iterate through all Y and update
    state = sampleY(state,model,α)

    # Iterate through all ϕ and update
    for k in keys(state.ϕ)
      state.ϕ[k] = sample_posterior(model,state.Y[k])
    end
  end
  return state
end

function DPCluster(Y::Array{Float64,2}, model::Distribution, α::Float64; iters::Int64=5000)
  # Initialize the clusters, returning c and phi
  state::DMMState = DMMState(Y,model)

  # Iterate
  for i in 1:iters
    # Iterate through all Y and update
    state = sampleY(state,model,α)

    # Iterate through all ϕ and update
    for k in keys(state.ϕ)
      state.ϕ[k] = sample_posterior(model,state.Y[k])
    end
  end
  return state
end

#
# Iterate over all data points in the state, drawing a new cluster for each.
# Returns a new state object.
#

function sampleY(state::DMMState, model::Distribution, α::Float64)
  N=sum(values(state.n))
  nextstate = DMMState(state.ϕ,state.n)

  for k in keys(state.Y)
    for y in state.Y[k]
      nextstate.n[k] -= 1
      K = collect(keys(nextstate.n))

      q=[pdf_likelihood(model,y,nextstate.ϕ[i])*nextstate.n[i]/(N-1+α) for i in K]
      r=marginal_likelihood(model,y)*α/(N-1+α)
      b= 1/(r+sum(q))
      r *= b
      q *= b

      rd=rand()
      p=r
      if rd < p
        ϕk = sample_posterior(model,y)
        addnew!(nextstate, y, ϕk)
      else
        for i in 1:length(K)
          p += q[i]
          if rd < p
            addto!(nextstate, y, K[i])
            break
          end
        end
      end
    end
  end
  cleanup!(nextstate)
  return nextstate
end
