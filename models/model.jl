#
# Dirichlet Mixture Models
#
# By Kira Selby
#

#
# The code in this file provides a container for distribution pairs to use in
# Dirichlet Mixture Models, as well as several examples for common conjugate
# pairs of distributions.
#


#
# The type Model defines a conjugate pair of distributions (F, G0) for our DMM.
# F(θ) is the base distribution for the clusters. G0(η) is the base measure for the
# Dirichlet Process which defines the prior on θ. η are the hyperparameters
# for the model.
#
# The Dirichlet mixture code requires the ability to:
#   a) Calculate the likelihood F(y,θ)
#   b) Sample from the posterior of G0 | Y,η
#   c) Calculate the integral R=∫F(y|θ)dG0(θ|η)
#
# The pdf function must return the pdf of the likelihood evaluated at x, with
# parameters θ.
# The posterior function must take y or Y as a parameter and return a sample
# from the posterior of G0 | Y,η
# The expectation functionmust take y as a parameter and return the value
# of ∫F(y|θ)dG0(θ|η)

#
# Each distribution pair should define a constructor to create an instance of
# model from an object of that distribution, defining the functions appropriately.
#
