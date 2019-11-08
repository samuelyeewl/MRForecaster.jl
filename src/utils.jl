"""
	Utility functions for MRForecaster.jl
"""

"""
	split_hyper_linear(hyper)

Splits a set of hyperparameters into an array of parameters
for a broken power law.

Returns an array of intercepts, slopes, dispersions and transition points.

"""
function split_hyper_linear(hyper::AbstractVector)
	c0, slope, sigma, trans =
	hyper[1], hyper[2:1+N_POPS], hyper[2+N_POPS:1+2*N_POPS],
	hyper[2+2*N_POPS:end]

	# Compute individual intercepts
	c = Vector{Float64}(undef, N_POPS)
	c[1] = c0
	for i = 2:N_POPS
		c[i] = c[i-1] + trans[i-1]*(slope[i-1]-slope[i])
	end

	return c, slope, sigma, trans
end


"""
	piece_linear(hyper, M, prob_R)

Generate a radius from the given mass M from the piecewise model
specified by the hyperparameters in hyper, with a dispersion
parameter of prob_R.
"""
function piece_linear(hyper, M, prob_R)
	c, slope, sigma, trans = split_hyper_linear(hyper)
	if !isa(M, AbstractVector)
		mval = M
		M = Vector{Float64}(undef, 1)
		M[1] = mval
		prob_rval = prob_R
		prob_R = Vector{Float64}(undef, 1)
		prob_R[1] = prob_rval
	end
	R = similar(M)
	for i in 1:N_POPS
		# Determine the masses in population i
		ind = indicate(M, trans, i)
		mu = c[i] .+ (M[ind] .* slope[i])
		R[ind] = quantile.(Normal.(mu, sigma[i]), prob_R[ind])
	end
	return R
end


# Preallocate vectors
const trans_compare = Vector{Float64}(undef, N_POPS+1)
trans_compare[1] = -Inf
trans_compare[end] = Inf
"""
	indicate(M, trans, i)

Determine the masses in the array M that belong to
population i, as specified by the transition parameter trans.
"""
function indicate(M, trans, i)
	# Create array for comparison
	trans_compare[2:N_POPS] = trans
	ind = (M .> trans_compare[i]) .& (M .<= trans_compare[i+1])
	return ind
end

"""
	prob_R_given_M(radii, M, hyper)

Given an array of masses and set of hyperparameters,
return the probability of the radius.
"""
function prob_R_given_M(radii, M, hyper)
	# Get hyperparameteers
	c, slope, sigma, trans = split_hyper_linear(hyper)

	prob = similar(M)
	for i in 1:N_POPS
		# Determine the masses in population i
		ind = indicate(M, trans, i)
		mu = c[i] .+ (M[ind] .* slope[i])
		prob[ind] = pdf.(Normal.(mu, sigma[i]), radii)
	end

	prob /= sum(prob)
	return prob
end

