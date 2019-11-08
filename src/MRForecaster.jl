module MRForecaster

using HDF5
using Distributions, StatsBase
using Printf

# Constants
# Model range
const MODEL_MIN_MASS = 3e-4
const MODEL_MIN_LOGMASS = -3.522
const MODEL_MAX_MASS = 3e5
const MODEL_MAX_LOGMASS = 5.477
const MODEL_MIN_RADIUS = 1e-1
const MODEL_MAX_RADIUS = 1e2
const N_POPS = 4

# Conversion factors
const MEARTH_MJUP = 317.828
const MEARTH_MSUN = 333060.4
const REARTH_RJUP = 11.21
const REARTH_RSUN = 109.2

# Load hyperparameters
datapath = joinpath(dirname(pathof(MRForecaster)), "..", "data")
hyper_file = joinpath(datapath, "fitting_parameters.h5")
all_hyper = h5read(hyper_file, "hyper_posterior")

include("utils.jl")

"""
    mass_to_radius(mass::AbstractVector, unit, classify)

Forecast a radius distribution given a mass distribution.
"""
function mass_to_radius(mass::Union{AbstractVector{<:AbstractFloat}, AbstractFloat};
                        unit="Earth", classify=false)
    # Convert mass units
    if unit == "Jupiter"
        mass *= MEARTH_MJUP
    elseif unit == "Sun" || unit == "Solar"
        mass*= MEARTH_MSUN
    elseif unit != "Earth"
        throw(ArgumentError(unit, "Mass unit must be Jupiter, Sun, or Earth."))
    end

    # Ensure we are within model mass range
    if (minimum(mass) < MODEL_MIN_MASS) || (maximum(mass) > MODEL_MAX_MASS)
        throw(ArgumentError(mass, "Mass outside model range"))
    end

    logm = log10.(mass)
    sample_size = length(mass)
    logr = Vector{Float64}(undef, sample_size)

    # Draw random realizations of the posterior
    hyper_params = all_hyper[:, rand(1:size(all_hyper)[2], sample_size)]

    if classify
        classification(logm, hyper_params[end-2:end,:])
    end

    # Add random dispersion around mean
    prob = rand(sample_size)

    for i = 1:sample_size
        logr[i] = piece_linear(hyper_params[:,i], logm[i], prob[i])[1]
    end

    radius = 10 .^ logr

    # Convert back to right unit
    if unit == "Jupiter"
        radius /= REARTH_RJUP
    elseif unit == "Sun" || unit == "Solar"
        radius /= REARTH_RSUN
    end

    return radius

end # mass_to_radius

"""
    Mstat2R(mass_mean::AbstractFloat, mass_std::AbstractFloat)

Forecast a radius mean and upper/lower uncertainties given a mass
mean and standard deviation.

Assumes a normal distribution with the mean and standard deviation,
truncated at the mass range limit of the model.
"""
function Mstat2R(mass_mean::AbstractFloat,
                 mass_std::AbstractFloat,
                 unit="Earth", sample_size::Int=1000, classify::Bool=false)
    # Convert mass units
    if unit == "Jupiter"
        mass_mean *= mearth2mjup
        mass_std *= mearth2mjup
    elseif unit == "Sun" || unit == "Solar"
        mass_mean *= mearth2msun
        mass_std *= mearth2msun
    elseif unit != "Earth"
        throw(ArgumentError(unit, "Mass unit must be Jupiter, Sun, or Earth."))
    end

    # Draw random samples
    mass_dist = TruncatedNormal(mass_mean, mass_std, MODEL_MIN_MASS, MODEL_MAX_MASS)
    mass_rvs = rand(mass_dist, sample_size)

    radius = mass_to_radius(mass_rvs, unit="Earth", classify=classify)

    # Convert to original units
    if unit == "Jupiter"
        radius /= REARTH_RJUP
    elseif unit == "Sun" || unit == "Solar"
        radius /= REARTH_RSUN
    end

    r_stat = quantile(radius, [0.5, 0.841, 0.159])

    return r_stat[1], r_stat[2:3] .- r_stat[1]
end # Mstat2R

"""
    radius_to_mass(radius::AbstractVector, unit, grid_size, classify)

Forecast a mass given a radius or radius distribution.
"""
function radius_to_mass(radius::Union{AbstractVector{<:AbstractFloat}, AbstractFloat};
                        unit="Earth", grid_size::Int=1000, classify::Bool=false)

    # Convert radius units
    if unit == "Jupiter"
        radius *= REARTH_RJUP
    elseif unit == "Sun" || unit == "Solar"
        radius *= REARTH_RSUN
    elseif unit != "Earth"
        throw(ArgumentError(unit, "Radius unit must be Jupiter, Sun, or Earth."))
    end

    # Ensure we are within model radius range
    if (minimum(radius) < MODEL_MIN_RADIUS) || (maximum(radius) > MODEL_MAX_RADIUS)
        throw(ArgumentError(string(radius), "Radius outside model range"))
    end
    # Generate a sample grid
    if grid_size < 10
        throw(ArgumentError(string(grid_size), "Grid size must be at least 10."))
    end

    logr = log10.(radius)
    sample_size = length(radius)
    logm = Vector{Float64}(undef, sample_size)

    # Draw random realizations of the posterior
    hyper_params = all_hyper[:, rand(1:size(all_hyper)[2], sample_size)]

    logm_grid = LinRange(MODEL_MIN_LOGMASS, MODEL_MAX_LOGMASS, grid_size)

    for i = 1:sample_size
        prob = prob_R_given_M(logr[i], logm_grid, hyper_params[:,i])
        logm[i] = sample(logm_grid, Weights(prob))
    end

    mass = 10 .^ logm

    # Convert back to right unit
    if unit == "Jupiter"
        mass /= MEARTH_MJUP
    elseif unit == "Sun" || unit == "Solar"
        mass /= MEARTH_MSUN
    end

    return mass

end # radius_to_mass

"""
    Rstat2M(radius_mean, radius_std)

Forecast a mass mean and upper/lower uncertainties given a radius mean
and standard deviation.

Assumes a normal distribution with the mean and standard deviation,
truncated at the radius range limit of the model.
"""
function Rstat2M(radius_mean::AbstractFloat,
                 radius_std::AbstractFloat,
                 unit="Earth", sample_size::Int=1000, grid_size::Int=1000,
                 classify::Bool=false)
    # Convert radius units
    if unit == "Jupiter"
        radius_mean *= REARTH_RJUP
        radius_std *= REARTH_RJUP
    elseif unit == "Sun" || unit == "Solar"
        radius_mean *= REARTH_RSUN
        radius_std *= REARTH_RSUN
    elseif unit != "Earth"
        throw(ArgumentError(unit, "Radius unit must be Jupiter, Sun, or Earth."))
    end

    # Draw random samplse
    radius_dist = TruncatedNormal(radius_mean, radius_std, MODEL_MIN_RADIUS, MODEL_MAX_RADIUS)
    radius_rvs = rand(radius_dist, sample_size)

    mass = radius_to_mass(radius_rvs, unit="Earth", classify=classify)

    # Convert back to original units
    if unit == "Jupiter"
        mass /= MEARTH_MJUP
    elseif unit == "Sun" || unit == "Solar"
        mass /= MEARTH_MSUN
    end

    m_stat = quantile(mass, [0.5, 0.841, 0.159])

    return m_stat[1], m_stat[2:3] .- m_stat[1]
end # Rstat2M


"""
    classification(logm, trans)

Classify the planet according to its mass. Units should be Earth units.
"""
function classification(logm::Union{AbstractVector{<:AbstractFloat}, AbstractFloat},
                        trans::AbstractArray{<:AbstractFloat})
    println("Classify")
    count = Vector{Float64}(undef, N_POPS)
    sample_size = length(logm)

    for iclass in 1:N_POPS
        for isample in 1:sample_size
            ind = indicate(logm[isample], trans[:,isample], iclass)
            count[iclass] += ind
        end
    end

    prob = count / sum(count) * 100.
    @printf("Terran %.1f%%, Neptunian %.1f%%, Jovian %.1f%%, Star %.1f%%\n",
            prob[1], prob[2], prob[3], prob[4])
    return
end

end # module
