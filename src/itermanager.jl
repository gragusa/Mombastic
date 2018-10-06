# ------------------ #
# Iteration managers #
# ------------------ #

abstract type IterationManager end

struct OneStepGMM <: IterationManager
    k::CovarianceMatrices.RobustVariance
    demean::Bool
end

struct TwoStepGMM <: IterationManager
    k::CovarianceMatrices.RobustVariance
    demean::Bool
end

struct IterativeGMM <: IterationManager
    k::CovarianceMatrices.RobustVariance
    demean::Bool
    tol::Float64
    maxiter::Int
end

# kwarg constructors with default values
OneStepGMM(;k::CovarianceMatrices.RobustVariance=HC0(), demean::Bool = false) = OneStepGMM(k, demean)
TwoStepGMM(;k::CovarianceMatrices.RobustVariance=HC0(), demean::Bool = false) = TwoStepGMM(k, demean)

OneStepGMM(k::CovarianceMatrices.RobustVariance) = OneStepGMM(k, false)
TwoStepGMM(k::CovarianceMatrices.RobustVariance) = TwoStepGMM(k, false)

function IterativeGMM(;k::CovarianceMatrices.RobustVariance=HC0(), demean::Bool = false, tol::Float64=1e-12, maxiter::Int=500)
    IterativeGMM(k, demean, tol, maxiter)
end

mutable struct IterationState
    n::Array{Int, 1}
    change::Array{Float64, 1}
    prev::Array{Float64, 1}  # previous value
end

finished(::OneStepGMM, ist::IterationState) = ist.n[1] > 1
finished(::TwoStepGMM, ist::IterationState) = ist.n[1] > 2

function finished(mgr::IterativeGMM, ist::IterationState)
    ist.n[1] > mgr.maxiter[1] || abs(ist.change[1]) <= mgr.tol
end

function next!(mgr::IterationManager, ist::IterationState)
    if !finished(mgr, ist)
        ist.n[1] += 1
    end
    return
end
