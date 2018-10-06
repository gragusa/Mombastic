module Mombastic

using DiffEqDiffTools
using ForwardDiff
using NLSolversBase
using CovarianceMatrices
using Divergences
using MathProgBase
using LinearAlgebra
using Ipopt

include("types.jl")
include("momfun.jl")
include("itermanager.jl")
include("constraints.jl")
include("estimators.jl")
include("mathprogbase/gmm.jl")

export OnceDiffMoments, esteq!, esteq!!, jacobian_esteq!, objective_gmm, gradient_gmm!

end # module
