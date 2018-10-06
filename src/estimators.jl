mutable struct GMMEstimator{BC, CS, MF, TI} <: MathProgBase.AbstractNLPEvaluator
    bc::BC
    cs::CS
    mf::MF
    mgr::TI
    W::Tuple
end

struct MDResults{D, M}
    divergence::D
    measure::M
end

function gmm(mf::Mombastic.OnceDiffMoments, x_start, bc::BoxConstraints, mgr::Mombastic.IterationManager = TwoStepGMM(), cs::GenericConstraints = Unconstrained(); W_initial = I)
    n, m, p = size(mf)
    gmme = GMMEstimator(cs, bc, mf, mgr, (W_initial,))
    nl = MathProgBase.NonlinearModel(IpoptSolver(hessian_approximation = "limited-memory", print_level=0, sb = "yes"))
    MathProgBase.loadproblem!(nl, p, 0, bc.lx, bc.ux, Float64[], Float64[], :Min, gmme)
    MathProgBase.setwarmstart!(nl, x_start)
    MathProgBase.optimize!(nl)
    return nl
end
#gmm(mf::OnceDiffMoments, x_start; divergence, solver:: = , constraint::Constraints)
#md(mf, x_start, Constraints, Measure; solver)


#iteratorstate(itr::IterationManager, s::IterationState) = s.n[1]

# abstract type Measure end
#
# struct EmpiricalMeasure <: BaseMeasure end
#
# struct CustomMeasure{T<:AbstractFloat} <: BaseMeasure
#     w::Array{T,1}
# end
