mutable struct GMME{A, CS, BC, TI, MF} <: MathProgBase.AbstractNLPEvaluator
    diff::A
    constraints::CS
    boxconstraints::BC
    itermgr::TI
    W::Array{Array{Float64,2},1}
    mf::MF
end

function gmm(mf::Mombastic.OnceDiffMoments,
             x_start::Vector{F},
             itrmgr::Mombastic.IterationManager = TwoStepGMM(),
             bc::Union{BoxConstraints, Nothing} = nothing,
             gc::GenericConstraints = Unconstrained();
             W_initial::Union{Matrix{T}, Nothing} = nothing,
             fdtype::Symbol = :finitediff,
             solver = IpoptSolver(hessian_approximation = "limited-memory", print_level=0, sb = "yes")) where {T<:AbstractFloat, F}

    n, m, p = size(mf)

    nbc = setboxconstraint(bc, p)::BoxConstraints{Float64}
    nW0 = setinitialW(W_initial, m)::Array{Float64, 2}

    diff = fdtype == :forwarddiff ? AutoDiff : FiniteDiff
    gmme = GMME(diff, gc, nbc, itrmgr, [nW0], mf)

    solver = IpoptSolver(hessian_approximation = "limited-memory", print_level=0, sb = "yes")

    nl = MathProgBase.NonlinearModel(solver)
    MathProgBase.loadproblem!(nl,
                               p,
                               0,
                          nbc.lx,
                          nbc.ux,
                       Float64[], ## Lower bounds for constraint
                       Float64[], ## Upper bounds for constraint
                            :Min,
                            gmme)
    MathProgBase.setwarmstart!(nl, float(x_start))
    MathProgBase.optimize!(nl)
    return nl
end

function setboxconstraint(bc, p)
    if bc == nothing
        return BoxConstraints([-Inf for j = 1:p], [+Inf for j = 1:p])
    else
        return copy(bc)
    end
end

function setinitialW(W, m)
    if W == nothing
        return float(Matrix(I, m, m))
    else
        ##
        if !LinearAlgebra.isposdef(W)
            throw(IntitalWeightingNotPosDef())
        end
        return copy(W)
    end
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


struct IntitalWeightingNotPosDef <: Exception end

Base.showerror(io::IO, e::IntitalWeightingNotPosDef) = print(io, "The initial weighting matrix is not positive definite")
