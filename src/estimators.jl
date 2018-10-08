mutable struct GMME{A, CS, MF} <: MathProgBase.AbstractNLPEvaluator
    diff::A
    constraints::CS
    W::Array{Float64,2}
    mf::MF
end

function gmm(mf::Mombastic.OnceDiffMoments,
             x_start::Vector{F},
             itrmgr::Mombastic.IterationManager = TwoStepGMM(),
             bc::Union{BoxConstraints, Nothing} = nothing,
             gc::GenericConstraints = Unconstrained();
             W_initial::Union{Matrix{T}, Nothing} = nothing,
             demean_W::Bool = false,
             fdtype::Symbol = :finitediff,
             solver = IpoptSolver(hessian_approximation = "limited-memory", print_level=0, sb = "yes")) where {T<:AbstractFloat, F}

    n, m, p = size(mf)

    nbc = setboxconstraint(bc, p)::BoxConstraints{Float64}
    W   = setinitialW(W_initial, m)::Array{Float64, 2}

    difftype = fdtype == :forwarddiff ? AutoDiff : FiniteDiff
    gmme = GMME(difftype, gc, W, mf)

    state = IterationState([1], [10.], x0)

    #solver = IpoptSolver(hessian_approximation = "limited-memory", print_level=0, sb = "yes")

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

    while !(finished(itrmgr, state)
        if itern(state) > 1
            esteq!(mf, gmm.inner.x)
            CovarianceMatrices.vcov!(W, mf.gn_val, itr.k)
            demean!(W)
        end
        MathProgBase.optimize!(nl)
    end
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
