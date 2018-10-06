using Test
using Random

Random.seed!(1)
n, m, p = 1000, 20, 8
y = randn(1000,1);
x = randn(1000,p);
z = randn(1000,m);
gi_val = Array{Float64}(undef, n, m);


function gi!(gi_val, x0)
    gi_val .= z.*(y .- x*x0)
    return
end



x0 = fill(0,p)

mf = Mombastic.OnceDiffMoments(gi!, gi_val, x0)

@test isa(mf, Mombastic.OnceDiffMoments)

@test all(Mombastic.jacobian!(mf, x0) .≈ -z'*x)
@test all(Mombastic.momentfunction!(mf, x0) .≈ z'y)

@test all(Mombastic.jacobian!(mf, rand(8)) .≈ -z'*x)
@test all(Mombastic.momentfunction!(mf, x0) .≈ z'y)


mgr = Mombastic.TwoStepGMM()
cs  = NLSolversBase.OnceDifferentiableConstraints([-10 for j in 1:p], [10 for j in 1:p])

m = gmm(mf, x0, mgr, cs)

nl = MathProgBase.NonlinearModel(Ipopt.IpoptSolver())

MathProgBase.loadproblem!(nl, p, 0, getgmmbounds(m.c)..., Float64[], Float64[], :Min, m)

MathProgBase.setwarmstart!(nl, x0)

MathProgBase.optimize!(nl)



function getgmmbounds(cs::OnceDifferentiableConstraints)
    if NLSolversBase.nconstraints_x(cs.bounds) == 0
        ([-Inf for j in 1:p], [Inf for j in 1:p])
    else
        (reshape(cs.bounds.bx, 2, 8)[1,:], reshape(cs.bounds.bx, 2, 8)[2,:])
    end
end
