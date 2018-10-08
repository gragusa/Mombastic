using Mombastic
using Test
using Random
using LinearAlgebra

Random.seed!(1)
n, m, p = 1000, 6, 4
const y = randn(n,1);
const x = randn(n,p);
const z = randn(n,m);
gi_val = Array{Float64}(undef, n, m);
 
function gi!(gi_val, x0)
    gi_val .= z.*(y .- x*x0)
    return
end

x0 = fill(0.,p)
mf = Mombastic.OnceDiffMoments(gi!, gi_val, x0)

@test isa(mf, Mombastic.OnceDiffMoments)

@test all(Mombastic.jacobian_esteq!(mf, x0, Val{Mombastic.AutoDiff}) .≈ -z'*x)
@test all(Mombastic.jacobian_esteq!(mf, x0, Val{Mombastic.FiniteDiff}) .≈ -z'*x)

@test all(Mombastic.esteq!(mf, zeros(p)) .≈ z'y)
@test begin
    out = zeros(m)
    Mombastic.esteq!!(out, mf, zeros(p))
    all(out .≈ z'y)
end


@test begin
    xx = randn(p)
    objective_gmm(mf, Matrix(I, m, m), xx) == esteq!(mf, xx)'esteq!(mf, xx)
end

@test begin
    grad_gmm = zeros(p)
    xx = randn(p)
    all(gradient_gmm!(grad_gmm, mf, Matrix(I, m, m), xx) .≈ -(x'z)*sum(z.*(y-x*xx), dims = 1)')
end


gmm(mf. x0, TwoStepGMM())
