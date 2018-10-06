### Constraints
#
# Box Constraints are specified by the user as
#    lx_i ≤   x[i]  ≤ ux_i  # variable (box) constraints
# and become equality constraints with l_i = u_i. ±∞ are allowed for l
# and u, in which case the relevant side(s) are unbounded.
#


struct BoxConstraints{T}
    lx::Vector{T}
    ux::Vector{T}
    eqx::Vector{Int}   # index-vector of equality-constrained x (not actually variable...)
end

abstract type GenericConstraints end

function BoxConstraints(lx::AbstractArray, ux::AbstractArray)
    eqx = findall(lx.==ux)
    BoxConstraints(float(lx), float(ux), eqx)
end
# Constraints are specified by the user as
#    lc_i ≤ c(x)[i] ≤ uc_i  # linear/nonlinear constraints
struct Constraints{F, J, T} <: GenericConstraints
    c!::F        # c!(storage, x) stores the value of the constraint-functions at x
    jacobian!::J # jacobian!(storage, x) stores the value of the jacobain of c!
    lc::Vector{T}
    uc::Vector{T}
    sc::Vector{T}
    sj::Vector{T}
end

struct Unconstrained <: GenericConstraints end











# The user supplies functions to calculate c(x) and its derivatives.
#
# Of course we could unify the box-constraints into the
# linear/nonlinear constraints, but that would force the user to
# provide the variable-derivatives manually, which would be silly.
#
# This parametrization of the constraints gets "parsed" into a form
# that speeds and simplifies the IPNewton algorithm, at the cost of many
# additional variables. See `parse_constraints` for details.
