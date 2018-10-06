struct OnceDiffMoments{A, G, T, JC, AC, AG}
    gi::G       ## Moment Function
    #gwi::GW     ## Weighted moment Function
    # gn::GN      ## Avg. Moment Function
    # gwn::GWN    ## Weighted Avg. Moment Function
    # Jn::DG     ## Jacobian function
    # Jwn::DWG
    #Jgwn::DGW   ## Weighted Jacobian function
    gi_val::Array{T, 2}
    gn_val::Array{T, 1}
    gwn_val::Array{T, 1}

    Jn_val::Array{T, 2}
    Jwn_val::Array{T, 2}

    x_gn::Array{T, 1}
    x_gwn::Array{T, 1}
    x_Jn::Array{T, 1}
    x_Jwn::Array{T, 1}

    p_gw::Array{T, 1}
    p_Jw::Array{T, 1}
    size::NTuple{3, Int}

    diffcache::JC
    autocache::AC
    autocgf::AG

    ## Caches
    mxm::Array{T, 2}
    mxk::Array{T, 2}
    mx1::Array{T, 1}

end

struct AutoDiff end
struct FiniteDiff end

function OnceDiffMoments(gi!, gi_val, x0; autodiff = :finite)
    ## Check whether the moment condition can be
    ## correctly evaluated
    try
        gi!(gi_val, x0)
    catch
        throw()
    end

    autotag = any(autodiff .== (:finite, :central)) ? :finite : :auto

    ## Setup containers
    ## Containers need to account for the fact that
    ## autodiff can be forward
    k    = length(x0)
    n, m = size(gi_val)
    x0 = float(x0)

    gn_val   = Array{eltype(x0)}(undef, m)
    gwn_val  = Array{eltype(x0)}(undef, m)
    Jn_val   = Array{eltype(x0)}(undef, m, k)
    Jwn_val  = Array{eltype(x0)}(undef, m, k)

    autocfg = ForwardDiff.JacobianConfig(nothing, gn_val, x0)
    chunk_size = ForwardDiff.chunksize(autocfg)
    autocache = DiffCache(Float64, gi_val, Val{chunk_size}, Val{Nothing})

    diffcache = DiffEqDiffTools.JacobianCache(Array{Float64}(undef, k),
                                           Array{Float64}(undef, m),
                                           Array{Float64}(undef, m),
                                           Val{:central})



    OnceDiffMoments(gi!,
                  gi_val,
                  gn_val,
                  gwn_val,
                  Jn_val,
                  Jwn_val,
                  copy(x0), copy(x0), copy(x0), copy(x0),
                  Array{Float64, 1}(undef, n),
                  Array{Float64, 1}(undef, n),
                  (n, m, k), diffcache, autocache, autocfg,
                  Array{Float64, 1}(undef, m, m),
                  Array{Float64, 1}(undef, m, k),
                  Array{Float64, 1}(undef, m, 1))
end


function esteq!(mf::Mombastic.OnceDiffMoments, x0)
    gi_val = getcache(mf.Acache, eltype(x0))
    mf.gi(gi_val, x0)
    sum!(mf.gn_val', gi_val)'
end

function esteq!!(gn_val, mf::Mombastic.OnceDiffMoments, x0)
    gi_val = getcache(mf.Acache, eltype(x0))
    mf.gi(gi_val, x0)
    sum!(gn_val', gi_val)'
end

function jacobian_esteq!(mf::Mombastic.OnceDiffMoments, x0, ::Type{Val{FiniteDiff}})
    #gi_val = getcache(mf.Acache, eltype(x0))
    f!(gn_val, x0) = esteq!(gn_val, mf, x0)
    DiffEqDiffTools.finite_difference_jacobian!(mf.Jn_val, f!, x0, mf.Jcache)
end

function jacobian_esteq!(mf::Mombastic.OnceDiffMoments, x0, ::Type{Val{AutoDiff}})
    #gi_val = getcache(mf.Acache, eltype(x0))
    f!(gn_val, x0) = esteq!!(gn_val, mf, x0)
    ForwardDiff.jacobian!(mf.Jn_val, f!, mf.gn_val, x0, mf.Acgf)
end


function gmmobjective(mf::Mombastic.OnceDiffMoments, W, x0)
    g = Mombastic.esteq!(mf, x0)
    #mul!(mf.mx1, W, g)
    g'*W*g
end

function gradient_gmm!(grad_gmm, mf::Mombastic.OnceDiffMoments, W, x0)
    Mombastic.esteq!(mf, x0)
    Mombastic.jacobian_esteq(mf, x0)
    mul!(mf.mxk, dg', W)
    mul!(grad_gmm, mf.mxk, g)
end


Base.size(mf::Mombastic.OnceDiffMoments) = mf.size



function arediff(p, w)
    @inbounds for j in eachindex(p)
        if p[j] != w[j]
            return true
        end
    end
    return false
end







#
# struct OnceDiffMoments{G, GN, GWN, DG, DWG, GX, JX, TX}
#     gi::G       ## Moment Function
#     #gwi::GW     ## Weighted moment Function
#     gn::GN      ## Avg. Moment Function
#     gwn::GWN    ## Weighted Avg. Moment Function
#     Jn::DG     ## Jacobian function
#     Jwn::DWG
#     #Jgwn::DGW   ## Weighted Jacobian function
#
#     gn_val::GX
#     gwn_val::GX
#
#     Jn_val::JX
#     Jwn_val::JX
#
#     x_gn::TX
#     x_gwn::TX
#     x_Jn::TX
#     x_Jwn::TX
#
#     p_gw::TX
#     p_Jw::TX
#     size::NTuple{3, Int}
# end
#
# function OnceDiffMoments(gi!, gi_val, x0; autodiff = :finite)
#     ## Check whether the moment condition can be
#     ## correctly evaluated
#     try
#         gi!(gi_val, x0)
#     catch
#         throw()
#     end
#
#     autotag = any(autodiff .== (:finite, :central)) ? :finite : :auto
#
#     ## Setup containers
#     ## Containers need to account for the fact that
#     ## autodiff can be forward
#     k    = length(x0)
#     n, m = size(gi_val)
#     x0 = float(x0)
#
#     ## Construct Momentfunction
#     ## TODO: Add smoothing
#
#     gn! = (gn_val, gi_cache, gi!, x0) -> begin
#         gi_val = getcache(gi_cache, eltype(x0))
#         gi!(gi_val, x0)
#         sum!(gn_val', gi_val)'
#     end
#
#     gwn! =  (gn_val, gi_cache, gi!, p, x0) -> begin
#         gi_val = getcache(gi_cache, eltype(x0))
#         gi!(gi_val, x0)
#         broadcast!(*, gi_val, gi_val, p)
#         sum!(gn_val', gi_val)'
#     end
#     gn_val   = Array{eltype(x0)}(undef, m)
#     gwn_val  = Array{eltype(x0)}(undef, m)
#     Jn_val   = Array{eltype(x0)}(undef, m, k)
#     Jwn_val  = Array{eltype(x0)}(undef, m, k)
#     cfg = ForwardDiff.JacobianConfig(nothing, gn_val, x0)
#     chunk_size = ForwardDiff.chunksize(cfg)
#     gi_cache = DiffCache(Float64, gi_val, Val{chunk_size}, Val{Nothing}, Val{autotag})
#
#     f!(gn_val, x0) = gn!(gn_val, gi_cache, gi!, x0)
#     fw!(gn_val, p, x0) = gwn!(gn_val, gi_cache, gi!, p, x0)
#
#     if any(autodiff .== (:finite, :central))
#         ## Using finite difference derivatives
#         ## Construct Jacobiag cache
#         Jcache = DiffEqDiffTools.JacobianCache(Array{Float64}(undef, k),
#                                                Array{Float64}(undef, m),
#                                                Array{Float64}(undef, m),
#                                                Val{:central})
#         ## Construct gi_cache
#         ## This is mooth
#
#
#         Jn! = (Jn_val, gn_val, x0) -> begin
#             DiffEqDiffTools.finite_difference_jacobian!(Jn_val, f!, x0, Jcache)
#         end
#
#         Jwn! = (Jn_val, p, x0) -> begin
#             DiffEqDiffTools.finite_difference_jacobian!(Jn_val,
#                             (gn_val, x0) -> fw!(gn_val, p, x0), x0, Jcache)
#         end
#
#     elseif autodiff .== :auto
#         #f!(gn_val, x0) = gn!(gn_val, gi_cache, gi!, x0)
#
#         Jn! = (Jn_val, gn_val, x0) -> begin
#             ForwardDiff.jacobian!(Jn_val, f!, gn_val, x0, cfg)
#         end
#
#         Jwn! = (Jwn_val, gn_val, p, x0) -> begin
#             ForwardDiff.jacobian!(Jwn_val,
#                             (gn_val, x0) -> fw!(gn_val, p, x0), gn_val, x0, cfg)
#         end
#     end
#
#     gn!(gn_val, gi_cache, gi!, x0)
#     Jn!(Jn_val, gn_val, x0)
#
#
#     OnceDiffMoments(gi!,
#                   (gn_val, x0) -> gn!(gn_val, gi_cache, gi!, x0),
#                   (gwn_val, p, x0) -> gwn!(gn_val, gi_cache, gi!, p, x0),
#                   Jn!, Jwn!,
#                   gn_val, gwn_val,
#                   Jn_val, Jwn_val,
#                   copy(x0), copy(x0), copy(x0), copy(x0),
#                   Array{Float64, 1}(undef, n),
#                   Array{Float64, 1}(undef, n),
#                   (n, m, k))
# end

# function jacobian!(mf::Mombastic.OnceDiffMoments, x0)
#     #if mf.x_Jn != x0
#         mf.Jn(mf.Jn_val, mf.gn_val, x0)
#     #    copyto!(mf.x_Jn, x0)
#     #end
#     #return mf.Jn_val
# end
#
# function jacobian!(mf::Mombastic.OnceDiffMoments, p, x0)
#     if (mf.x_Jn != x0 & arediff(mf.p_Jw, p))
#         mf.Jwn(mf.Jwn_val, mf.gwn_val, x0)
#         copyto!(mf.x_Jwn, x0)
#         copyto!(mf.p_Jw, p)
#     end
#     return mf.Jwn_val
# end
#
# function momentfunction!(mf::Mombastic.OnceDiffMoments, x0)
#     if mf.x_gn != x0
#         mf.gn(mf.gn_val, x0)
#         copyto!(mf.x_gn, x0)
#     end
#     return mf.gn_val
# end
#
# function momentfunction!(mf::Mombastic.OnceDiffMoments, p, x0)
#     if (mf.x_gwn != x0 & arediff(mf.p_gw, p))
#         mf.gn(mf.gwn_val, x0)
#         copyto!(mf.x_gwn, x0)
#         copyto!(mf.p_gw, p)
#     end
#     return mf.gwn_val
# end
