struct DiffCache{T, S, N}
    du::Array{T, N}
    dual_du::Array{S, N}
end

# function DiffCache(T, template, ::Type{Val{chunk_size}}, ::Type{Val{tag}}, ::Type{Val{:finite}}) where {chunk_size, tag}
#     DiffCache(similar(template),
#         Array{Float64, vtype(template)}(undef,(repeat([0], outer = vtype(template))...,)...))
# end
#
# function DiffCache(T, template, ::Type{Val{chunk_size}}, ::Type{Val{tag}}, ::Type{Val{:auto}}) where {chunk_size, tag}
#     DiffCache(similar(template),
#         similar(template, ForwardDiff.Dual{tag, Float64, chunk_size}))
# end

function DiffCache(T, template, ::Type{Val{chunk_size}}, ::Type{Val{tag}}) where {chunk_size, tag}
    DiffCache(similar(template),
        similar(template, ForwardDiff.Dual{tag, Float64, chunk_size}))
end



vtype(x::Array{S,N}) where {S, N} = N

getcache(dc::DiffCache, ::Type{T}) where T<:ForwardDiff.Dual = dc.dual_du
getcache(dc::DiffCache, T) = dc.du
