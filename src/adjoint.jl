using SparseArrays
import Mooncake
using Mooncake: CoDual, NoFData, NoTangent, NoRData, MutableTangent, DefaultCtx, ReverseMode, @is_primitive, @zero_adjoint
using LinearAlgebra

# Help Mooncake handle complex types and avoid unnecessary tag analysis
Mooncake.tangent_type(::Type{<:SparseArrays.CHOLMOD.Factor}) = Mooncake.NoTangent
Mooncake.tangent_type(::Type{NetworkTopology}) = Mooncake.NoTangent
Mooncake.tangent_type(::Type{LoadData}) = Mooncake.NoTangent
Mooncake.tangent_type(::Type{GeometryData}) = Mooncake.NoTangent
Mooncake.tangent_type(::Type{AnchorInfo}) = Mooncake.NoTangent
Mooncake.tangent_type(::Type{OptimizationParameters}) = Mooncake.NoTangent
Mooncake.tangent_type(::Type{OptimizationState}) = Mooncake.NoTangent
Mooncake.tangent_type(::Type{<:OptimizationProblem}) = Mooncake.NoTangent
Mooncake.tangent_type(::Type{<:ObjectiveWrapper}) = Mooncake.NoTangent

@is_primitive DefaultCtx ReverseMode Tuple{typeof(update_factorization!), FDMContext}
@is_primitive DefaultCtx ReverseMode Tuple{typeof(solve_explicit!), FDMContext, Vararg}
@zero_adjoint DefaultCtx Tuple{typeof(log_trace_info!), OptimizationState, Vararg}

# Manual rule for update_factorization!
function Mooncake.rrule!!(
    f::CoDual{typeof(update_factorization!)},
    ctx::CoDual
)
    res = update_factorization!(ctx.x)
    function update_factorization_pullback(Δ)
        return NoRData(), NoRData()
    end
    return CoDual(res, NoFData()), update_factorization_pullback
end

function Mooncake.rrule!!(
    f::CoDual{typeof(solve_explicit!)},
    ctx::CoDual,
    q::CoDual,
    Cn::CoDual,
    Cf::CoDual,
    Pn::CoDual,
    Nf::CoDual
)
    # Primal
    solve_explicit!(ctx.x, q.x, Cn.x, Cf.x, Pn.x, Nf.x)
    
    function solve_explicit_pullback(Δ)
        # Δ is the rdata of the return value (ctx.x.xyz_free), which is NoRData() for a mutable array.
        # The actual gradient should be in the shadow memory of ctx: ctx.dx.fields.xyz_free
        # solve adjoint system K * Λ = Δ
        ldiv!(ctx.x.adj_rhs, ctx.x.factorization, ctx.dx.fields.xyz_free)
        Λ = ctx.x.adj_rhs 
        
        # calculate gradients
        # xyz_free is (n_free x 3)
        # Cn is (n_edges x n_free)
        # Cf is (n_edges x n_fixed)
        # Nf is (n_fixed x 3)
        V_full = Cn.x * ctx.x.xyz_free + Cf.x * Nf.x # (n_edges x 3)
        V_adj = Cn.x * Λ # (n_edges x 3)
        
        # accumulate into q.dx
        if !(q.dx isa NoTangent)
            # q.dx is (n_edges,)
            # V_full and V_adj are (n_edges x 3)
            # we want -sum_j (V_full[i,j] * V_adj[i,j])
            for j in 1:size(V_full, 2)
                for i in 1:length(q.x)
                    q.dx[i] -= V_full[i, j] * V_adj[i, j]
                end
            end
        end
        
        # update Pn.dx
        if !(Pn.dx isa NoTangent)
            Pn.dx .+= Λ
        end
        
        # update Nf.dx
        if !(Nf.dx isa NoTangent)
            Nf.dx .-= Cf.x' * (Diagonal(q.x) * V_adj)
        end
        
        # Zero out consumed gradient in shadow memory
        ctx.dx.fields.xyz_free .= 0
        
        return NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), NoRData()
    end
    
    return CoDual(ctx.x.xyz_free, ctx.dx.fields.xyz_free), solve_explicit_pullback
end
