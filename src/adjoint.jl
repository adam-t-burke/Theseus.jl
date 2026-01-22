module Adjoints

using ..Theseus
using LinearAlgebra
using SparseArrays
using LinearSolve
using Mooncake
using TimerOutputs
using Mooncake: NoTangent, @is_primitive, ReverseMode, DefaultCtx, CoDual, NoRData, NoFData

# Make TimerOutput invisible to Mooncake to avoid recursive tangent type search
Mooncake.tangent_type(::Type{TimerOutputs.TimerOutput}) = NoTangent

# Register solve_FDM! as a primitive for Mooncake
@is_primitive DefaultCtx ReverseMode Tuple{typeof(Theseus.solve_FDM!), Vararg{Any}}

"""
    solve_adjoint!(cache::Theseus.OptimizationCache, dJ_dx::AbstractMatrix)

Solves the adjoint system A^T λ = dJ_dx using the pre-factorized matrix from the forward pass.
Since A is symmetric (C'QC), we reuse the same factorization.
"""
function solve_adjoint!(cache::Theseus.OptimizationCache, dJ_dx::AbstractMatrix)
    @timeit cache.to "solve_adjoint!" begin
        # Solve A * lambda = dJ_dx
        # We DO NOT update cache.factor here because we want to reuse the 
        # factors already calculated in solve_FDM!. 
        ldiv!(cache.λ, cache.factor, dJ_dx)
    end
    return nothing
end

"""
    accumulate_gradients!(cache::Theseus.OptimizationCache, p::Theseus.OptimizationProblem)

Computes the gradient with respect to q and Nf (fixed node positions).
∂J/∂q_k = -Δλ_k ⋅ ΔN_k
∂J/∂N_v += -q_k * Δλ (if v is fixed)
"""
function accumulate_gradients!(cache::Theseus.OptimizationCache, p::Theseus.OptimizationProblem)
    @timeit cache.to "accumulate_gradients!" begin
        fill!(cache.grad_q, 0.0)
        fill!(cache.grad_Nf, 0.0)
        
        for k in 1:p.topology.num_edges
            u = cache.edge_starts[k]
            v = cache.edge_ends[k]
            
            u_free = cache.node_to_free_idx[u]
            v_free = cache.node_to_free_idx[v]
            
            for d in 1:3
                λ_u = u_free > 0 ? cache.λ[u_free, d] : 0.0
                λ_v = v_free > 0 ? cache.λ[v_free, d] : 0.0
                dλ = λ_v - λ_u
                
                # Nf contains ALL node positions (updated in solve_FDM!)
                dN = cache.Nf[v, d] - cache.Nf[u, d]
                
                # ∂J/∂q_k
                cache.grad_q[k] -= dλ * dN
                
                # ∂J/∂N_f
                # Total derivative: J = f(x(q, Nf), q, Nf)
                # We already handled dJ/dx in solve_adjoint!.
                # Explicit dependence:
                # -q_k * dλ is the contribution of edge k to the force at nodes.
                term = -cache.q[k] * dλ
                if v_free == 0
                    cache.grad_Nf[v, d] += term
                end
                if u_free == 0
                    cache.grad_Nf[u, d] -= term
                end
            end
        end
    end
    return nothing
end

# Mooncake Rule for solve_FDM!
# We provide 4 and 5 argument versions to match solve_FDM! signatures.

function _solve_FDM_pullback_common(cache_cd, q_cd, problem_cd, anchors_cd)
    topo = problem_cd.x.topology
    cache = cache_cd.x
    # 1. Map gradients from Nf back to x (for free nodes)
    @timeit cache.to "pullback map Nf->x" begin
        for j in 1:3
            for (i, idx) in enumerate(topo.free_node_indices)
                cache_cd.dx.fields.x[i, j] += cache_cd.dx.fields.Nf[idx, j]
            end
        end
    end

    # 2. Adjoint solve
    solve_adjoint!(cache, cache_cd.dx.fields.x)
    
    # 3. Accumulate: dJ/dq and dJ/dNf (indirect)
    accumulate_gradients!(cache, problem_cd.x)
    
    # 4. Update argument shadows
    @timeit cache.to "pullback update shadows" begin
        q_cd.dx .+= cache.grad_q
        
        var_indices = problem_cd.x.anchors.variable_indices
        for i in 1:length(var_indices)
            idx = var_indices[i]
            # Direct effect from objective on Nf
            anchors_cd.dx[i, 1] += cache_cd.dx.fields.Nf[idx, 1]
            anchors_cd.dx[i, 2] += cache_cd.dx.fields.Nf[idx, 2]
            anchors_cd.dx[i, 3] += cache_cd.dx.fields.Nf[idx, 3]
            
            # Indirect effect from x -> Nf
            anchors_cd.dx[i, 1] += cache.grad_Nf[idx, 1]
            anchors_cd.dx[i, 2] += cache.grad_Nf[idx, 2]
            anchors_cd.dx[i, 3] += cache.grad_Nf[idx, 3]
        end
    end

    # CRITICAL: Clear tangent buffers in cache shadow to prevent accumulation across iterations
    # We clear all major mutable buffers that are updated in the forward pass
    @timeit cache.to "pullback zero fields" begin
        fill!(cache_cd.dx.fields.x, 0.0)
        fill!(cache_cd.dx.fields.Nf, 0.0)
        fill!(cache_cd.dx.fields.q, 0.0)
        fill!(cache_cd.dx.fields.member_lengths, 0.0)
        fill!(cache_cd.dx.fields.member_forces, 0.0)
        fill!(cache_cd.dx.fields.reactions, 0.0)
    end
    return nothing
end

function Mooncake.rrule!!(
    f_cd::CoDual{typeof(Theseus.solve_FDM!)},
    cache_cd::CoDual{<:Theseus.OptimizationCache},
    q_cd::CoDual{<:AbstractVector{<:Real}},
    problem_cd::CoDual{<:Theseus.OptimizationProblem},
    anchors_cd::CoDual{<:Matrix{Float64}},
    perturbation_cd::CoDual{Float64}
)
    Theseus.solve_FDM!(cache_cd.x, q_cd.x, problem_cd.x, anchors_cd.x, perturbation_cd.x)
    
    out_cd = CoDual(nothing, NoFData())

    function solve_FDM_pullback_ext(dy::NoRData)
        _solve_FDM_pullback_common(cache_cd, q_cd, problem_cd, anchors_cd)
        return NoRData(), NoRData(), NoRData(), Mooncake.rdata(Mooncake.zero_tangent(problem_cd.x)), NoRData(), 0.0
    end

    return out_cd, solve_FDM_pullback_ext
end

function Mooncake.rrule!!(
    f_cd::CoDual{typeof(Theseus.solve_FDM!)},
    cache_cd::CoDual{<:Theseus.OptimizationCache},
    q_cd::CoDual{<:AbstractVector{<:Real}},
    problem_cd::CoDual{<:Theseus.OptimizationProblem},
    anchors_cd::CoDual{<:Matrix{Float64}}
)
    Theseus.solve_FDM!(cache_cd.x, q_cd.x, problem_cd.x, anchors_cd.x)
    
    out_cd = CoDual(nothing, NoFData())

    function solve_FDM_pullback(dy::NoRData)
        _solve_FDM_pullback_common(cache_cd, q_cd, problem_cd, anchors_cd)
        return NoRData(), NoRData(), NoRData(), Mooncake.rdata(Mooncake.zero_tangent(problem_cd.x)), NoRData()
    end

    return out_cd, solve_FDM_pullback
end

end # module
  
