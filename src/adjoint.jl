module Adjoints

using ..Theseus
using LinearAlgebra
using SparseArrays
using LinearSolve
using Mooncake
using Mooncake: NoTangent

"""
    solve_adjoint!(cache::Theseus.FDMCache, dJ_dx::AbstractMatrix)

Solves the adjoint system A^T λ = dJ_dx using the pre-factorized matrix from the forward pass.
Since A is symmetric (C'QC), we reuse the same integrator and factorization.
"""
function solve_adjoint!(cache::Theseus.FDMCache, dJ_dx::AbstractMatrix)
    # Seed the RHS of the linear solver with the objective gradient
    copyto!(cache.integrator.b, dJ_dx)
    
    # We DO NOT update integrator.A here because we want to reuse the 
    # factors already calculated in solve_explicit!. 
    # LinearSolve.solve! will skip factorization if integrator.A hasn't changed.
    LinearSolve.solve!(cache.integrator)
    
    # integrator.u now contains λ. Copy it to our buffer.
    copyto!(cache.λ, cache.integrator.u)
    return nothing
end

"""
    accumulate_gradients!(cache::Theseus.FDMCache, p::Theseus.OptimizationProblem)

Computes the gradient with respect to q and Nf (fixed node positions).
∂J/∂q_k = -Δλ_k ⋅ ΔN_k
∂J/∂N_v += -q_k * Δλ (if v is fixed)
"""
function accumulate_gradients!(cache::Theseus.FDMCache, p::Theseus.OptimizationProblem)
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
            
            # Nf contains ALL node positions (updated in solve_explicit!)
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
    return nothing
end

# Mooncake Rule for solve_explicit!
function Mooncake.rrule!!(
    ::Any,
    ::typeof(Theseus.solve_explicit!),
    cache::Theseus.FDMCache,
    problem::Theseus.OptimizationProblem,
    variable_anchor_positions::Matrix{Float64}
)
    # Primal
    x = Theseus.solve_explicit!(cache, problem, variable_anchor_positions)

    # Pullback
    function solve_explicit_pullback(dx)
        # 1. Backsolve to get lambda
        solve_adjoint!(cache, dx)
        
        # 2. Accumulate gradients for q and Nf
        accumulate_gradients!(cache, problem)
        
        # 3. Return tangents
        # Since cache and problem are mutable and we've updated their internal 
        # gradient buffers (grad_q, grad_Nf), we return NoTangent for the 
        # objects themselves in the Mooncake sense for now.
        return (NoTangent(), NoTangent(), NoTangent(), NoTangent())
    end

    return x, solve_explicit_pullback
end

end # module
  