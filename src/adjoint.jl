using Mooncake
using LinearSolve
using CUDA
using ChainRulesCore

import Mooncake: CoDual, rrule!!, NoRData, NoFData

"""
    Mooncake.rrule!!(::CoDual{typeof(solve_fdm!)}, cache_dual, q_dual, problem_dual)

Custom Mooncake rule for the FDM solver. 
Exploits symmetry and reuses CPU factorization for the adjoint solve.
"""
function Mooncake.rrule!!(
    ::CoDual{typeof(solve_fdm!)},
    problem_dual::CoDual{<:OptimizationProblem},
    q_dual::CoDual{<:AbstractVector},
    anchors_dual::CoDual{<:AbstractMatrix},
    cache_dual::CoDual{<:FDMCache}
)
    problem = problem_dual.x
    q = q_dual.x
    anchors = anchors_dual.x
    cache = cache_dual.x
    
    # 1. Forward Pass
    x_free = solve_fdm!(problem, q, anchors, cache)
    
    function solve_fdm_pullback!!(dx_free_rdata)
        # dx_free_rdata is (N_f, 3) nodal sensitiveities.
        # We assume problem and cache don't need rdata updates for now 
        # (they are largely static or mutated in ways we handle manually).
        
        # 1. Adjoint solve on CPU: A λ = dx_free
        # A is symmetric, so we reuse the forward factorization.
        # integrator.b is used for the RHS.
        cache.integrator.b .= dx_free_rdata
        solve!(cache.integrator)
        lambda_free = cache.integrator.u # (N_f, 3)
        
        # 2. Prepare GPU sensitivity calculation
        # a. Transfer nodal sensitivities (lambda) to GPU λ_gpu
        copyto!(cache.λ_free_gpu, lambda_free)
        CUDA.fill!(cache.λ_gpu, 0.0)
        
        n_free = length(cache.free_node_indices_gpu)
        threads = 256 # multiple of 32 (warp size), good balance of occupancy and resource usage
        blocks = ceil(Int, n_free / threads)
        @cuda threads=threads blocks=blocks kernel_scatter_lambda!(cache.λ_gpu, cache.λ_free_gpu, cache.free_node_indices_gpu)
        
        # x_gpu is already synced by solve_fdm!
        
        # 3. Compute Edge Gradient dq on GPU
        n_edges = size(cache.edge_nodes, 1)
        blocks_m = ceil(Int, n_edges / threads)
        @cuda threads=threads blocks=blocks_m kernel_compute_edge_gradients!(cache.x_gpu, cache.λ_gpu, cache.edge_nodes, cache.dq_gpu)
        
        # 4. Download dq to CPU
        dq_rdata = vec(Array(cache.dq_gpu))
        da_rdata = zero(anchors)
        
        # Return RData for (typeof(solve_fdm!), problem, q, anchors, cache)
        return NoRData(), NoRData(), dq_rdata, da_rdata, NoRData()
    end
    
    return CoDual(x_free, NoFData()), solve_fdm_pullback!!
end
