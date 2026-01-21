#######
# contains functions for analyzing an FDM network
#######

```
Explicit solver function
```
function solve_explicit(
    q::AbstractVector{<:Real}, #Vector of force densities
    Cn::SparseMatrixCSC{Int64,Int64}, #Index matrix of free nodes
    Cf::SparseMatrixCSC{Int64,Int64}, #Index matrix of fixed nodes
    Pn::AbstractMatrix{<:Real}, #Matrix of free node loads
    Nf::AbstractMatrix{<:Real}, #Matrix of fixed node positions
    )

    # Scale columns of Cn and Cf by q
    Cnq = Cn .* q
    Cfq = Cf .* q

    # Compute result using scaled matrices
    return (Cn' * Cnq) \ (Pn - Cn' * Cfq * Nf)
end

"""
High-performance in-place forward solver.
Updates cache.x based on current cache.q and variable_anchor_positions.
Uses LDLFactorization and applies conditional perturbations if singular.
"""
function solve_explicit!(cache::FDMCache, problem::OptimizationProblem, variable_anchor_positions::Matrix{Float64}; perturbation=1e-12)
    # 1. Update A.nzval in-place
    fill!(cache.A.nzval, 0.0)
    for k in 1:length(cache.q)
        qk = cache.q[k]
        for (nz_idx, coeff) in cache.q_to_nz[k]
            cache.A.nzval[nz_idx] += qk * coeff
        end
    end

    # 2. Prepare RHS: integrator.b = Pn - Cn' * diag(q) * Cf * Nf
    current_fixed_positions!(cache.Nf, problem, variable_anchor_positions)
    
    fixed_indices = problem.topology.fixed_node_indices
    Nf_fixed = @view cache.Nf[fixed_indices, :]
    mul!(cache.Cf_Nf, cache.Cf, Nf_fixed)
    
    for j in 1:3
        for i in 1:length(cache.q)
            cache.Q_Cf_Nf[i, j] = cache.q[i] * cache.Cf_Nf[i, j]
        end
    end
    
    rhs = cache.integrator.b
    copyto!(rhs, cache.Pn)
    mul!(rhs, cache.Cn', cache.Q_Cf_Nf, -1.0, 1.0)

    # 3. Solve A * x = RHS
    # Explicitly signal that A has changed to force re-factorization
    cache.integrator.A = cache.A

    max_retries = 1
    for retry in 0:max_retries
        sol = LinearSolve.solve!(cache.integrator)
        
        if sol.retcode == LinearSolve.ReturnCode.Success
            copyto!(cache.x, sol.u)
            return cache.x
        end
        
        if retry < max_retries
            @warn "Linear solve failed with retcode $(sol.retcode). Applying perturbation of $perturbation to diagonal."
            for i in 1:size(cache.A, 1)
                nz_idx = find_nz_index(cache.A, i, i)
                cache.A.nzval[nz_idx] += perturbation
            end
            # Signal change again
            cache.integrator.A = cache.A
        end
    end
    
    error("solve_explicit! failed after $max_retries retries.")
end

