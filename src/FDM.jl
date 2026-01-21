#######
# contains functions for analyzing an FDM network
#######

using TimerOutputs

"""
High-performance in-place forward solver.
Updates cache.x based on current q and variable_anchor_positions.
Uses LDLFactorization and applies conditional perturbations if singular.
"""
function solve_FDM!(cache::OptimizationCache, q::AbstractVector{<:Real}, problem::OptimizationProblem, variable_anchor_positions::Matrix{Float64}, perturbation::Float64=1e-12)
    @timeit cache.to "solve_FDM!" begin
        # 0. Sync q to cache
        @timeit cache.to "sync q" copyto!(cache.q, q)

        # 1. Update A.nzval in-place
        @timeit cache.to "update nzval" begin
            fill!(cache.A.nzval, 0.0)
            for k in 1:length(cache.q)
                qk = cache.q[k]
                for (nz_idx, coeff) in cache.q_to_nz[k]
                    cache.A.nzval[nz_idx] += qk * coeff
                end
            end
        end

        # 2. Prepare RHS: integrator.b = Pn - Cn' * diag(q) * Cf * Nf
        @timeit cache.to "prepare RHS" begin
            current_fixed_positions!(cache.Nf, problem, variable_anchor_positions)
            
            fixed_indices = problem.topology.fixed_node_indices
            # Copy to dense buffer to avoid slow sparse * view multiplication
            for j in 1:3
                for i in 1:length(fixed_indices)
                    cache.Nf_fixed[i, j] = cache.Nf[fixed_indices[i], j]
                end
            end
            mul!(cache.Cf_Nf, cache.Cf, cache.Nf_fixed)
            
            for j in 1:3
                for i in 1:length(cache.q)
                    cache.Q_Cf_Nf[i, j] = cache.q[i] * cache.Cf_Nf[i, j]
                end
            end
            
            copyto!(cache.grad_x, cache.Pn) 
            mul!(cache.grad_x, cache.Cn', cache.Q_Cf_Nf, -1.0, 1.0)
        end

        # 3. Solve A * x = RHS
        @timeit cache.to "linear solve" begin
            max_retries = 1
            for retry in 0:max_retries
                try
                    @timeit cache.to "factorize" LDLFactorizations.ldl_factorize!(cache.A, cache.factor)
                    @timeit cache.to "ldiv!" ldiv!(cache.x, cache.factor, cache.grad_x)
                    
                    # Update Nf buffer with free node positions for subsequent gradient calls
                    @timeit cache.to "update Nf" begin
                        free_indices = problem.topology.free_node_indices
                        for j in 1:3
                            for i in 1:length(free_indices)
                                cache.Nf[free_indices[i], j] = cache.x[i, j]
                            end
                        end
                    end
                    return nothing
                catch e
                    if retry < max_retries
                        @warn "Linear solve failed. Applying perturbation of $perturbation to diagonal."
                        for i in 1:size(cache.A, 1)
                            nz_idx = find_nz_index(cache.A, i, i)
                            cache.A.nzval[nz_idx] += perturbation
                        end
                    else
                        rethrow(e)
                    end
                end
            end
        end
    end
    
    error("solve_FDM! failed after 1 retries.")
end

"""
    solve_FDM!(cache::OptimizationCache, problem::OptimizationProblem, variable_anchor_positions::Matrix{Float64})

Convenience overload that uses the `q` already stored in `cache`.
"""
function solve_FDM!(cache::OptimizationCache, problem::OptimizationProblem, variable_anchor_positions::Matrix{Float64})
    solve_FDM!(cache, cache.q, problem, variable_anchor_positions)
end

