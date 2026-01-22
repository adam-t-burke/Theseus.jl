using SparseArrays
using LinearAlgebra
using CUDA
using LinearSolve
using LDLFactorizations
using SparseDiffTools
using Graphs

"""
    find_nz_index(A::SparseMatrixCSC, row::Int, col::Int)

Find the index in `A.nzval` corresponding to the entry `(row, col)`.
"""
function find_nz_index(A::SparseMatrixCSC, row::Int, col::Int)
    for idx in A.colptr[col]:(A.colptr[col+1]-1)
        if A.rowval[idx] == row
            return idx
        end
    end
    error("Index ($row, $col) not found in sparse matrix")
end

"""
    compute_q_to_nz(Cn::SparseMatrixCSC, A::SparseMatrixCSC)

Identify which indices in `A.nzval` are updated by each force density `q_i`.
Each force density `q_i` for edge `i` (connecting nodes `u` and `v`) contributes to:
- `A[u, u]` and `A[v, v]` (if `u`, `v` are free)
- `A[u, v]` and `A[v, u]` (if both `u` and `v` are free)
"""
function compute_q_to_nz(Cn::SparseMatrixCSC, A::SparseMatrixCSC)
    num_edges = size(Cn, 1)
    q_to_nz = [Int32[] for _ in 1:num_edges]
    
    # iterate over each edge
    for i in 1:num_edges
        nz_indices = Cn[i, :].nzind
        # Edge i connects free nodes nodes[nz_indices]
        for (idx_u, u) in enumerate(nz_indices)
            push!(q_to_nz[i], Int32(find_nz_index(A, u, u)))
            for v in nz_indices[idx_u+1:end]
                push!(q_to_nz[i], Int32(find_nz_index(A, u, v)))
                push!(q_to_nz[i], Int32(find_nz_index(A, v, u)))
            end
        end
    end
    return q_to_nz
end

"""
    compute_edge_coloring(incidence::SparseMatrixCSC)

Perform edge coloring to ensure no two edges in a color group share a node.
Used for race-free parallel writes to nodal buffers.
"""
function compute_edge_coloring(incidence::SparseMatrixCSC)
    num_edges, num_nodes = size(incidence)
    adj = incidence * incidence' # Edge adjacency via shared nodes (off-diagonals)
    # diagonal elements are 2 (since each edge has 2 nodes), we need to clear them for SparseDiffTools
    for i in 1:num_edges
        adj[i, i] = 0
    end
    dropzeros!(adj)
    colors = matrix_colors(adj)
    
    num_colors = maximum(colors)
    color_groups = [Int32[] for _ in 1:num_colors]
    for i in 1:num_edges
        push!(color_groups[colors[i]], Int32(i))
    end
    return [CuArray(group) for group in color_groups]
end

"""
    initialize_fdm_cache(problem::OptimizationProblem)

Pre-allocate all CPU and GPU buffers for zero-allocation FDM optimization.
"""
function initialize_fdm_cache(problem::OptimizationProblem)
    topo = problem.topology
    Cn = topo.free_incidence
    Cf = topo.fixed_incidence
    
    # 1. CPU Linear Algebra setup
    # Initial A = Cn' * Cn (assuming q=1)
    A = Cn' * Cn
    # Add a small diagonal shift for stability as per spec
    for i in 1:size(A, 1)
        A[i, i] += 1e-9
    end
    
    # Map q_i to A.nzval indices
    q_to_nz = compute_q_to_nz(Cn, A)
    diag_nz_indices = [Int32(find_nz_index(A, i, i)) for i in 1:size(A, 1)]
    
    # Initialize LinearSolve integrator
    b = zeros(Float64, size(Cn, 2), 3)
    temp_M3 = zeros(Float64, topo.num_edges, 3)
    q_cpu = zeros(Float64, topo.num_edges)
    prob = LinearProblem(A, b)
    integrator = init(prob, LDLFactorizations.LDLFactorization())
    
    # 2. GPU Pre-allocation
    M = topo.num_edges
    N = topo.num_nodes
    
    x_gpu = CUDA.zeros(Float64, N, 3)
    λ_gpu = CUDA.zeros(Float64, N, 3)
    L_gpu = CUDA.zeros(Float64, M, 1)
    F_gpu = CUDA.zeros(Float64, M, 1)
    q_gpu = CUDA.zeros(Float64, M, 1)
    dq_gpu = CUDA.zeros(Float64, M, 1)
    dL_gpu = CUDA.zeros(Float64, M, 1)
    dF_gpu = CUDA.zeros(Float64, M, 1)
    λ_free_gpu = CUDA.zeros(Float64, size(Cn, 2), 3)
    reactions_gpu = CUDA.zeros(Float64, size(Cf, 2), 3)
    
    # Topology on GPU
    # edge_nodes as (M, 2)
    edge_nodes_cpu = zeros(Int32, M, 2)
    rows = rowvals(topo.incidence)
    for j in 1:N
        for idx in nzrange(topo.incidence, j)
            i = rows[idx]
            if edge_nodes_cpu[i, 1] == 0 # uninitialized
                edge_nodes_cpu[i, 1] = Int32(j)
            else
                edge_nodes_cpu[i, 2] = Int32(j)
            end
        end
    end
    edge_nodes_gpu = CuArray(edge_nodes_cpu)
    
    # Coloring
    color_groups = compute_edge_coloring(topo.incidence)
    
    # Anderson buffers
    x_prev_gpu = CUDA.zeros(Float64, N, 3)
    g_prev_gpu = CUDA.zeros(Float64, N, 3)
    
    # Consensus Buffers (ADMM)
    z_consensus = CUDA.zeros(Float64, M, 3)
    y_dual = CUDA.zeros(Float64, M, 3)
    l_min = CUDA.zeros(Float64, M, 1)
    l_max = CUDA.zeros(Float64, M, 1)
    f_target = CUDA.zeros(Float64, M, 1)
    
    # Node indices on GPU
    free_node_indices_gpu = CuArray{Int32}(topo.free_node_indices)
    fixed_node_indices_gpu = CuArray{Int32}(topo.fixed_node_indices)
    
    fixed_node_to_fixed_idx_cpu = zeros(Int32, N)
    for (i, node_idx) in enumerate(topo.fixed_node_indices)
        fixed_node_to_fixed_idx_cpu[node_idx] = Int32(i)
    end
    fixed_node_to_fixed_idx_gpu = CuArray(fixed_node_to_fixed_idx_cpu)
    
    return FDMCache(
        A, integrator, q_to_nz, diag_nz_indices, b, temp_M3, q_cpu,
        x_gpu, λ_gpu, L_gpu, F_gpu, q_gpu, dq_gpu, dL_gpu, dF_gpu, λ_free_gpu, reactions_gpu,
        edge_nodes_gpu, free_node_indices_gpu, fixed_node_indices_gpu, fixed_node_to_fixed_idx_gpu, color_groups,
        x_prev_gpu, g_prev_gpu, z_consensus, y_dual,
        l_min, l_max, f_target,
        Dict{AbstractObjective, CuArray}()
    )
end

"""
    solve_fdm!(problem::OptimizationProblem, q::AbstractVector{Float64}, variable_anchor_positions::AbstractMatrix{Float64}, cache::FDMCache)

In-place FDM solver reusing CPU factorization and syncing result to GPU.
"""
function solve_fdm!(problem::OptimizationProblem, q::AbstractVector{Float64}, variable_anchor_positions::AbstractMatrix{Float64}, cache::FDMCache)
    # 1. Update A in-place
    cache.A.nzval .= 0.0
    for i in 1:length(q)
        val = q[i]
        for nz_idx in cache.q_to_nz[i]
            cache.A.nzval[nz_idx] += val
        end
    end
    # Add diagonal shift
    for nz_idx in cache.diag_nz_indices
        cache.A.nzval[nz_idx] += 1e-9
    end

    # 2. Update b = Pn - Cn' * Cfq * Nf
    topo = problem.topology
    fixed_pos = current_fixed_positions(problem, variable_anchor_positions)
    
    # Calculate Cf * Nf -> temp_M3 (M, 3)
    cache.temp_M3 .= 0.0
    Cf = topo.fixed_incidence
    rows_cf = rowvals(Cf)
    for j in 1:size(Cf, 2) # fixed nodes
        pos_j1 = fixed_pos[j, 1]
        pos_j2 = fixed_pos[j, 2]
        pos_j3 = fixed_pos[j, 3]
        for idx in nzrange(Cf, j)
            i = rows_cf[idx]
            val = Cf.nzval[idx]
            cache.temp_M3[i, 1] += val * pos_j1
            cache.temp_M3[i, 2] += val * pos_j2
            cache.temp_M3[i, 3] += val * pos_j3
        end
    end
    
    # Apply q: temp_M3 = q .* (Cf * Nf)
    for i in 1:topo.num_edges
        q_i = q[i]
        cache.temp_M3[i, 1] *= q_i
        cache.temp_M3[i, 2] *= q_i
        cache.temp_M3[i, 3] *= q_i
    end
    
    # b = Pn - Cn' * temp_M3
    Cn = topo.free_incidence
    cache.b .= problem.loads.free_node_loads
    rows_cn = rowvals(Cn)
    for j in 1:size(Cn, 2) # free nodes
        for idx in nzrange(Cn, j)
            i = rows_cn[idx]
            val = Cn.nzval[idx]
            cache.b[j, 1] -= val * cache.temp_M3[i, 1]
            cache.b[j, 2] -= val * cache.temp_M3[i, 2]
            cache.b[j, 3] -= val * cache.temp_M3[i, 3]
        end
    end

    # 3. Solve Ax = b on CPU
    solve!(cache.integrator)
    
    # 4. Sync Result to GPU
    # a) Update fixed nodes in x_gpu (variable anchors may have moved)
    fixed_pos_gpu = CuArray(fixed_pos)
    threads = 256 # multiple of 32 (warp size), good balance of occupancy and resource usage
    blocks_fixed = ceil(Int, length(topo.fixed_node_indices) / threads)
    @cuda threads=threads blocks=blocks_fixed kernel_update_fixed_nodes!(
        cache.x_gpu, fixed_pos_gpu, cache.fixed_node_indices_gpu
    )
    
    # b) Scatter solved free nodes to x_gpu
    # cache.integrator.u is (N_free, 3) on CPU
    x_free_gpu = CuArray(cache.integrator.u)
    blocks_free = ceil(Int, length(topo.free_node_indices) / threads)
    @cuda threads=threads blocks=blocks_free kernel_scatter_x!(
        cache.x_gpu, x_free_gpu, cache.free_node_indices_gpu
    )
    
    return cache.integrator.u
end

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

