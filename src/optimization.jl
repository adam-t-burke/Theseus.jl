using LinearAlgebra
using Optim
using Mooncake
using ChainRulesCore: ignore_derivatives
using Logging
using CUDA

"""
    evaluate_geometry!(cache::FDMCache, q::AbstractVector{Float64}, variable_anchors::AbstractMatrix{Float64}, problem::OptimizationProblem)

High-performance, zero-allocation geometry evaluation using GPU kernels.
"""
function evaluate_geometry!(cache::FDMCache, q::AbstractVector{Float64}, variable_anchors::AbstractMatrix{Float64}, problem::OptimizationProblem)
    # 1. Update force densities on GPU
    copyto!(cache.q_gpu, q)

    # 2. Solve FDM (Updates x_gpu internally)
    solve_fdm!(problem, q, variable_anchors, cache)
    
    # 3. Compute L and F on GPU
    n_edges = size(cache.edge_nodes, 1)
    threads = 256 # multiple of 32 (warp size), good balance of occupancy and resource usage
    blocks = ceil(Int, n_edges / threads)
    @cuda threads=threads blocks=blocks kernel_compute_geometry!(cache.x_gpu, cache.edge_nodes, cache.q_gpu, cache.L_gpu, cache.F_gpu)
    
    return x_free
end

function evaluate_geometry(problem::OptimizationProblem, q::AbstractVector{<:Real}, anchor_positions::AbstractMatrix{<:Real})
    fixed_positions = current_fixed_positions(problem, anchor_positions)
    xyz_free = solve_explicit(q, problem.topology.free_incidence, problem.topology.fixed_incidence, problem.loads.free_node_loads, fixed_positions)
    xyz_full = vcat(xyz_free, fixed_positions)
    member_vectors = problem.topology.incidence * xyz_full
    member_lengths = map(norm, eachrow(member_vectors))
    member_forces = q .* member_lengths
    reactions = anchor_reactions(problem.topology, q, xyz_full)
    GeometrySnapshot(xyz_free, fixed_positions, xyz_full, member_lengths, member_forces, reactions)
end

function objective_loss(obj::TargetXYZObjective, snapshot::GeometrySnapshot)
    obj.weight * target_xyz(snapshot.xyz_full, obj.target, obj.node_indices)
end

function objective_loss(obj::TargetXYObjective, snapshot::GeometrySnapshot)
    obj.weight * target_xy(snapshot.xyz_full, obj.target, obj.node_indices)
end

function objective_loss(obj::TargetLengthObjective, snapshot::GeometrySnapshot)
    obj.weight * lenTarget(snapshot.member_lengths, obj.target, obj.edge_indices)
end

function objective_loss(obj::LengthVariationObjective, snapshot::GeometrySnapshot)
    obj.weight * lenVar(snapshot.member_lengths, obj.edge_indices)
end

function objective_loss(obj::ForceVariationObjective, snapshot::GeometrySnapshot)
    obj.weight * forceVar(snapshot.member_forces, obj.edge_indices)
end

function objective_loss(obj::SumForceLengthObjective, snapshot::GeometrySnapshot)
    edges = obj.edge_indices
    obj.weight * sum(snapshot.member_lengths[edges] .* snapshot.member_forces[edges])
end

function objective_loss(obj::MinLengthObjective, snapshot::GeometrySnapshot)
    obj.weight * minPenalty(snapshot.member_lengths, obj.threshold, obj.edge_indices, obj.sharpness)
end

function objective_loss(obj::MaxLengthObjective, snapshot::GeometrySnapshot)
    obj.weight * maxPenalty(snapshot.member_lengths, obj.threshold, obj.edge_indices, obj.sharpness)
end

function objective_loss(obj::MinForceObjective, snapshot::GeometrySnapshot)
    obj.weight * minPenalty(snapshot.member_forces, obj.threshold, obj.edge_indices, obj.sharpness)
end

function objective_loss(obj::MaxForceObjective, snapshot::GeometrySnapshot)
    obj.weight * maxPenalty(snapshot.member_forces, obj.threshold, obj.edge_indices, obj.sharpness)
end

function objective_loss(obj::RigidSetCompareObjective, snapshot::GeometrySnapshot)
    obj.weight * rigidSetCompare(snapshot.xyz_full, obj.node_indices, obj.target)
end

function objective_loss(obj::ReactionDirectionObjective, snapshot::GeometrySnapshot)
    obj.weight * reaction_direction_loss(snapshot.reactions, obj)
end

function objective_loss(obj::ReactionDirectionMagnitudeObjective, snapshot::GeometrySnapshot)
    obj.weight * reaction_direction_magnitude_loss(snapshot.reactions, obj)
end

objective_loss(::AbstractObjective, snapshot::GeometrySnapshot) = zero(eltype(snapshot.member_lengths))

function total_loss(problem::OptimizationProblem, q::AbstractVector{<:Real}, anchor_positions::AbstractMatrix{<:Real}, snapshot::GeometrySnapshot)
    loss = zero(eltype(snapshot.member_lengths))
    for obj in problem.parameters.objectives
        loss += objective_loss(obj, snapshot)
    end
    loss
end

function total_loss_gpu(problem::OptimizationProblem, cache::FDMCache)
    loss = 0.0
    for obj in problem.parameters.objectives
        loss += objective_loss_gpu(obj, cache)
    end
    loss
end

# Default fallback
objective_loss_gpu(obj::AbstractObjective, cache::FDMCache) = 0.0

function get_obj_gpu_data(obj::AbstractObjective, cache::FDMCache, field_name::Symbol)
    # Check if we already have it
    key = (obj, field_name)
    if !haskey(cache.objective_targets, key)
        # Upload data to GPU
        data = getfield(obj, field_name)
        cache.objective_targets[key] = CuArray(data)
    end
    return cache.objective_targets[key]
end

function objective_loss_gpu(obj::SumForceLengthObjective, cache::FDMCache)
    @view(cache.L_gpu[obj.edge_indices])' * @view(cache.F_gpu[obj.edge_indices]) * obj.weight
end

function objective_loss_gpu(obj::TargetXYZObjective, cache::FDMCache)
    target = get_obj_gpu_data(obj, cache, :target)
    diff = @view(cache.x_gpu[obj.node_indices, :]) .- target
    obj.weight * CUDA.sum(diff .^ 2)
end

function objective_loss_gpu(obj::TargetXYObjective, cache::FDMCache)
    target = get_obj_gpu_data(obj, cache, :target)
    # Only compare first 2 columns (x and y)
    diff = @view(cache.x_gpu[obj.node_indices, 1:2]) .- @view(target[:, 1:2])
    obj.weight * CUDA.sum(diff .^ 2)
end

function objective_loss_gpu(obj::TargetLengthObjective, cache::FDMCache)
    target = get_obj_gpu_data(obj, cache, :target)
    diff = @view(cache.L_gpu[obj.edge_indices]) .- target
    obj.weight * CUDA.sum(diff .^ 2)
end

function objective_loss_gpu(obj::MinLengthObjective, cache::FDMCache)
    threshold = get_obj_gpu_data(obj, cache, :threshold)
    vals = @view(cache.L_gpu[obj.edge_indices])
    # z = k * (threshold - vals) - 1
    obj.weight * CUDA.sum(log1p.(exp.(obj.sharpness .* (threshold .- vals) .- 1.0)))
end

function objective_loss_gpu(obj::MaxLengthObjective, cache::FDMCache)
    threshold = get_obj_gpu_data(obj, cache, :threshold)
    vals = @view(cache.L_gpu[obj.edge_indices])
    # z = sharpness * (vals - threshold) - 1
    obj.weight * CUDA.sum(log1p.(exp.(obj.sharpness .* (vals .- threshold) .- 1.0)))
end

function objective_loss_gpu(obj::MinForceObjective, cache::FDMCache)
    threshold = get_obj_gpu_data(obj, cache, :threshold)
    vals = @view(cache.F_gpu[obj.edge_indices])
    obj.weight * CUDA.sum(log1p.(exp.(obj.sharpness .* (threshold .- vals) .- 1.0)))
end

function objective_loss_gpu(obj::MaxForceObjective, cache::FDMCache)
    threshold = get_obj_gpu_data(obj, cache, :threshold)
    vals = @view(cache.F_gpu[obj.edge_indices])
    obj.weight * CUDA.sum(log1p.(exp.(obj.sharpness .* (vals .- threshold) .- 1.0)))
end

function objective_loss_gpu(obj::LengthVariationObjective, cache::FDMCache)
    vals = @view(cache.L_gpu[obj.edge_indices])
    obj.weight * (maximum(vals) - minimum(vals))
end

function objective_loss_gpu(obj::ForceVariationObjective, cache::FDMCache)
    vals = @view(cache.F_gpu[obj.edge_indices])
    obj.weight * (maximum(vals) - minimum(vals))
end

function objective_loss_gpu(obj::RigidSetCompareObjective, cache::FDMCache)
    xyz = @view(cache.x_gpu[obj.node_indices, :])
    target = get_obj_gpu_data(obj, cache, :target)
    
    # Calculate pair distances on GPU
    # This matches the CPU rigidSetCompare logic
    n = size(xyz, 1)
    
    # For small n, we can do this with broadcasting
    # xyz is (n, 3). We want (n, n) matrix of distances.
    # dist[i, j] = norm(xyz[i, :] - xyz[j, :])
    
    # Broadcast subtraction: (n, 1, 3) - (1, n, 3) -> (n, n, 3)
    diffs = reshape(xyz, n, 1, 3) .- reshape(xyz, 1, n, 3)
    test_dists = sqrt.(sum(diffs.^2, dims=3))
    
    # Same for target
    target_diffs = reshape(target, n, 1, 3) .- reshape(target, 1, n, 3)
    target_dists = sqrt.(sum(target_diffs.^2, dims=3))
    
    obj.weight * CUDA.sum((target_dists .- test_dists).^2)
end

function objective_loss_gpu(obj::ReactionDirectionObjective, cache::FDMCache)
    # Pre-map anchor indices to fixed indices if not done
    key = (obj, :mapped_fixed_indices)
    if !haskey(cache.objective_targets, key)
        node_to_fixed = Vector(cache.fixed_node_to_fixed_idx_gpu)
        fixed_indices = [Int(node_to_fixed[i]) for i in obj.anchor_indices]
        cache.objective_targets[key] = CuArray(fixed_indices)
    end
    fixed_indices = cache.objective_targets[key]
    
    reactions = @view(cache.reactions_gpu[fixed_indices, :])
    targets = get_obj_gpu_data(obj, cache, :target_directions)
    
    # R_norm = sqrt(Rx^2 + Ry^2 + Rz^2)
    # misalignment = 1 - (R ⋅ target) / R_norm
    r_norms = sqrt.(sum(reactions.^2, dims=2))
    # Avoid div by zero
    dots = sum(reactions .* targets, dims=2)
    misalignments = 1.0 .- (dots ./ max.(r_norms, 1e-12))
    
    # misalignment is 1.0 if r_norm is 0
    # handled by max.(r_norms, 1e-12) -> misalignment approx 1 - 0 = 1
    
    obj.weight * CUDA.sum(misalignments)
end

function objective_loss_gpu(obj::ReactionDirectionMagnitudeObjective, cache::FDMCache)
    key = (obj, :mapped_fixed_indices)
    if !haskey(cache.objective_targets, key)
        node_to_fixed = Vector(cache.fixed_node_to_fixed_idx_gpu)
        fixed_indices = [Int(node_to_fixed[i]) for i in obj.anchor_indices]
        cache.objective_targets[key] = CuArray(fixed_indices)
    end
    fixed_indices = cache.objective_targets[key]
    
    reactions = @view(cache.reactions_gpu[fixed_indices, :])
    targets = get_obj_gpu_data(obj, cache, :target_directions)
    target_mags = get_obj_gpu_data(obj, cache, :target_magnitudes)
    
    r_norms = sqrt.(sum(reactions.^2, dims=2))
    dots = sum(reactions .* targets, dims=2)
    dir_loss = 1.0 .- (dots ./ max.(r_norms, 1e-12))
    
    mag_loss = max.(r_norms .- target_mags, 0.0)
    
    obj.weight * CUDA.sum(dir_loss .+ mag_loss)
end

function pack_parameters(problem::OptimizationProblem, state::OptimizationState)
    if isempty(problem.anchors.variable_indices)
        return copy(state.force_densities)
    end
    vcat(state.force_densities, vec(state.variable_anchor_positions))
end

function unpack_parameters(problem::OptimizationProblem, θ::AbstractVector{T}) where {T<:Real}
    ne = problem.topology.num_edges
    q = copy(@view θ[1:ne])
    nvar = length(problem.anchors.variable_indices)
    if nvar == 0
        return q, zeros(T, 0, 3)
    end
    anchors = reshape(copy(@view θ[ne + 1:end]), 3, nvar)'
    q, anchors
end

function form_finding_objective(problem::OptimizationProblem, trace_state::OptimizationState, lb, ub, lb_idx, ub_idx, geo_scale, barrier_weight, sharpness)
    empty!(trace_state.loss_trace)
    empty!(trace_state.penalty_trace)
    empty!(trace_state.node_trace)
    trace_state.iterations = 0

    function objective(θ)
        q, anchors = unpack_parameters(problem, θ)
        snapshot = evaluate_geometry(problem, q, anchors)
        
        barrier_loss = pBounds(θ, lb, ub, lb_idx, ub_idx, sharpness, sharpness)
        
        loss = (geometric_loss * geo_scale) + (barrier_loss * barrier_weight)
        
        if !isderiving()
            ignore_derivatives() do
                trace_state.force_densities = copy(q)
                trace_state.variable_anchor_positions = copy(anchors)
                push!(trace_state.loss_trace, loss)
                push!(trace_state.penalty_trace, barrier_loss * barrier_weight)
                if problem.parameters.tracing.record_nodes
                    push!(trace_state.node_trace, copy(snapshot.xyz_full))
                end
            end
        end
        loss
    end

    objective
end

function parameter_bounds(problem::OptimizationProblem)
    bounds = problem.parameters.bounds
    lower = copy(bounds.lower)
    upper = copy(bounds.upper)
    nvar = length(problem.anchors.variable_indices)
    if nvar > 0
        lower = vcat(lower, fill(-Inf, 3nvar))
        upper = vcat(upper, fill(Inf, 3nvar))
    end
    lower, upper
end

function make_gradient(objective)
    function g!(G, θ)
        grad = gradient(objective, θ)[1]
        copyto!(G, grad)
    end
    g!
end

function optimize_problem!(problem::OptimizationProblem, state::OptimizationState; on_iteration=nothing)
    solver = problem.parameters.solver
    lower_bounds, upper_bounds = parameter_bounds(problem)
    θ0 = pack_parameters(problem, state)
    
    # Initialize cache if needed
    if state.cache === nothing
        state.cache = initialize_fdm_cache(problem)
    end
    cache = state.cache

    # Phase 1: ADMM (Global Search)
    if solver.use_admm
        @info "Starting ADMM Global Search..."
        run_admm!(problem, state, cache, θ0; on_iteration=on_iteration)
        θ0 = pack_parameters(problem, state) # Update start for L-BFGS
    end

    # Phase 2: L-BFGS (Local Refinement)
    if solver.use_lbfgs
        @info "Starting L-BFGS Refinement..."
        # Precompute barrier indices
        lb_idx = findall(isfinite, lower_bounds)
        ub_idx = findall(isfinite, upper_bounds)
        
        # check initial bounds
        initial_violations_lower = θ0[lb_idx] .< lower_bounds[lb_idx]
        initial_violations_upper = θ0[ub_idx] .> upper_bounds[ub_idx]
        if any(initial_violations_lower) || any(initial_violations_upper)
            @warn "Initial guess is outside bounds" num_lower=sum(initial_violations_lower) num_upper=sum(initial_violations_upper)
        end

        # Auto-scaling
        geo_scale = 1.0
        if solver.use_auto_scaling
            q0, a0 = unpack_parameters(problem, θ0)
            snap0 = evaluate_geometry(problem, q0, a0)
            L0 = total_loss(problem, q0, a0, snap0)
            geo_scale = 1.0 / max(L0, 1e-6)
        end

        objective = form_finding_objective(problem, state, lower_bounds, upper_bounds, lb_idx, ub_idx, geo_scale, solver.barrier_weight, solver.barrier_sharpness)
        gradient! = make_gradient(objective)
        
        outer_iter = Ref(0)
        callback = function (_opt_state)
            outer_iter[] += 1
            state.iterations = outer_iter[]
            if on_iteration === nothing || isempty(state.loss_trace)
                return false
            end
            snapshot = evaluate_geometry(problem, state.force_densities, state.variable_anchor_positions)
            loss = state.loss_trace[end]
            on_iteration(state, snapshot, loss)
            return false
        end
        
        result = Optim.optimize(
            objective,
            gradient!,
            θ0,
            LBFGS(),
            Optim.Options(
                iterations = solver.max_iterations,
                f_abstol = solver.absolute_tolerance,
                f_reltol = solver.relative_tolerance,
                callback = callback,
            ),
        )

        min_θ = Optim.minimizer(result)
        q, anchors = unpack_parameters(problem, min_θ)
        state.force_densities = copy(q)
        state.variable_anchor_positions = copy(anchors)
        snapshot = evaluate_geometry(problem, q, anchors)

        return result, snapshot
    end

    # If only ADMM ran, we still return a result
    return :ADMM_ONLY, evaluate_geometry(problem, state.force_densities, state.variable_anchor_positions)
end

function run_admm!(problem::OptimizationProblem, state::OptimizationState, cache::FDMCache, θ0; on_iteration=nothing)
    q, anchors = unpack_parameters(problem, θ0)
    copyto!(cache.q_gpu, q)
    
    # Initialize targets on GPU
    extract_admm_targets!(problem, cache)
    
    m = problem.topology.num_edges
    threads = 256 # multiple of 32 (warp size), good balance of occupancy and resource usage
    blocks = ceil(Int, m / threads)
    
    solver = problem.parameters.solver
    
    for i in 1:solver.admm_iterations
        # 1. x-update: Physics (Sparse Solve on CPU)
        # Pull current force densities to CPU
        copyto!(cache.q_cpu, cache.q_gpu)
        
        # solve_fdm! updates cache.x_gpu internally
        solve_fdm!(problem, cache.q_cpu, anchors, cache)
        
        # 2. z-update: Geometric Projections (GPU)
        @cuda threads=threads blocks=blocks kernel_admm_z_update!(
            cache.x_gpu, cache.edge_nodes, cache.y_dual, cache.z_consensus,
            cache.l_min_consolidated, cache.l_max_consolidated, cache.f_target_consolidated, cache.q_gpu
        )
        
        # 3. Dual and Force Density Update (GPU)
        @cuda threads=threads blocks=blocks kernel_admm_dual_q_update!(
            cache.x_gpu, cache.edge_nodes, cache.z_consensus, cache.y_dual, 
            cache.q_gpu, solver.rho
        )
        
        # Progress Reporting
        if solver.show_progress && (i % solver.report_frequency == 0)
            @info "ADMM Iteration $i/$(solver.admm_iterations)"
        end

        if on_iteration !== nothing && (i % solver.report_frequency == 0)
            state.force_densities .= Vector(cache.q_gpu)
            snapshot = evaluate_geometry(problem, state.force_densities, anchors)
            on_iteration(state, snapshot, 0.0)
        end
    end
    
    # Final synchronization
    state.force_densities .= Vector(cache.q_gpu)
    return state
end

function extract_admm_targets!(problem::OptimizationProblem, cache::FDMCache)
    m = problem.topology.num_edges
    l_min = fill(0.0, m)
    l_max = fill(1e10, m)
    f_target = fill(-1.0, m)
    
    for obj in problem.parameters.objectives
        if obj isa MinLengthObjective
            l_min[obj.edge_indices] .= max.(l_min[obj.edge_indices], obj.threshold)
        elseif obj isa MaxLengthObjective
            l_max[obj.edge_indices] .= min.(l_max[obj.edge_indices], obj.threshold)
        elseif obj isa TargetLengthObjective
            l_min[obj.edge_indices] .= obj.target
            l_max[obj.edge_indices] .= obj.target
        elseif obj isa MinForceObjective
            # Force targets are handled via length adjustments in the kernel if needed, 
            # but here we can set a target force if we want the kernel to prioritize it.
            f_target[obj.edge_indices] .= max.(f_target[obj.edge_indices], obj.threshold)
        end
    end
    
    copyto!(cache.l_min_consolidated, l_min)
    copyto!(cache.l_max_consolidated, l_max)
    copyto!(cache.f_target_consolidated, f_target)
end
