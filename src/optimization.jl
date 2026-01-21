using LinearAlgebra
using Optim
using Mooncake
using ChainRulesCore: ignore_derivatives
using Logging
using CUDA

"""
    evaluate_geometry!(cache::FDMCache, q::AbstractVector{Float64}, problem::OptimizationProblem)

In-place geometry evaluation using GPU kernels.
"""
function evaluate_geometry!(cache::FDMCache, q::AbstractVector{Float64}, problem::OptimizationProblem)
    # 1. Solve FDM on CPU
    x_free = solve_fdm!(cache, q, problem)
    
    # 2. Update GPU buffers
    # Sync nodal positions to x_gpu
    fixed_pos = current_fixed_positions(problem, zeros(0,3))
    copyto!(cache.x_gpu, [x_free; fixed_pos]) # Ideally this should be more direct
    
    # Sync q to q_gpu
    copyto!(cache.q_gpu, q)
    
    # 3. Compute L and F on GPU
    n_edges = size(cache.edge_nodes, 1)
    threads = 256
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
        
        geometric_loss = total_loss(problem, q, anchors, snapshot)
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
    @info "ADMM implementation in progress..."
    # TODO: Implement ADMM consensus loop
    return state
end
