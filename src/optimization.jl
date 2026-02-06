using LinearAlgebra
using Optim
import Mooncake
import ADTypes
import DifferentiationInterface
using TimerOutputs

function evaluate_geometry(problem::OptimizationProblem, q::AbstractVector{<:Real}, anchor_positions::AbstractMatrix{<:Real}, cache::OptimizationCache)
    # In-place update of cache buffers
    solve_FDM!(cache, q, problem, anchor_positions)
    xyz_full = cache.Nf
    ne = length(cache.member_lengths)
    for i in 1:ne
        s = cache.edge_starts[i]
        e = cache.edge_ends[i]
        
        dx = xyz_full[e, 1] - xyz_full[s, 1]
        dy = xyz_full[e, 2] - xyz_full[s, 2]
        dz = xyz_full[e, 3] - xyz_full[s, 3]
        
        len = sqrt(dx^2 + dy^2 + dz^2)
        cache.member_lengths[i] = len
        cache.member_forces[i] = q[i] * len
    end

    fill!(cache.reactions, 0.0)
    for i in 1:ne
        s = cache.edge_starts[i]
        e = cache.edge_ends[i]
        qi = q[i]
        
        rx = (xyz_full[e, 1] - xyz_full[s, 1]) * qi
        ry = (xyz_full[e, 2] - xyz_full[s, 2]) * qi
        rz = (xyz_full[e, 3] - xyz_full[s, 3]) * qi
        
        cache.reactions[s, 1] += rx
        cache.reactions[s, 2] += ry
        cache.reactions[s, 3] += rz
        
        cache.reactions[e, 1] -= rx
        cache.reactions[e, 2] -= ry
        cache.reactions[e, 3] -= rz
    end

    xyz_free = @view xyz_full[problem.topology.free_node_indices, :]
    xyz_fixed = @view xyz_full[problem.topology.fixed_node_indices, :]
    
    snapshot = GeometrySnapshot(xyz_free, xyz_fixed, xyz_full, cache.member_lengths, cache.member_forces, cache.reactions)
    cache.last_snapshot = snapshot
    return snapshot
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
    loss = zero(eltype(snapshot.member_lengths))
    for idx in edges
        loss += snapshot.member_lengths[idx] * snapshot.member_forces[idx]
    end
    obj.weight * loss
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
    # Note: we use state.cache.to if we have access to it, but total_loss 
    # doesn't take cache. We could pass it, but for now we'll just sum.
    # Actually, evaluate_geometry already updated the cache.
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

function form_finding_objective(problem::OptimizationProblem, cache::OptimizationCache, lb, ub, lb_idx, ub_idx, barrier_weight, sharpness)
    function objective(θ)
        q, anchors = unpack_parameters(problem, θ)
        snapshot = evaluate_geometry(problem, q, anchors, cache)
        
        geometric_loss = total_loss(problem, q, anchors, snapshot)
        barrier_loss = pBounds(θ, lb, ub, lb_idx, ub_idx, sharpness, sharpness)
        
        loss = geometric_loss + (barrier_loss * barrier_weight)
        return loss
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

function optimize_problem!(problem::OptimizationProblem, state::OptimizationState; on_iteration=nothing)
    # Ensure cache is initialized
    if isnothing(state.cache)
        state.cache = OptimizationCache(problem)
    end
    reset_timer!(state.cache.to)

    # Clear traces
    empty!(state.loss_trace)
    empty!(state.penalty_trace)
    empty!(state.node_trace)
    state.iterations = 0

    solver = problem.parameters.solver
    lower_bounds, upper_bounds = parameter_bounds(problem)
    θ0 = pack_parameters(problem, state)
    
    # Precompute barrier indices
    lb_idx = findall(isfinite, lower_bounds)
    ub_idx = findall(isfinite, upper_bounds)
    
    # check initial bounds
    initial_violations_lower = θ0[lb_idx] .< lower_bounds[lb_idx]
    initial_violations_upper = θ0[ub_idx] .> upper_bounds[ub_idx]
    if any(initial_violations_lower) || any(initial_violations_upper)
        @warn "Initial guess is outside bounds" num_lower=sum(initial_violations_lower) num_upper=sum(initial_violations_upper)
    end

    objective = form_finding_objective(problem, state.cache, lower_bounds, upper_bounds, lb_idx, ub_idx, solver.barrier_weight, solver.barrier_sharpness)
    
    # Setup Mooncake backend via DifferentiationInterface
    backend = ADTypes.AutoMooncake(config=Mooncake.Config(friendly_tangents=true))
    prep = DifferentiationInterface.prepare_gradient(objective, backend, θ0)
    
    function fg!(F, G, θ)
        if G !== nothing
            val, _ = DifferentiationInterface.value_and_gradient!(objective, G, prep, backend, θ)
            return val
        elseif F !== nothing
            return objective(θ)
        end
        return nothing
    end
    
    callback = function (os)
        state.iterations = os.iteration
        
        # Unpack current θ
        # Note: We don't perform a NEW evaluation here because fg!/objective already did it
        # and stored the result in state.cache.last_snapshot
        snapshot = state.cache.last_snapshot
        q = snapshot.member_forces ./ snapshot.member_lengths # approximate q if needed, but better to get from metadata if possible
        # Actually, let's just use the metadata x if we really need q, anchors for state
        q_new, anchors_new = unpack_parameters(problem, os.metadata["x"])
        state.force_densities = q_new
        state.variable_anchor_positions = anchors_new
        
        push!(state.loss_trace, os.value)
        penalty = pBounds(os.metadata["x"], lower_bounds, upper_bounds, lb_idx, ub_idx, solver.barrier_sharpness, solver.barrier_sharpness)
        push!(state.penalty_trace, penalty * solver.barrier_weight)
        
        if problem.parameters.tracing.record_nodes
            push!(state.node_trace, copy(snapshot.xyz_full))
        end
        
        if on_iteration !== nothing
            on_iteration(state, snapshot, os.value)
        end
        return false
    end
    
    result = Optim.optimize(
        Optim.only_fg!(fg!),
        θ0,
        LBFGS(),
        Optim.Options(
            iterations = solver.max_iterations,
            f_abstol = solver.absolute_tolerance, # Change in objective
            f_reltol = solver.relative_tolerance, # Relative change in objective
            g_abstol = solver.absolute_tolerance, # Gradient norm
            g_tol = solver.relative_tolerance,    # Gradient tolerance (often used as relative)
            callback = callback,
            extended_trace = true
        ),
    )

    min_θ = Optim.minimizer(result)
    q, anchors = unpack_parameters(problem, min_θ)
    state.force_densities = copy(q)
    state.variable_anchor_positions = copy(anchors)
    # One last evaluation with the cache to ensure state is consistent and get the final snapshot
    snapshot = evaluate_geometry(problem, q, anchors, state.cache)

    return result, snapshot
end
