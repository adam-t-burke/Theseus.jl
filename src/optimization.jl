using LinearAlgebra
using Optim
using Mooncake
using Logging

function evaluate_geometry!(problem::OptimizationProblem{T}, q::AbstractVector{Float64}, anchor_positions::AbstractMatrix{Float64}) where {T}
    ctx = problem.context
    nf = length(problem.topology.free_node_indices)
    
    # In-place updates
    update_fixed_positions!(ctx, problem.anchors, anchor_positions)
    solve_explicit!(ctx, q, problem.topology.free_incidence, problem.topology.fixed_incidence, problem.loads.free_node_loads, ctx.fixed_positions)
    
    # xyz_full assembly
    ctx.xyz_full[1:nf, :] .= ctx.xyz_free
    # fixed positions in the global indexing are at the end
    ctx.xyz_full[nf+1:end, :] .= ctx.fixed_positions
    
    # incidence * xyz_full
    mul!(ctx.member_vectors, problem.topology.incidence, ctx.xyz_full)
    
    # member_lengths
    for i in 1:length(ctx.member_lengths)
        @inbounds ctx.member_lengths[i] = sqrt(ctx.member_vectors[i,1]^2 + ctx.member_vectors[i,2]^2 + ctx.member_vectors[i,3]^2)
    end
    
    # member_forces
    ctx.member_forces .= q .* ctx.member_lengths
    
    # reactions
    anchor_reactions!(ctx, problem.topology, q)
    
    return GeometrySnapshot(ctx.xyz_free, ctx.fixed_positions, ctx.xyz_full, ctx.member_lengths, ctx.member_forces, ctx.reactions)
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

function total_loss(problem::OptimizationProblem{T}, q::AbstractVector{Float64}, anchor_positions::AbstractMatrix{Float64}, snapshot::GeometrySnapshot) where {T}
    loss = 0.0
    for obj in problem.parameters.objectives
        loss += objective_loss(obj, snapshot)
    end
    return loss
end

function pack_parameters(problem::OptimizationProblem{T}, state::OptimizationState) where {T}
    if isempty(problem.anchors.variable_indices)
        return copy(state.force_densities)
    end
    return vcat(state.force_densities, vec(state.variable_anchor_positions))
end

function unpack_parameters(problem::OptimizationProblem{T}, θ::AbstractVector{S}) where {T, S<:Real}
    ne = problem.topology.num_edges
    q = copy(@view θ[1:ne])
    nvar = length(problem.anchors.variable_indices)
    if nvar == 0
        return q, zeros(S, 0, 3)
    end
    anchors = reshape(copy(@view θ[ne + 1:end]), 3, nvar)'
    return q, collect(anchors)
end

function log_trace_info!(trace_state::OptimizationState, q::AbstractVector{Float64}, anchors::AbstractMatrix{Float64}, geometric_loss::Float64, barrier_loss::Float64, snapshot::GeometrySnapshot)
    trace_state.force_densities .= q
    if !isempty(anchors)
        trace_state.variable_anchor_positions .= anchors
    end
    push!(trace_state.loss_trace, geometric_loss)
    push!(trace_state.penalty_trace, barrier_loss)
    # push!(trace_state.node_trace, copy(snapshot.xyz_full))
    return nothing
end

struct ObjectiveWrapper{TFact}
    problem::OptimizationProblem{TFact}
    trace_state::OptimizationState
    lb::Vector{Float64}
    ub::Vector{Float64}
    lb_idx::Vector{Int64}
    ub_idx::Vector{Int64}
    geo_scale::Float64
    barrier_weight::Float64
    sharpness::Float64
end

function (obj::ObjectiveWrapper{TFact})(θ) where {TFact}
    q, anchors = unpack_parameters(obj.problem, θ)
    snapshot = evaluate_geometry!(obj.problem, q, anchors)
    
    geometric_loss = total_loss(obj.problem, q, anchors, snapshot)
    barrier_loss = pBounds(θ, obj.lb, obj.ub, obj.lb_idx, obj.ub_idx, obj.sharpness, obj.sharpness)
    
    loss = (geometric_loss * obj.geo_scale) + (barrier_loss * obj.barrier_weight)
    
    log_trace_info!(obj.trace_state, q, anchors, geometric_loss, barrier_loss, snapshot)
    
    loss
end

function form_finding_objective(problem::OptimizationProblem{TFact}, trace_state::OptimizationState, lb, ub, lb_idx, ub_idx, geo_scale, barrier_weight, sharpness) where {TFact}
    empty!(trace_state.loss_trace)
    empty!(trace_state.penalty_trace)
    empty!(trace_state.node_trace)
    trace_state.iterations = 0

    return ObjectiveWrapper{TFact}(problem, trace_state, lb, ub, lb_idx, ub_idx, geo_scale, barrier_weight, sharpness)
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

function make_gradient(objective, θ0)
    # Using friendly_tangents=false avoids complex struct recreation Mooncake's "friendly" conversion,
    # which is likely failing on your objective struct/closure due to captured factorizations 
    # or Julia 1.12 memory layout changes.
    cache = Mooncake.prepare_gradient_cache(objective, θ0; friendly_tangents=false)
    function g!(G, θ)
        _, grads = Mooncake.value_and_gradient!!(
            cache, objective, θ; 
            args_to_zero=(false, true)
        )
        # With friendly_tangents=false, the gradient is often a Tangent or MutableTangent.
        # For a Vector input, it should be obtainable via .fields or directly if it's Bits.
        if grads[2] isa AbstractVector
            copyto!(G, grads[2])
        else
            copyto!(G, grads[2].fields)
        end
    end
    g!
end

function optimize_problem!(problem::OptimizationProblem{T}, state::OptimizationState; on_iteration=nothing) where {T}
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

    # Auto-scaling
    geo_scale = 1.0
    if solver.use_auto_scaling
        q0, a0 = unpack_parameters(problem, θ0)
        snap0 = evaluate_geometry!(problem, q0, a0)
        L0 = total_loss(problem, q0, a0, snap0)
        geo_scale = 1.0 / max(L0, 1e-6)
    end

    objective = form_finding_objective(problem, state, lower_bounds, upper_bounds, lb_idx, ub_idx, geo_scale, solver.barrier_weight, solver.barrier_sharpness)
    gradient! = make_gradient(objective, θ0)
    
    outer_iter = Ref(0)
    callback = function (_opt_state)
        outer_iter[] += 1
        state.iterations = outer_iter[]
        if on_iteration === nothing || isempty(state.loss_trace)
            return false
        end
        snapshot = evaluate_geometry!(problem, state.force_densities, state.variable_anchor_positions)
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
    snapshot = evaluate_geometry!(problem, q, anchors)

    return result, snapshot
end
