using LinearAlgebra
using Optim
using Zygote
using ChainRulesCore: ignore_derivatives
using Logging

struct GeometrySnapshot{TF, TX, TA, VL, VF, TR}
    xyz_free::TF
    xyz_fixed::TX
    xyz_full::TA
    member_lengths::VL
    member_forces::VF
    reactions::TR
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

function form_finding_objective(problem::OptimizationProblem, trace_state::OptimizationState)
    empty!(trace_state.loss_trace)
    empty!(trace_state.node_trace)
    trace_state.iterations = 0

    function objective(θ)
        q, anchors = unpack_parameters(problem, θ)
        snapshot = evaluate_geometry(problem, q, anchors)
        loss = total_loss(problem, q, anchors, snapshot)
        if !isderiving()
            ignore_derivatives() do
                trace_state.force_densities = copy(q)
                trace_state.variable_anchor_positions = copy(anchors)
                push!(trace_state.loss_trace, loss)
                if problem.parameters.tracing.record_nodes
                    push!(trace_state.node_trace, copy(snapshot.xyz_full))
                end
            end
        end
        loss
    end

    objective
end

function log_lambda_multipliers(min_θ::AbstractVector{<:Real}, grad::AbstractVector{<:Real}, lower::AbstractVector{<:Real}, upper::AbstractVector{<:Real})
    tol = sqrt(eps(Float64))
    lower_values = Float64[]
    upper_values = Float64[]
    for i in eachindex(min_θ)
        if isfinite(lower[i]) && min_θ[i] <= lower[i] + tol
            push!(lower_values, max(0.0, grad[i]))
        elseif isfinite(upper[i]) && min_θ[i] >= upper[i] - tol
            push!(upper_values, max(0.0, -grad[i]))
        end
    end

    stats(values) = isempty(values) ? (count=0, max=0.0, sum=0.0) : (count=length(values), max=maximum(values), sum=sum(values))
    lower_stats = stats(lower_values)
    upper_stats = stats(upper_values)

    @info "KKT multipliers" lower_active=lower_stats.count upper_active=upper_stats.count max_lower=lower_stats.max max_upper=upper_stats.max sum_lower=lower_stats.sum sum_upper=upper_stats.sum
    @debug "Lower-bound multipliers" values=lower_values
    @debug "Upper-bound multipliers" values=upper_values
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
    objective = form_finding_objective(problem, state)
    gradient! = make_gradient(objective)
    θ0 = pack_parameters(problem, state)
    lower_bounds, upper_bounds = parameter_bounds(problem)
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
        lower_bounds,
        upper_bounds,
        θ0,
        Fminbox(LBFGS()),
        Optim.Options(
            iterations = problem.parameters.solver.max_iterations,
            f_abstol = problem.parameters.solver.absolute_tolerance,
            f_reltol = problem.parameters.solver.relative_tolerance,
            callback = callback,
        ),
    )

    min_θ = Optim.minimizer(result)
    q, anchors = unpack_parameters(problem, min_θ)
    state.force_densities = copy(q)
    state.variable_anchor_positions = copy(anchors)
    snapshot = evaluate_geometry(problem, q, anchors)

    grad_at_solution = gradient(objective, min_θ)[1]
    log_lambda_multipliers(min_θ, grad_at_solution, lower_bounds, upper_bounds)
    return result, snapshot
end
