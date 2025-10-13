using LinearAlgebra
using Optim
using Zygote
using ChainRulesCore: ignore_derivatives

struct GeometrySnapshot
    xyz_free::Matrix{Float64}
    xyz_fixed::Matrix{Float64}
    xyz_full::Matrix{Float64}
    member_lengths::Vector{Float64}
    member_forces::Vector{Float64}
end

bounds_penalty(state_q::Vector{Float64}, bounds::Bounds) = pBounds(
    state_q,
    bounds.lower,
    bounds.upper,
    DEFAULT_BARRIER_SHARPNESS,
    DEFAULT_BARRIER_SHARPNESS,
)

function evaluate_geometry(problem::OptimizationProblem, q::Vector{Float64}, anchor_positions::Matrix{Float64})
    fixed_positions = current_fixed_positions(problem, anchor_positions)
    xyz_free = solve_explicit(q, problem.topology.free_incidence, problem.topology.fixed_incidence, problem.loads.free_node_loads, fixed_positions)
    xyz_full = vcat(xyz_free, fixed_positions)
    member_vectors = problem.topology.incidence * xyz_full
    member_lengths = map(norm, eachrow(member_vectors))
    member_forces = q .* member_lengths
    GeometrySnapshot(xyz_free, fixed_positions, xyz_full, member_lengths, member_forces)
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

objective_loss(::AbstractObjective, ::GeometrySnapshot) = 0.0

function total_loss(problem::OptimizationProblem, q::Vector{Float64}, anchor_positions::Matrix{Float64}, snapshot::GeometrySnapshot)
    loss = bounds_penalty(q, problem.parameters.bounds)
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

function unpack_parameters(problem::OptimizationProblem, θ::AbstractVector{<:Real})
    ne = problem.topology.num_edges
    q = Float64.(θ[1:ne])
    nvar = length(problem.anchors.variable_indices)
    if nvar == 0
        return q, zeros(Float64, 0, 3)
    end
    anchors = reshape(θ[ne + 1:end], 3, nvar)'
    q, anchors
end

function form_finding_objective(problem::OptimizationProblem, trace_state::OptimizationState; on_iteration=nothing)
    empty!(trace_state.loss_trace)
    empty!(trace_state.node_trace)
    trace_state.iterations = 0

    function objective(θ)
        q, anchors = unpack_parameters(problem, θ)
        snapshot = evaluate_geometry(problem, q, anchors)
        loss = total_loss(problem, q, anchors, snapshot)
        if !isderiving()
            ignore_derivatives() do
                trace_state.iterations += 1
                trace_state.force_densities = copy(q)
                trace_state.variable_anchor_positions = copy(anchors)
                push!(trace_state.loss_trace, loss)
                if problem.parameters.tracing.record_nodes
                    push!(trace_state.node_trace, copy(snapshot.xyz_full))
                end
                if on_iteration !== nothing
                    on_iteration(trace_state, snapshot, loss)
                end
            end
        end
        loss
    end

    objective
end

function make_gradient(objective)
    function g!(G, θ)
        grad = gradient(objective, θ)[1]
        copyto!(G, grad)
    end
    g!
end

function optimize_problem!(problem::OptimizationProblem, state::OptimizationState; on_iteration=nothing)
    objective = form_finding_objective(problem, state; on_iteration=on_iteration)
    gradient! = make_gradient(objective)
    θ0 = pack_parameters(problem, state)
    result = Optim.optimize(
        objective,
        gradient!,
        θ0,
        LBFGS(),
        Optim.Options(
            iterations = problem.parameters.solver.max_iterations,
            f_abstol = problem.parameters.solver.absolute_tolerance,
            f_reltol = problem.parameters.solver.relative_tolerance,
        ),
    )

    min_θ = Optim.minimizer(result)
    q, anchors = unpack_parameters(problem, min_θ)
    state.force_densities = copy(q)
    state.variable_anchor_positions = copy(anchors)
    snapshot = evaluate_geometry(problem, q, anchors)
    return result, snapshot
end
