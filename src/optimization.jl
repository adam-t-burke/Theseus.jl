using LinearAlgebra
using Optim
using DifferentiationInterface

const MOONCAKE_BACKEND = AutoMooncake()

struct MaxEvaluationLimit <: Exception
    max_evals::Int
end

Base.showerror(io::IO, err::MaxEvaluationLimit) = print(io, "Maximum evaluation limit of $(err.max_evals) reached")

struct TheseusLimitedResult
    minimizer::Vector{Float64}
    minimum::Float64
    iterations::Int
    f_calls::Int
    message::String
end

import Optim: minimizer, minimum, iterations, converged, f_calls

minimizer(res::TheseusLimitedResult) = res.minimizer
minimum(res::TheseusLimitedResult) = res.minimum
iterations(res::TheseusLimitedResult) = res.iterations
f_calls(res::TheseusLimitedResult) = res.f_calls
converged(::TheseusLimitedResult) = false

function evaluate_geometry!(workspace::GeometryWorkspace, problem::OptimizationProblem, q::Vector{Float64}, anchor_positions::Matrix{Float64})
    cache = workspace.fdm.cache
    fixed_positions = cache.fixed_positions
    current_fixed_positions!(fixed_positions, problem, anchor_positions)
    solve_geometry(workspace.fdm, q, problem.loads)

    nf = size(workspace.xyz_free, 1)
    total_nodes = problem.topology.num_nodes
    xyz_full = workspace.xyz_full

    @inbounds for row in 1:nf, col in 1:3
        xyz_full[row, col] = workspace.xyz_free[row, col]
    end
    if nf < total_nodes
        fixed_rows = total_nodes - nf
        @inbounds for row in 1:fixed_rows, col in 1:3
            xyz_full[nf + row, col] = fixed_positions[row, col]
        end
    end

    mul!(workspace.member_vectors, problem.topology.incidence, xyz_full)

    @inbounds for edge in 1:problem.topology.num_edges
        vx = workspace.member_vectors[edge, 1]
        vy = workspace.member_vectors[edge, 2]
        vz = workspace.member_vectors[edge, 3]
        length = sqrt(vx * vx + vy * vy + vz * vz)
        workspace.member_lengths[edge] = length
        workspace.member_forces[edge] = q[edge] * length
    end

    workspace.snapshot
end

function objective_loss(obj::TargetXYZObjective, snapshot::GeometrySnapshot)
    target_xyz(snapshot.xyz_full, obj.target, obj.node_indices) * obj.weight
end

function objective_loss(obj::TargetXYObjective, snapshot::GeometrySnapshot)
    target_xy(snapshot.xyz_full, obj.target, obj.node_indices) * obj.weight
end

function objective_loss(obj::TargetLengthObjective, snapshot::GeometrySnapshot)
    lenTarget(snapshot.member_lengths, obj.target, obj.edge_indices) * obj.weight
end

function objective_loss(obj::LengthVariationObjective, snapshot::GeometrySnapshot)
    lenVar(snapshot.member_lengths, obj.edge_indices) * obj.weight
end

function objective_loss(obj::ForceVariationObjective, snapshot::GeometrySnapshot)
    forceVar(snapshot.member_forces, obj.edge_indices) * obj.weight
end

function objective_loss(obj::SumForceLengthObjective, snapshot::GeometrySnapshot)
    indices = obj.edge_indices
    total = 0.0
    @inbounds for idx in indices
        total += snapshot.member_lengths[idx] * snapshot.member_forces[idx]
    end
    obj.weight * total
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
    loss = 0.0
    for obj in problem.parameters.objectives
        loss += objective_loss(obj, snapshot)
    end
    loss
end

function pack_parameters(problem::OptimizationProblem, state::OptimizationState)
    ne = problem.topology.num_edges
    nvar = size(state.variable_anchor_positions, 1)
    if nvar == 0
        return copy(state.force_densities)
    end
    θ = Vector{Float64}(undef, ne + 3 * nvar)
    copyto!(θ, 1, state.force_densities, 1, ne)
    offset = ne
    @inbounds for row in 1:nvar, col in 1:3
        θ[offset + (row - 1) * 3 + col] = state.variable_anchor_positions[row, col]
    end
    θ
end

function unpack_parameters!(problem::OptimizationProblem, θ::AbstractVector{<:Real}, q_buffer::Vector{Float64}, anchor_buffer::Matrix{Float64})
    ne = problem.topology.num_edges
    @inbounds for i in 1:ne
        q_buffer[i] = Float64(θ[i])
    end
    nvar = size(anchor_buffer, 1)
    if nvar > 0
        offset = ne
        @inbounds for row in 1:nvar, col in 1:3
            anchor_buffer[row, col] = Float64(θ[offset + (row - 1) * 3 + col])
        end
    end
    q_buffer, anchor_buffer
end

function ensure_parameter_bounds!(workspace::GeometryWorkspace, problem::OptimizationProblem, state::OptimizationState)
    bounds = problem.parameters.bounds
    ne = problem.topology.num_edges
    nvar = size(state.variable_anchor_positions, 1)
    total = ne + 3 * nvar
    lower = workspace.theta_lower
    upper = workspace.theta_upper

    if length(lower) != total
        resize!(lower, total)
        resize!(upper, total)
    end

    if total == 0
        return lower, upper
    end

    copyto!(lower, 1, bounds.lower, 1, ne)
    copyto!(upper, 1, bounds.upper, 1, ne)

    if total > ne
        start_idx = ne + 1
        @inbounds for i in start_idx:total
            lower[i] = -Inf
            upper[i] = Inf
        end
    end

    lower, upper
end

function validate_initial_parameters(θ::AbstractVector{<:Real}, lower::Vector{Float64}, upper::Vector{Float64})
    if (length(lower) != length(upper)) || (length(lower) != length(θ))
        throw(ArgumentError("Bounds vectors must match parameter length"))
    end
    @inbounds for i in eachindex(θ)
        lo = lower[i]
        hi = upper[i]
        val = Float64(θ[i])
        if (isfinite(lo) && val < lo) || (isfinite(hi) && val > hi)
            throw(ArgumentError("Initial parameter θ[$i] = $val violates bounds [$lo, $hi]"))
        end
    end
    θ
end

function objective_core!(ctx::GeometryWorkspace, problem::OptimizationProblem, θ::AbstractVector{<:Real}; record::Bool=false, state::Union{OptimizationState,Nothing}=nothing)
    q, anchors = unpack_parameters!(problem, θ, ctx.q_buffer, ctx.anchor_buffer)
    snapshot = evaluate_geometry!(ctx, problem, q, anchors)
    loss = total_loss(problem, q, anchors, snapshot)

    if record
        state === nothing && throw(ArgumentError("Recording objective evaluations requires a state"))
        copy!(state.force_densities, q)
        if size(state.variable_anchor_positions, 1) == size(anchors, 1)
            copy!(state.variable_anchor_positions, anchors)
        end
        state.iterations += 1
        push!(state.loss_trace, loss)
        if problem.parameters.tracing.record_nodes
            push!(state.node_trace, copy(snapshot.xyz_full))
        end
    end

    loss
end

function objective_core(
    θ::AbstractVector{<:Real},
    fdm_ctx::DifferentiationInterface.Cache,
    workspace_ctx::DifferentiationInterface.Cache,
    problem_const::DifferentiationInterface.Constant,
)
    ctx = DifferentiationInterface.unwrap(fdm_ctx)
    workspace = DifferentiationInterface.unwrap(workspace_ctx)
    problem = DifferentiationInterface.unwrap(problem_const)
    objective_core(θ, ctx, workspace, problem)
end

function objective_core(
    θ::AbstractVector{<:Real},
    ctx::FDMContext,
    workspace::GeometryWorkspace,
    problem::OptimizationProblem,
)
    current_ctx = getfield(workspace, :fdm)
    current_ctx === ctx || setfield!(workspace, :fdm, ctx)
    objective_core!(workspace, problem, θ)
end

function objective_core(θ::AbstractVector{<:Real}, ctx::GeometryWorkspace, problem::OptimizationProblem)
    objective_core(
        θ,
        ctx.fdm,
        ctx,
        problem,
    )
end

function objective_context(problem::OptimizationProblem, state::OptimizationState)
    DifferentiationInterface.Cache(state.workspace.fdm),
    DifferentiationInterface.Cache(state.workspace),
    DifferentiationInterface.Constant(problem)
end

function evaluate_objective!(problem::OptimizationProblem, state::OptimizationState, θ::AbstractVector{<:Real}, record::Bool)
    objective_core!(state.workspace, problem, θ; record=record, state=record ? state : nothing)
end

function prepare_objectives(problem::OptimizationProblem, state::OptimizationState, max_evals::Int; on_iteration=nothing)
    objective_record = θ -> begin
        if state.iterations >= max_evals
            throw(MaxEvaluationLimit(max_evals))
        end
        loss = objective_core!(state.workspace, problem, θ; record=true, state=state)
        if on_iteration !== nothing
            on_iteration(state, state.workspace.snapshot, loss)
        end
        loss
    end

    objective_plain = objective_core

    objective_record, objective_plain
end

function make_gradient(objective_plain, prep, fdm_ctx_cache, workspace_ctx, problem_const)
    function g!(G, θ)
        DifferentiationInterface.gradient!(
            objective_plain,
            G,
            prep,
            MOONCAKE_BACKEND,
            θ,
            fdm_ctx_cache,
            workspace_ctx,
            problem_const,
        )
        nothing
    end
    g!
end

function optimize_problem!(problem::OptimizationProblem, state::OptimizationState; on_iteration=nothing)
    empty!(state.loss_trace)
    empty!(state.node_trace)
    state.iterations = 0

    workspace = state.workspace
    lower, upper = ensure_parameter_bounds!(workspace, problem, state)

    solver_opts = problem.parameters.solver
    max_iters = max(1, solver_opts.max_iterations)
    sizehint!(state.loss_trace, max_iters)
    if problem.parameters.tracing.record_nodes
        sizehint!(state.node_trace, max_iters)
    end
    objective_record, objective_plain = prepare_objectives(problem, state, max_iters; on_iteration=on_iteration)
    θ0 = pack_parameters(problem, state)
    validate_initial_parameters(θ0, lower, upper)
    fdm_cache_ctx, workspace_ctx, problem_const = objective_context(problem, state)
    prep = DifferentiationInterface.prepare_gradient(
        objective_plain,
        MOONCAKE_BACKEND,
        θ0,
        fdm_cache_ctx,
        workspace_ctx,
        problem_const,
    )
    gradient! = make_gradient(objective_plain, prep, fdm_cache_ctx, workspace_ctx, problem_const)

    options = Optim.Options(
        iterations = max_iters,
        outer_iterations = max_iters,
        f_calls_limit = max_iters,
        g_calls_limit = max_iters,
        show_trace = solver_opts.show_progress,
        show_every = max(1, solver_opts.report_frequency),
        f_abstol = solver_opts.absolute_tolerance,
        f_reltol = solver_opts.relative_tolerance,
    )

    result = try
        Optim.optimize(
            objective_record,
            gradient!,
            lower,
            upper,
            θ0,
            Optim.Fminbox(LBFGS()),
            options,
        )
    catch err
        if err isa MaxEvaluationLimit
            θ_current = pack_parameters(problem, state)
            loss = isempty(state.loss_trace) ? Inf : state.loss_trace[end]
            TheseusLimitedResult(θ_current, loss, max_iters, state.iterations, "Maximum evaluations $(err.max_evals) reached")
        else
            rethrow()
        end
    end

    if state.iterations > max_iters
        state.iterations = max_iters
    end

    min_θ = Optim.minimizer(result)
    q, anchors = unpack_parameters!(problem, min_θ, workspace.q_buffer, workspace.anchor_buffer)
    snapshot = evaluate_geometry!(workspace, problem, q, anchors)
    copy!(state.force_densities, q)
    if size(state.variable_anchor_positions, 1) == size(anchors, 1)
        copy!(state.variable_anchor_positions, anchors)
    end
    final_loss = Optim.minimum(result)
    if isempty(state.loss_trace) || state.loss_trace[end] != final_loss
        push!(state.loss_trace, final_loss)
    end

    result, snapshot
end
