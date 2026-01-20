using JSON3
using SparseArrays
using LinearAlgebra

const DEFAULT_BARRIER_SHARPNESS = 10.0

abstract type AbstractObjective end

Base.@kwdef struct TargetXYZObjective <: AbstractObjective
    weight::Float64
    node_indices::Vector{Int}
    target::Matrix{Float64}
end

Base.@kwdef struct TargetXYObjective <: AbstractObjective
    weight::Float64
    node_indices::Vector{Int}
    target::Matrix{Float64}
end

Base.@kwdef struct TargetLengthObjective <: AbstractObjective
    weight::Float64
    edge_indices::Vector{Int}
    target::Vector{Float64}
end

Base.@kwdef struct LengthVariationObjective <: AbstractObjective
    weight::Float64
    edge_indices::Vector{Int}
end

Base.@kwdef struct ForceVariationObjective <: AbstractObjective
    weight::Float64
    edge_indices::Vector{Int}
end

Base.@kwdef struct SumForceLengthObjective <: AbstractObjective
    weight::Float64
    edge_indices::Vector{Int}
end

Base.@kwdef struct MinLengthObjective <: AbstractObjective
    weight::Float64
    edge_indices::Vector{Int}
    threshold::Vector{Float64}
    sharpness::Float64 = DEFAULT_BARRIER_SHARPNESS
end

Base.@kwdef struct MaxLengthObjective <: AbstractObjective
    weight::Float64
    edge_indices::Vector{Int}
    threshold::Vector{Float64}
    sharpness::Float64 = DEFAULT_BARRIER_SHARPNESS
end

Base.@kwdef struct MinForceObjective <: AbstractObjective
    weight::Float64
    edge_indices::Vector{Int}
    threshold::Vector{Float64}
    sharpness::Float64 = DEFAULT_BARRIER_SHARPNESS
end

Base.@kwdef struct MaxForceObjective <: AbstractObjective
    weight::Float64
    edge_indices::Vector{Int}
    threshold::Vector{Float64}
    sharpness::Float64 = DEFAULT_BARRIER_SHARPNESS
end

Base.@kwdef struct RigidSetCompareObjective <: AbstractObjective
    weight::Float64
    node_indices::Vector{Int}
    target::Matrix{Float64}
end

Base.@kwdef struct ReactionDirectionObjective <: AbstractObjective
    weight::Float64
    anchor_indices::Vector{Int}
    target_directions::Matrix{Float64}
end

Base.@kwdef struct ReactionDirectionMagnitudeObjective <: AbstractObjective
    weight::Float64
    anchor_indices::Vector{Int}
    target_directions::Matrix{Float64}
    target_magnitudes::Vector{Float64}
end

struct Bounds
    lower::Vector{Float64}
    upper::Vector{Float64}
end

struct SolverOptions
    absolute_tolerance::Float64
    relative_tolerance::Float64
    max_iterations::Int
    report_frequency::Int
    show_progress::Bool
    barrier_weight::Float64
    barrier_sharpness::Float64
    use_auto_scaling::Bool
end

struct TracingOptions
    record_nodes::Bool
    emit_frequency::Int
end

struct OptimizationParameters
    objectives::Vector{AbstractObjective}
    bounds::Bounds
    solver::SolverOptions
    tracing::TracingOptions
end

struct NetworkTopology
    incidence::SparseMatrixCSC{Int, Int}
    free_incidence::SparseMatrixCSC{Int, Int}
    fixed_incidence::SparseMatrixCSC{Int, Int}
    num_edges::Int
    num_nodes::Int
    free_node_indices::Vector{Int}
    fixed_node_indices::Vector{Int}
end

struct LoadData
    free_node_loads::Matrix{Float64}
end

struct GeometryData
    fixed_node_positions::Matrix{Float64}
end

Base.@kwdef struct AnchorInfo
    variable_indices::Vector{Int}
    fixed_indices::Vector{Int}
    reference_positions::Matrix{Float64}
    initial_variable_positions::Matrix{Float64}
end

AnchorInfo(reference_positions::Matrix{Float64}) = AnchorInfo(Int[], collect(1:size(reference_positions, 1)), reference_positions, zeros(0, 3))

struct OptimizationProblem
    topology::NetworkTopology
    loads::LoadData
    geometry::GeometryData
    anchors::AnchorInfo
    parameters::OptimizationParameters
end

mutable struct OptimizationState
    force_densities::Vector{Float64}
    variable_anchor_positions::Matrix{Float64}
    loss_trace::Vector{Float64}
    penalty_trace::Vector{Float64}
    node_trace::Vector{Matrix{Float64}}
    iterations::Int
end

OptimizationState(force_densities::Vector{Float64}, variable_anchor_positions::Matrix{Float64}) = OptimizationState(force_densities, variable_anchor_positions, Float64[], Float64[], Matrix{Float64}[], 0)

struct ObjectiveContext
    num_edges::Int
    num_nodes::Int
    free_node_indices::Vector{Int}
    fixed_node_indices::Vector{Int}
end

default_bounds(ne::Int) = Bounds(fill(1e-8, ne), fill(Inf, ne))

function current_fixed_positions(problem::OptimizationProblem, anchor_positions::AbstractMatrix{<:Real})
    anchor = problem.anchors
    ref = anchor.reference_positions
    T = promote_type(eltype(ref), eltype(anchor_positions))
    if isempty(anchor.variable_indices)
        return T === eltype(ref) ? copy(ref) : convert(Matrix{T}, ref)
    end
    positions = T === eltype(ref) ? copy(ref) : convert(Matrix{T}, ref)
    if !isempty(anchor.variable_indices)
        anchors_converted = eltype(anchor_positions) === T ? anchor_positions : convert(Matrix{T}, anchor_positions)
        positions[anchor.variable_indices, :] .= anchors_converted
    end
    return positions
end

current_fixed_positions(problem::OptimizationProblem, state::OptimizationState) =
    current_fixed_positions(problem, state.variable_anchor_positions)

function objective_context(problem::OptimizationProblem)
    topo = problem.topology
    ObjectiveContext(topo.num_edges, topo.num_nodes, topo.free_node_indices, topo.fixed_node_indices)
end

function parse_indices(obj_json, ctx::ObjectiveContext, objective_id::Int)
    if haskey(obj_json, "Indices")
        raw_indices = obj_json["Indices"]
        if !isempty(raw_indices) && raw_indices[1] == -1
            if objective_id == 1
                return copy(ctx.free_node_indices)
            elseif objective_id == 0
                return copy(ctx.fixed_node_indices)
            elseif objective_id in (12, 13)
                return copy(ctx.fixed_node_indices)
            else
                return collect(1:ctx.num_edges)
            end
        else
            return Int.(raw_indices) .+ 1
        end
    else
        return collect(1:ctx.num_edges)
    end
end

function parse_target_matrix(obj_json)
    values = haskey(obj_json, "Values") ? obj_json["Values"] : obj_json["Points"]
    matrix = reduce(hcat, values)
    convert(Matrix{Float64}, matrix')
end

function parse_direction_targets(obj_json, indices::Vector{Int})
    haskey(obj_json, "Vectors") || haskey(obj_json, "Points") || error("Objective is missing direction targets")
    values = haskey(obj_json, "Vectors") ? obj_json["Vectors"] : obj_json["Points"]

    vectors = if length(values) == 1 && length(indices) > 1
        vec = Float64.(values[1])
        hcat(ntuple(_ -> copy(vec), length(indices))...)
    else
        reduce(hcat, values)
    end

    matrix = convert(Matrix{Float64}, vectors')
    size(matrix, 1) == length(indices) || error("Number of direction targets must match indices or be 1")
    size(matrix, 2) == 3 || error("Direction targets must have exactly three components")
    matrix
end

function normalize_direction_rows(matrix::Matrix{Float64})
    normalized = similar(matrix)
    for row in 1:size(matrix, 1)
        vec = @view matrix[row, :]
        mag = norm(vec)
        mag > eps(Float64) || error("Direction target contains a zero vector")
        normalized[row, :] .= vec ./ mag
    end
    normalized
end

function parse_value_vector(obj_json, indices::Vector{Int}, ctx_length::Int)
    haskey(obj_json, "Values") || error("Objective is missing value data")
    values = Float64.(obj_json["Values"])
    if length(values) == length(indices)
        return values
    elseif length(values) == 1
        return fill(values[1], length(indices))
    elseif length(values) == ctx_length
        return values
    else
        return fill(values[1], length(indices))
    end
end

function build_objective(obj_json::JSON3.Object, ctx::ObjectiveContext)
    id = Int(obj_json.OBJID)
    weight = Float64(obj_json.Weight)
    indices = parse_indices(obj_json, ctx, id)
    if id == -1
        return nothing
    elseif id == 1
        target = parse_target_matrix(obj_json)
        return TargetXYZObjective(weight=weight, node_indices=indices, target=target)
    elseif id == 2
        return LengthVariationObjective(weight=weight, edge_indices=indices)
    elseif id == 3
        return ForceVariationObjective(weight=weight, edge_indices=indices)
    elseif id == 4
        return SumForceLengthObjective(weight=weight, edge_indices=indices)
    elseif id == 5
        values = parse_value_vector(obj_json, indices, ctx.num_edges)
        return MinLengthObjective(weight=weight, edge_indices=indices, threshold=values)
    elseif id == 6
        values = parse_value_vector(obj_json, indices, ctx.num_edges)
        return MaxLengthObjective(weight=weight, edge_indices=indices, threshold=values)
    elseif id == 7
        values = parse_value_vector(obj_json, indices, ctx.num_edges)
        return MinForceObjective(weight=weight, edge_indices=indices, threshold=values)
    elseif id == 8
        values = parse_value_vector(obj_json, indices, ctx.num_edges)
        return MaxForceObjective(weight=weight, edge_indices=indices, threshold=values)
    elseif id == 9
        values = parse_value_vector(obj_json, indices, ctx.num_edges)
        return TargetLengthObjective(weight=weight, edge_indices=indices, target=values)
    elseif id == 10
        target = parse_target_matrix(obj_json)
        return TargetXYObjective(weight=weight, node_indices=indices, target=target)
    elseif id == 11
        target = parse_target_matrix(obj_json)
        return RigidSetCompareObjective(weight=weight, node_indices=indices, target=target)
    elseif id == 12
        raw_targets = parse_direction_targets(obj_json, indices)
        directions = normalize_direction_rows(raw_targets)
        return ReactionDirectionObjective(weight=weight, anchor_indices=indices, target_directions=directions)
    elseif id == 13
        raw_targets = parse_direction_targets(obj_json, indices)
        directions = normalize_direction_rows(raw_targets)
        magnitudes = collect(map(norm, eachrow(raw_targets)))
        all(>(eps(Float64)), magnitudes) || error("Magnitude targets must be non-zero to define a direction")
        return ReactionDirectionMagnitudeObjective(
            weight=weight,
            anchor_indices=indices,
            target_directions=directions,
            target_magnitudes=magnitudes,
        )
    else
        error("Unsupported objective identifier: $id")
    end
end

function build_objectives(objectives_json, ctx::ObjectiveContext)
    objectives = AbstractObjective[]
    for entry in objectives_json
        candidate = build_objective(entry, ctx)
        if candidate !== nothing
            push!(objectives, candidate)
        end
    end
    return objectives
end

function parse_bounds(parameters_json::JSON3.Object, num_edges::Int)
    lower = haskey(parameters_json, "LowerBound") ? Float64.(parameters_json["LowerBound"]) : fill(1e-8, num_edges)
    upper = haskey(parameters_json, "UpperBound") ? Float64.(parameters_json["UpperBound"]) : fill(Inf, num_edges)
    if length(lower) == 1
        lower = fill(lower[1], num_edges)
    end
    if length(upper) == 1
        upper = fill(upper[1], num_edges)
    end
    Bounds(lower, upper)
end

function parse_solver_options(parameters_json::JSON3.Object)
    abs_tol = haskey(parameters_json, "AbsTol") ? Float64(parameters_json["AbsTol"]) : 1e-6
    rel_tol = haskey(parameters_json, "RelTol") ? Float64(parameters_json["RelTol"]) : 1e-6
    max_iter = haskey(parameters_json, "MaxIterations") ? Int(parameters_json["MaxIterations"]) : 500
    freq = haskey(parameters_json, "UpdateFrequency") ? Int(parameters_json["UpdateFrequency"]) : 1
    show = haskey(parameters_json, "ShowIterations") ? Bool(parameters_json["ShowIterations"]) : false
    
    barrier_weight = haskey(parameters_json, "BarrierWeight") ? Float64(parameters_json["BarrierWeight"]) : 1000.0
    barrier_sharpness = haskey(parameters_json, "BarrierSharpness") ? Float64(parameters_json["BarrierSharpness"]) : DEFAULT_BARRIER_SHARPNESS
    auto_scale = haskey(parameters_json, "AutoScale") ? Bool(parameters_json["AutoScale"]) : true
    
    SolverOptions(abs_tol, rel_tol, max_iter, freq, show, barrier_weight, barrier_sharpness, auto_scale)
end

function parse_tracing_options(parameters_json::JSON3.Object)
    record_nodes = haskey(parameters_json, "NodeTrace") ? Bool(parameters_json["NodeTrace"]) : false
    freq = haskey(parameters_json, "UpdateFrequency") ? Int(parameters_json["UpdateFrequency"]) : 1
    TracingOptions(record_nodes, freq)
end

function parse_anchor_info(problem::JSON3.Object, fixed_positions::Matrix{Float64})
    if !haskey(problem, "VariableAnchors")
        return AnchorInfo(fixed_positions)
    end

    anchors = problem["VariableAnchors"]
    variable_indices = haskey(problem, "NodeIndex") ? Int.(problem["NodeIndex"]) .+ 1 : Int[]
    fixed_indices = haskey(problem, "FixedAnchorIndices") ? Int.(problem["FixedAnchorIndices"]) .+ 1 : setdiff(collect(1:size(fixed_positions, 1)), variable_indices)

    init_positions = Matrix{Float64}(undef, length(variable_indices), 3)
    for (row, anchor) in enumerate(anchors)
        init_positions[row, 1] = Float64(anchor["InitialX"])
        init_positions[row, 2] = Float64(anchor["InitialY"])
        init_positions[row, 3] = Float64(anchor["InitialZ"])
    end

    AnchorInfo(variable_indices=variable_indices, fixed_indices=fixed_indices, reference_positions=fixed_positions, initial_variable_positions=init_positions)
end

function build_force_densities(values::Vector{Float64}, num_edges::Int)
    if length(values) == 1
        return fill(values[1], num_edges)
    elseif length(values) == num_edges
        return values
    else
        return fill(1.0, num_edges)
    end
end

function build_topology(problem::JSON3.Object)
    ne = Int(problem["Network"]["Graph"]["Ne"])
    nn = Int(problem["Network"]["Graph"]["Nn"])

    free_nodes = collect(1:length(problem["Network"]["FreeNodes"]))
    fixed_nodes = collect(length(free_nodes) + 1:length(free_nodes) + length(problem["Network"]["FixedNodes"]))

    i = Int.(problem["I"]) .+ 1
    j = Int.(problem["J"]) .+ 1
    v = Int.(problem["V"])

    incidence = sparse(i, j, v, ne, nn)
    free_incidence = incidence[:, 1:length(free_nodes)]
    fixed_incidence = incidence[:, length(free_nodes) + 1:end]

    NetworkTopology(incidence, free_incidence, fixed_incidence, ne, nn, free_nodes, fixed_nodes)
end

function build_geometry(problem::JSON3.Object)
    matrix = reduce(hcat, problem["XYZf"])
    GeometryData(convert(Matrix{Float64}, matrix'))
end

function build_loads(problem::JSON3.Object, topo::NetworkTopology)
    loads_matrix = reduce(hcat, problem["P"])
    loads = convert(Matrix{Float64}, loads_matrix')

    full_loads = zeros(topo.num_nodes, 3)
    load_nodes = haskey(problem, "LoadNodes") ? Int.(problem["LoadNodes"]) .+ 1 : Int[]

    if !isempty(load_nodes)
        if length(load_nodes) == size(loads, 1)
            full_loads[load_nodes, :] .= loads
        elseif size(loads, 1) == 1
            for node in load_nodes
                full_loads[node, :] .= loads[1, :]
            end
        end
    else
        if size(loads, 1) == 1
            for node in topo.free_node_indices
                full_loads[node, :] .= loads[1, :]
            end
        elseif size(loads, 1) == length(topo.free_node_indices)
            for (row, node) in enumerate(topo.free_node_indices)
                full_loads[node, :] .= loads[row, :]
            end
        end
    end

    LoadData(full_loads[topo.free_node_indices, :])
end

function build_parameters(problem::JSON3.Object, topo::NetworkTopology)
    if !haskey(problem, "Parameters")
        return OptimizationParameters(AbstractObjective[], default_bounds(topo.num_edges), SolverOptions(1e-6, 1e-6, 1, 1, false, 1000.0, DEFAULT_BARRIER_SHARPNESS, true), TracingOptions(false, 1))
    end

    params_json = problem["Parameters"]
    ctx = ObjectiveContext(topo.num_edges, topo.num_nodes, topo.free_node_indices, topo.fixed_node_indices)
    objectives = haskey(params_json, "Objectives") ? build_objectives(params_json["Objectives"], ctx) : AbstractObjective[]
    bounds = parse_bounds(params_json, topo.num_edges)
    solver = parse_solver_options(params_json)
    tracing = parse_tracing_options(params_json)

    OptimizationParameters(objectives, bounds, solver, tracing)
end

function build_problem(problem::JSON3.Object)
    topo = build_topology(problem)
    geometry = build_geometry(problem)
    loads = build_loads(problem, topo)

    anchors = parse_anchor_info(problem, geometry.fixed_node_positions)
    parameters = build_parameters(problem, topo)

    q_values = build_force_densities(Float64.(problem["Q"]), topo.num_edges)
    q_init = q_values

    problem_struct = OptimizationProblem(topo, loads, geometry, anchors, parameters)
    state = OptimizationState(q_init, anchors.initial_variable_positions)

    problem_struct, state
end