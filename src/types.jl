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

Base.@kwdef struct Bounds
    lower::Vector{Float64}
    upper::Vector{Float64}
end

Base.@kwdef struct SolverOptions
    absolute_tolerance::Float64
    relative_tolerance::Float64
    max_iterations::Int
    report_frequency::Int
    show_progress::Bool
end

Base.@kwdef struct TracingOptions
    record_nodes::Bool
    emit_frequency::Int
end

Base.@kwdef struct OptimizationParameters
    objectives::Vector{AbstractObjective}
    bounds::Bounds
    solver::SolverOptions
    tracing::TracingOptions
end

Base.@kwdef struct NetworkTopology
    incidence::SparseMatrixCSC{Int, Int}
    free_incidence::SparseMatrixCSC{Int, Int}
    fixed_incidence::SparseMatrixCSC{Int, Int}
    num_edges::Int
    num_nodes::Int
    free_node_indices::Vector{Int}
    fixed_node_indices::Vector{Int}
end

Base.@kwdef struct LoadData
    free_node_loads::Matrix{Float64}
end

Base.@kwdef struct GeometryData
    fixed_node_positions::Matrix{Float64}
end

Base.@kwdef struct GeometrySnapshot
    xyz_free::Matrix{Float64}
    xyz_fixed::Matrix{Float64}
    xyz_full::Matrix{Float64}
    member_lengths::Vector{Float64}
    member_forces::Vector{Float64}
end

mutable struct FDMSolverWorkspace
    # Structural incidence matrices (integer valued, never mutated)
    Cn_struct::SparseMatrixCSC{Int64,Int64}
    Cf_struct::SparseMatrixCSC{Int64,Int64}

    # Numeric copies with identical sparsity used for per-iteration scaling
    Cn_numeric::SparseMatrixCSC{Float64,Int64}
    Cf_numeric::SparseMatrixCSC{Float64,Int64}

    # Preserved baseline nzval vectors (Float64 conversion of structural signs)
    base_Cn_nzval::Vector{Float64}
    base_Cf_nzval::Vector{Float64}

    # Edge -> incidence lookup (CSR-style pointers into structural nzval indices)
    edge_ptr::Vector{Int}
    edge_indices::Vector{Int}
    edge_cols::Vector{Int}

    # Fixed-incidence lookup for rhs sensitivities
    cf_edge_ptr::Vector{Int}
    cf_edge_indices::Vector{Int}
    cf_edge_cols::Vector{Int}

    # Edge pair -> lhs contribution lookup
    pair_ptr::Vector{Int}
    pair_k1::Vector{Int}
    pair_k2::Vector{Int}
    pair_lhs_idx::Vector{Int}

    # Sparse system matrix pattern and helpers
    lhs_sparse::SparseMatrixCSC{Float64,Int64}
    lhs_diag_indices::Vector{Int}
    factor::Union{SparseArrays.CHOLMOD.Factor{Float64},Nothing}
    factor_valid::Bool

    # Dense work buffers (reuse across solves)
    CfNf::Matrix{Float64}
    qCfNf::Matrix{Float64}
    CnT_qCfNf::Matrix{Float64}
    rhs::Matrix{Float64}
    solution::Matrix{Float64}
    adjoint_rhs::Matrix{Float64}
    grad_q::Vector{Float64}
    grad_fixed::Matrix{Float64}

    ndims::Int
    reg_eps::Float64

    # Legacy dense copies retained temporarily for compatibility during refactor
    Cn_dense::Matrix{Float64}
    Cf_dense::Matrix{Float64}
    Cnq_dense::Matrix{Float64}
    Cfq_dense::Matrix{Float64}
    lhs_dense::Matrix{Float64}
end

@inline pairkey(row::Int, col::Int) = (Int64(row) << 32) | Int64(col)

function build_edge_lookup(mat::SparseMatrixCSC{<:Any,Int64})
    ne = size(mat, 1)
    counts = zeros(Int, ne)
    for col in 1:size(mat, 2)
        col_range = mat.colptr[col]:(mat.colptr[col + 1] - 1)
        @inbounds for idx in col_range
            edge = mat.rowval[idx]
            counts[edge] += 1
        end
    end
    ptr = Vector{Int}(undef, ne + 1)
    ptr[1] = 1
    for edge in 1:ne
        ptr[edge + 1] = ptr[edge] + counts[edge]
    end
    total = ptr[end] - 1
    indices = Vector{Int}(undef, total)
    cols = Vector{Int}(undef, total)
    fill!(counts, 0)
    for col in 1:size(mat, 2)
        col_range = mat.colptr[col]:(mat.colptr[col + 1] - 1)
        @inbounds for idx in col_range
            edge = mat.rowval[idx]
            offset = counts[edge]
            position = ptr[edge] + offset
            indices[position] = idx
            cols[position] = col
            counts[edge] = offset + 1
        end
    end
    ptr, indices, cols
end

function build_edge_pairs(edge_ptr::Vector{Int}, edge_indices::Vector{Int}, edge_cols::Vector{Int}, lhs::SparseMatrixCSC{Float64,Int64})
    ne = length(edge_ptr) - 1
    total_pairs = 0
    for edge in 1:ne
        len = edge_ptr[edge + 1] - edge_ptr[edge]
        total_pairs += len * len
    end
    pair_ptr = Vector{Int}(undef, ne + 1)
    pair_ptr[1] = 1
    accum = 1
    for edge in 1:ne
        len = edge_ptr[edge + 1] - edge_ptr[edge]
        accum += len * len
        pair_ptr[edge + 1] = accum
    end
    total = pair_ptr[end] - 1
    pair_k1 = Vector{Int}(undef, total)
    pair_k2 = Vector{Int}(undef, total)
    pair_lhs_idx = Vector{Int}(undef, total)

    lookup = Dict{Int64, Int}()
    for col in 1:size(lhs, 2)
        col_range = lhs.colptr[col]:(lhs.colptr[col + 1] - 1)
        @inbounds for idx in col_range
            row = lhs.rowval[idx]
            lookup[pairkey(row, col)] = idx
        end
    end

    cursor = 1
    for edge in 1:ne
        start = edge_ptr[edge]
        stop = edge_ptr[edge + 1] - 1
        @inbounds for offset_i in start:stop
            k1 = edge_indices[offset_i]
            col_i = edge_cols[offset_i]
            for offset_j in start:stop
                k2 = edge_indices[offset_j]
                col_j = edge_cols[offset_j]
                pair_k1[cursor] = k1
                pair_k2[cursor] = k2
                pair_lhs_idx[cursor] = lookup[pairkey(col_i, col_j)]
                cursor += 1
            end
        end
    end

    pair_ptr, pair_k1, pair_k2, pair_lhs_idx
end

function compute_lhs_diag_indices(lhs::SparseMatrixCSC{Float64,Int64})
    n = size(lhs, 1)
    diag = Vector{Int}(undef, n)
    @inbounds for col in 1:n
        diag_idx = 0
        for idx in lhs.colptr[col]:(lhs.colptr[col + 1] - 1)
            if lhs.rowval[idx] == col
                diag_idx = idx
                break
            end
        end
        diag[col] = diag_idx
    end
    diag
end

struct FDMConstantData
    topology::NetworkTopology
    ndims::Int
    reg_eps::Float64
end

mutable struct FDMCache
    solver::FDMSolverWorkspace
    fixed_positions::Matrix{Float64}
end

struct FDMContext
    cache::FDMCache
    constants::FDMConstantData
end

mutable struct GeometryWorkspace
    fdm::FDMContext
    xyz_free::Matrix{Float64}
    xyz_full::Matrix{Float64}
    member_vectors::Matrix{Float64}
    member_lengths::Vector{Float64}
    member_forces::Vector{Float64}
    q_buffer::Vector{Float64}
    anchor_buffer::Matrix{Float64}
    theta_lower::Vector{Float64}
    theta_upper::Vector{Float64}
    snapshot::GeometrySnapshot
end

function Base.getproperty(workspace::GeometryWorkspace, name::Symbol)
    if name === :solver
        return getfield(getfield(workspace, :fdm), :cache).solver
    elseif name === :fixed_positions
        return getfield(getfield(workspace, :fdm), :cache).fixed_positions
    elseif name === :fdm
        return getfield(workspace, :fdm)
    else
        return getfield(workspace, name)
    end
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
    node_trace::Vector{Matrix{Float64}}
    iterations::Int
    workspace::GeometryWorkspace
end

OptimizationState(force_densities::Vector{Float64}, variable_anchor_positions::Matrix{Float64}, workspace::GeometryWorkspace) =
    OptimizationState(force_densities, variable_anchor_positions, Float64[], Matrix{Float64}[], 0, workspace)

struct ObjectiveContext
    num_edges::Int
    num_nodes::Int
    free_node_indices::Vector{Int}
    fixed_node_indices::Vector{Int}
end

default_bounds(ne::Int) = Bounds(fill(-Inf, ne), fill(Inf, ne))

function current_fixed_positions!(dest::Matrix{Float64}, problem::OptimizationProblem, anchor_positions::Matrix{Float64})
    anchor = problem.anchors
    copy!(dest, anchor.reference_positions)
    if isempty(anchor.variable_indices)
        return dest
    end
    @inbounds for (local_idx, node_idx) in enumerate(anchor.variable_indices)
        dest[node_idx, 1] = anchor_positions[local_idx, 1]
        dest[node_idx, 2] = anchor_positions[local_idx, 2]
        dest[node_idx, 3] = anchor_positions[local_idx, 3]
    end
    dest
end

function current_fixed_positions(problem::OptimizationProblem, anchor_positions::Matrix{Float64})
    dest = copy(problem.anchors.reference_positions)
    current_fixed_positions!(dest, problem, anchor_positions)
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
    id = Int(obj_json["OBJID"])
    weight = Float64(obj_json["Weight"])
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
    lower = haskey(parameters_json, "LowerBound") ? Float64.(parameters_json["LowerBound"]) : fill(-Inf, num_edges)
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
    SolverOptions(abs_tol, rel_tol, max_iter, freq, show)
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
    if isempty(anchors)
        return AnchorInfo(fixed_positions)
    end

    variable_indices = if haskey(problem, "NodeIndex")
        Int.(problem["NodeIndex"]) .+ 1
    else
        [Int(anchor["NodeIndex"]) + 1 for anchor in anchors]
    end

    nvar = length(variable_indices)
    init_positions = Matrix{Float64}(undef, nvar, 3)
    for row in 1:nvar
        idx = variable_indices[row]
        anchor = anchors[row]
        ref = fixed_positions[idx, :]
        init_positions[row, 1] = haskey(anchor, "InitialX") ? Float64(anchor["InitialX"]) : ref[1]
        init_positions[row, 2] = haskey(anchor, "InitialY") ? Float64(anchor["InitialY"]) : ref[2]
        init_positions[row, 3] = haskey(anchor, "InitialZ") ? Float64(anchor["InitialZ"]) : ref[3]
    end

    fixed_indices = haskey(problem, "FixedAnchorIndices") ? Int.(problem["FixedAnchorIndices"]) .+ 1 : setdiff(collect(1:size(fixed_positions, 1)), variable_indices)

    AnchorInfo(variable_indices=variable_indices, fixed_indices=fixed_indices, reference_positions=fixed_positions, initial_variable_positions=init_positions)
end

function repeat_force_densities(values::Vector{Float64}, num_edges::Int)
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
        return OptimizationParameters(AbstractObjective[], default_bounds(topo.num_edges), SolverOptions(1e-6, 1e-6, 1, 1, false), TracingOptions(false, 1))
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

    q_values = repeat_force_densities(Float64.(problem["Q"]), topo.num_edges)
    q_init = clamp.(q_values, parameters.bounds.lower, parameters.bounds.upper)

    problem_struct = OptimizationProblem(topo, loads, geometry, anchors, parameters)
    workspace = create_geometry_workspace(problem_struct)
    state = OptimizationState(copy(q_init), copy(anchors.initial_variable_positions), workspace)

    problem_struct, state
end

function create_fdmsolver_workspace(topo::NetworkTopology; ndims::Int = 3, reg_eps::Float64 = 1e-12)
    Cn_struct = convert(SparseMatrixCSC{Int64,Int64}, topo.free_incidence)
    Cf_struct = convert(SparseMatrixCSC{Int64,Int64}, topo.fixed_incidence)

    Cn_numeric = SparseMatrixCSC{Float64,Int64}(Cn_struct)
    Cf_numeric = SparseMatrixCSC{Float64,Int64}(Cf_struct)

    base_Cn_nzval = Float64.(Cn_struct.nzval)
    base_Cf_nzval = Float64.(Cf_struct.nzval)
    Cn_numeric.nzval .= base_Cn_nzval
    Cf_numeric.nzval .= base_Cf_nzval

    edge_ptr, edge_indices, edge_cols = build_edge_lookup(Cn_struct)
    cf_edge_ptr, cf_edge_indices, cf_edge_cols = build_edge_lookup(Cf_struct)

    lhs_pattern = SparseMatrixCSC{Float64,Int64}(transpose(Cn_struct) * Cn_struct)
    pair_ptr, pair_k1, pair_k2, pair_lhs_idx = build_edge_pairs(edge_ptr, edge_indices, edge_cols, lhs_pattern)
    lhs_diag_indices = compute_lhs_diag_indices(lhs_pattern)

    ne = topo.num_edges
    nf = size(Cn_struct, 2)
    nfixed = size(Cf_struct, 2)

    CfNf = Matrix{Float64}(undef, ne, ndims)
    qCfNf = Matrix{Float64}(undef, ne, ndims)
    CnT_qCfNf = Matrix{Float64}(undef, nf, ndims)
    rhs = Matrix{Float64}(undef, nf, ndims)
    solution = Matrix{Float64}(undef, nf, ndims)
    adjoint_rhs = Matrix{Float64}(undef, nf, ndims)
    grad_q = Vector{Float64}(undef, ne)
    grad_fixed = Matrix{Float64}(undef, nfixed, ndims)

    Cn_dense = Matrix{Float64}(Cn_struct)
    Cf_dense = Matrix{Float64}(Cf_struct)
    Cnq_dense = similar(Cn_dense)
    Cfq_dense = similar(Cf_dense)
    lhs_dense = Matrix{Float64}(undef, nf, nf)

    FDMSolverWorkspace(
        Cn_struct,
        Cf_struct,
        Cn_numeric,
        Cf_numeric,
        base_Cn_nzval,
        base_Cf_nzval,
        edge_ptr,
        edge_indices,
        edge_cols,
    cf_edge_ptr,
    cf_edge_indices,
    cf_edge_cols,
        pair_ptr,
        pair_k1,
        pair_k2,
        pair_lhs_idx,
        lhs_pattern,
        lhs_diag_indices,
        nothing,
        false,
        CfNf,
        qCfNf,
        CnT_qCfNf,
        rhs,
        solution,
        adjoint_rhs,
        grad_q,
    grad_fixed,
        ndims,
        reg_eps,
        Cn_dense,
        Cf_dense,
        Cnq_dense,
        Cfq_dense,
        lhs_dense,
    )
end

function create_geometry_workspace(problem::OptimizationProblem)
    topo = problem.topology
    ndims = size(problem.geometry.fixed_node_positions, 2)
    solver = create_fdmsolver_workspace(topo; ndims=ndims)
    constants = FDMConstantData(topo, ndims, solver.reg_eps)
    fixed_positions = copy(problem.geometry.fixed_node_positions)
    cache = FDMCache(solver, fixed_positions)
    context = FDMContext(cache, constants)

    xyz_free = cache.solver.solution
    xyz_full = Matrix{Float64}(undef, topo.num_nodes, 3)
    member_vectors = Matrix{Float64}(undef, topo.num_edges, 3)
    member_lengths = Vector{Float64}(undef, topo.num_edges)
    member_forces = Vector{Float64}(undef, topo.num_edges)
    q_buffer = Vector{Float64}(undef, topo.num_edges)
    nvar = size(problem.anchors.initial_variable_positions, 1)
    anchor_buffer = Matrix{Float64}(undef, nvar, 3)
    θ_len = topo.num_edges + 3 * nvar
    theta_lower = Vector{Float64}(undef, θ_len)
    theta_upper = Vector{Float64}(undef, θ_len)
    if θ_len > 0
        copyto!(theta_lower, 1, problem.parameters.bounds.lower, 1, topo.num_edges)
        copyto!(theta_upper, 1, problem.parameters.bounds.upper, 1, topo.num_edges)
        if θ_len > topo.num_edges
            start_idx = topo.num_edges + 1
            @inbounds for i in start_idx:θ_len
                theta_lower[i] = -Inf
                theta_upper[i] = Inf
            end
        end
    end

    snapshot = GeometrySnapshot(xyz_free, cache.fixed_positions, xyz_full, member_lengths, member_forces)
    GeometryWorkspace(context, xyz_free, xyz_full, member_vectors, member_lengths, member_forces, q_buffer, anchor_buffer, theta_lower, theta_upper, snapshot)
end
