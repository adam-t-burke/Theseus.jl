using JSON3
using SparseArrays
using LinearAlgebra
using TimerOutputs

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

Base.@kwdef struct SolverOptions
    absolute_tolerance::Float64
    relative_tolerance::Float64
    max_iterations::Int
    report_frequency::Int
    show_progress::Bool
    barrier_weight::Float64
    barrier_sharpness::Float64
    use_auto_scaling::Bool
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

struct GeometrySnapshot
    xyz_free::Matrix{Float64}
    xyz_fixed::Matrix{Float64}
    xyz_full::Matrix{Float64}
    member_lengths::Vector{Float64}
    member_forces::Vector{Float64}
    reactions::Matrix{Float64}
end

mutable struct FDMContext{TFact}
    # Matrices
    K::SparseMatrixCSC{Float64, Int64}
    factorization::TFact
    
    # Primal Buffers
    rhs::Matrix{Float64}
    xyz_free::Matrix{Float64}
    fixed_positions::Matrix{Float64}
    xyz_full::Matrix{Float64}
    
    member_vectors::Matrix{Float64}
    member_lengths::Vector{Float64}
    member_forces::Vector{Float64}
    reactions::Matrix{Float64}
    
    # Adjoint Buffers
    adj_xyz_full::Matrix{Float64}
    adj_xyz_free::Matrix{Float64}
    adj_q::Vector{Float64}
    adj_rhs::Matrix{Float64}
    adj_anchor_positions::Matrix{Float64}

    # Solver mode (true for cholesky, false for ldlt)
    is_chol::Bool
end

function FDMContext(topo::NetworkTopology, loads::LoadData, anchors::AnchorInfo)
    ne = topo.num_edges
    nn = topo.num_nodes
    nf = length(topo.free_node_indices)
    nx = length(topo.fixed_node_indices)
    dim = 3
    
    # K matrix pattern initialization
    q_init = ones(ne)
    K = topo.free_incidence' * (Diagonal(q_init) * topo.free_incidence)
    
    # Initial factorization (numerical doesn't matter much yet, but patterns do)
    # We use ldlt as baseline because it handles indefinite matrices
    factorization = ldlt(K)
    
    rhs = zeros(nf, dim)
    xyz_free = zeros(nf, dim)
    fixed_positions = zeros(nx, dim)
    xyz_full = zeros(nn, dim)
    
    member_vectors = zeros(ne, dim)
    member_lengths = zeros(ne)
    member_forces = zeros(ne)
    reactions = zeros(nn, dim)
    
    adj_xyz_full = zeros(nn, dim)
    adj_xyz_free = zeros(nf, dim)
    adj_q = zeros(ne)
    adj_rhs = zeros(nf, dim)
    nvar = length(anchors.variable_indices)
    adj_anchor_positions = zeros(nvar, dim)
    
    return FDMContext{typeof(factorization)}(
        K, factorization,
        rhs, xyz_free, fixed_positions, xyz_full,
        member_vectors, member_lengths, member_forces, reactions,
        adj_xyz_full, adj_xyz_free, adj_q, adj_rhs, adj_anchor_positions,
        false
    )
end

struct OptimizationProblem{TFact}
    topology::NetworkTopology
    loads::LoadData
    geometry::GeometryData
    anchors::AnchorInfo
    parameters::OptimizationParameters
    context::FDMContext{TFact}
end

struct GeometrySnapshot{TF, TX, TA, VL, VF, TR}
    xyz_free::TF
    xyz_fixed::TX
    xyz_full::TA
    member_lengths::VL
    member_forces::VF
    reactions::TR
end

mutable struct OptimizationCache
    # CPU Solver
    A::SparseMatrixCSC{Float64, Int64}
    factor::LDLFactorizations.LDLFactorization{Float64, Int64}
    q_to_nz::Vector{Vector{Tuple{Int, Float64}}} # maps edge index -> indices and coeffs in A.nzval
    edge_starts::Vector{Int} # node index (1..nn)
    edge_ends::Vector{Int}   # node index (1..nn)
    node_to_free_idx::Vector{Int} # node index (1..nn) -> index in Cn (0 if fixed)

    # Constant topology buffers
    Cn::SparseMatrixCSC{Float64, Int64}
    Cf::SparseMatrixCSC{Float64, Int64}

    # Buffers (primal)
    x::Matrix{Float64}
    λ::Matrix{Float64}
    grad_x::Matrix{Float64}
    q::Vector{Float64}
    grad_q::Vector{Float64}
    grad_Nf::Matrix{Float64} # Gradient w.r.t. fixed node positions
    
    # Primal result buffers
    member_lengths::Vector{Float64}
    member_forces::Vector{Float64}
    reactions::Matrix{Float64}

    # Intermediate buffers for RHS
    Cf_Nf::Matrix{Float64}
    Q_Cf_Nf::Matrix{Float64}
    Pn::Matrix{Float64}
    Nf::Matrix{Float64} # Buffer for current full node positions
    Nf_fixed::Matrix{Float64} # Dense buffer for fixed nodes only

    # Profiling and state
    to::TimerOutput
    last_snapshot::Union{Nothing, GeometrySnapshot}

    function OptimizationCache(problem::OptimizationProblem)
        topo = problem.topology
        Cn = convert(SparseMatrixCSC{Float64, Int64}, topo.free_incidence)
        Cf = convert(SparseMatrixCSC{Float64, Int64}, topo.fixed_incidence)
        ne = topo.num_edges
        nn_free = length(topo.free_node_indices)

        # 1. Construct sparsity pattern for A = Cn' * diag(q) * Cn
        A_template = sparse(Cn' * Cn)
        nz = nnz(A_template)
        A = SparseMatrixCSC{Float64, Int64}(A_template.m, A_template.n, A_template.colptr, A_template.rowval, zeros(nz))
        
        # Mapping from edge to nzval indices
        q_to_nz = [Tuple{Int, Float64}[] for _ in 1:ne]
        edge_starts = zeros(Int, ne)
        edge_ends = zeros(Int, ne)
        node_to_free_idx = zeros(Int, topo.num_nodes)
        for (i, node_idx) in enumerate(topo.free_node_indices)
            node_to_free_idx[node_idx] = i
        end

        # Pre-process incidence to get nodes of each edge
        incidence = problem.topology.incidence
        for col in 1:topo.num_nodes
            for idx in incidence.colptr[col]:(incidence.colptr[col+1]-1)
                row = incidence.rowval[idx]
                val = incidence.nzval[idx]
                if val == -1.0
                    edge_starts[row] = col
                elseif val == 1.0
                    edge_ends[row] = col
                end
            end
        end

        # Pre-process Cn to get nodes of each edge
        edge_to_nodes = [Int[] for _ in 1:ne]
        for j in 1:nn_free
            for idx in Cn.colptr[j]:(Cn.colptr[j+1]-1)
                push!(edge_to_nodes[Cn.rowval[idx]], j)
            end
        end

        for k in 1:ne
            nodes = edge_to_nodes[k]
            for n1 in nodes
                # find Cn[k, n1]
                val_n1 = 0.0
                for idx in Cn.colptr[n1]:(Cn.colptr[n1+1]-1)
                    if Cn.rowval[idx] == k; val_n1 = Cn.nzval[idx]; break; end
                end
                for n2 in nodes
                    # find Cn[k, n2]
                    val_n2 = 0.0
                    for idx in Cn.colptr[n2]:(Cn.colptr[n2+1]-1)
                        if Cn.rowval[idx] == k; val_n2 = Cn.nzval[idx]; break; end
                    end
                    nz_idx = find_nz_index(A, n1, n2)
                    push!(q_to_nz[k], (nz_idx, val_n1 * val_n2))
                end
            end
        end

        # 2. Initialize LDLFactorization
        # Fill diagonal to ensure initial symbolic/numeric pass works
        for i in 1:nn_free; A.nzval[find_nz_index(A, i, i)] = 1.0; end
        factor = LDLFactorizations.ldl(A)

        # 3. Pre-allocate buffers
        x = zeros(nn_free, 3)
        λ = zeros(nn_free, 3)
        grad_x = zeros(nn_free, 3)
        q = zeros(ne)
        grad_q = zeros(ne)
        grad_Nf = zeros(topo.num_nodes, 3)
        
        member_lengths = zeros(ne)
        member_forces = zeros(ne)
        reactions = zeros(topo.num_nodes, 3)

        Cf_Nf = zeros(ne, 3)
        Q_Cf_Nf = zeros(ne, 3)
        Pn = copy(problem.loads.free_node_loads)
        Nf = zeros(topo.num_nodes, 3)
        Nf_fixed = zeros(length(topo.fixed_node_indices), 3)

        to = TimerOutput()
        last_snapshot = nothing

        new(A, factor, q_to_nz, edge_starts, edge_ends, node_to_free_idx,
            Cn, Cf,
            x, λ, grad_x, q, grad_q, grad_Nf,
            member_lengths, member_forces, reactions,
            Cf_Nf, Q_Cf_Nf, Pn, Nf, Nf_fixed,
            to, last_snapshot)
    end
end

# Helper to find nz index in SparseMatrixCSC
function find_nz_index(A::SparseMatrixCSC, i::Integer, j::Integer)
    for nz_idx in A.colptr[j]:(A.colptr[j+1]-1)
        if A.rowval[nz_idx] == i
            return nz_idx
        end
    end
    error("Index ($i, $j) not found in sparse matrix")
end

mutable struct OptimizationState
    force_densities::Vector{Float64}
    variable_anchor_positions::Matrix{Float64}
    loss_trace::Vector{Float64}
    penalty_trace::Vector{Float64}
    node_trace::Vector{Matrix{Float64}}
    iterations::Int
    cache::Union{Nothing, OptimizationCache}
end

OptimizationState(force_densities::Vector{Float64}, variable_anchor_positions::Matrix{Float64}) = 
    OptimizationState(force_densities, variable_anchor_positions, Float64[], Float64[], Matrix{Float64}[], 0, nothing)

struct ObjectiveContext
    num_edges::Int
    num_nodes::Int
    free_node_indices::Vector{Int}
    fixed_node_indices::Vector{Int}
end

default_bounds(ne::Int) = Bounds(fill(1e-8, ne), fill(Inf, ne))

function current_fixed_positions!(dest::AbstractMatrix{T}, problem::OptimizationProblem, anchor_positions::AbstractMatrix{<:Real}) where T
    anchor = problem.anchors
    ref = anchor.reference_positions
    fixed_indices = problem.topology.fixed_node_indices
    
    # 1. Overlay reference positions of fixed nodes
    # We assume 'ref' contains positions for the nodes listed in 'fixed_indices'
    if size(ref, 1) == length(fixed_indices)
        for dim in 1:3
            for (i, node_idx) in enumerate(fixed_indices)
                dest[node_idx, dim] = ref[i, dim]
            end
        end
    elseif size(ref, 1) == problem.topology.num_nodes
        # Fallback if 'ref' is a full matrix of all nodes
        for dim in 1:3
            for node_idx in fixed_indices
                dest[node_idx, dim] = ref[node_idx, dim]
            end
        end
    else
        error("Dimension mismatch: reference_positions has $(size(ref, 1)) nodes but expected either $(length(fixed_indices)) (fixed only) or $(problem.topology.num_nodes) (all).")
    end
    
    # 2. Overlay variable anchors
    if !isempty(anchor.variable_indices)
        for dim in 1:3
            for (i, node_idx) in enumerate(anchor.variable_indices)
                dest[node_idx, dim] = anchor_positions[i, dim]
            end
        end
    end
    return dest
end

function current_fixed_positions(problem::OptimizationProblem, anchor_positions::AbstractMatrix{<:Real})
    dest = zeros(eltype(anchor_positions), problem.topology.num_nodes, 3)
    current_fixed_positions!(dest, problem, anchor_positions)
end

function update_fixed_positions!(ctx::FDMContext, anchor::AnchorInfo, variable_anchor_positions::AbstractMatrix{<:Real})
    ctx.fixed_positions .= anchor.reference_positions
    if !isempty(anchor.variable_indices)
        ctx.fixed_positions[anchor.variable_indices, :] .= variable_anchor_positions
    end
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

    context = FDMContext(topo, loads, anchors)

    problem_struct = OptimizationProblem(topo, loads, geometry, anchors, parameters, context)
    state = OptimizationState(q_init, anchors.initial_variable_positions)
    state.cache = OptimizationCache(problem_struct)

    problem_struct, state
end
