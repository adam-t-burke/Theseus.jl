using LinearAlgebra
using SparseArrays
using ChainRulesCore
import Mooncake

Mooncake.zero_tangent(::SparseArrays.CHOLMOD.Factor) = Mooncake.NoTangent()
Mooncake.zero_tangent(::Ptr) = Mooncake.NoTangent()
Mooncake.zero_tangent(cache::FDMCache) = Mooncake.MutableTangent((;
    solver = Mooncake.NoTangent(),
    fixed_positions = zero(cache.fixed_positions),
))
Mooncake.zero_tangent(ctx::FDMContext) = Mooncake.Tangent((;
    cache = Mooncake.zero_tangent(ctx.cache),
    constants = Mooncake.NoTangent(),
))

ChainRulesCore.zero_tangent(::FDMCache) = ChainRulesCore.NoTangent()
ChainRulesCore.zero_tangent(::FDMContext) = ChainRulesCore.NoTangent()

function _solve_geometry!(
    workspace::FDMSolverWorkspace,
    q::AbstractVector{<:Real},
    loads::LoadData,
    fixed_positions::AbstractMatrix{<:Real},
)
    Cn_struct = workspace.Cn_struct
    Cn_numeric = workspace.Cn_numeric
    Cf_struct = workspace.Cf_struct
    Cf_numeric = workspace.Cf_numeric

    # Update numeric incidence entries with current force densities
    cn_rows = Cn_struct.rowval
    cn_base = workspace.base_Cn_nzval
    cn_vals = Cn_numeric.nzval
    @inbounds for idx in eachindex(cn_vals)
        cn_vals[idx] = cn_base[idx] * q[cn_rows[idx]]
    end

    cf_rows = Cf_struct.rowval
    cf_base = workspace.base_Cf_nzval
    cf_vals = Cf_numeric.nzval
    @inbounds for idx in eachindex(cf_vals)
        cf_vals[idx] = cf_base[idx] * q[cf_rows[idx]]
    end

    # Assemble the left-hand side pattern without allocating new storage
    lhs_vals = workspace.lhs_sparse.nzval
    fill!(lhs_vals, 0.0)
    pair_ptr = workspace.pair_ptr
    pair_k1 = workspace.pair_k1
    pair_k2 = workspace.pair_k2
    pair_lhs_idx = workspace.pair_lhs_idx

    @inbounds for edge in 1:(length(pair_ptr) - 1)
        start = pair_ptr[edge]
        stop = pair_ptr[edge + 1] - 1
        for cursor in start:stop
            lhs_idx = pair_lhs_idx[cursor]
            lhs_vals[lhs_idx] += cn_base[pair_k1[cursor]] * cn_vals[pair_k2[cursor]]
        end
    end

    @inbounds for idx in workspace.lhs_diag_indices
        lhs_vals[idx] += workspace.reg_eps
    end

    # Form the right-hand side using sparse-dense multiplies only
    mul!(workspace.CfNf, Cf_numeric, fixed_positions)
    mul!(workspace.CnT_qCfNf, transpose(Cn_struct), workspace.CfNf)

    rhs = workspace.rhs
    copy!(rhs, loads.free_node_loads)
    ndims = workspace.ndims
    nf = size(rhs, 1)
    @inbounds for col in 1:ndims, row in 1:nf
        rhs[row, col] -= workspace.CnT_qCfNf[row, col]
    end

    # Factorize with cached CHOLMOD analysis and solve for all load cases
    lhs_cholmod = SparseArrays.CHOLMOD.Sparse(workspace.lhs_sparse)
    factor = workspace.factor
    if factor === nothing
        factor = SparseArrays.CHOLMOD.analyze(lhs_cholmod)
        workspace.factor = factor
    end
    SparseArrays.CHOLMOD.factorize!(lhs_cholmod, factor)
    workspace.factor_valid = true

    sol_dense = SparseArrays.CHOLMOD.solve(0, factor, SparseArrays.CHOLMOD.Dense(rhs))
    copy!(workspace.solution, Array(sol_dense))

    workspace.solution
end

function solve_geometry(cache::FDMCache, q::AbstractVector{<:Real}, loads::LoadData)
    _solve_geometry!(cache.solver, q, loads, cache.fixed_positions)
end

solve_geometry(ctx::FDMContext, q::AbstractVector{<:Real}, loads::LoadData) =
    solve_geometry(ctx.cache, q, loads)

function solve_explicit(
    workspace::FDMSolverWorkspace,
    q::AbstractVector{<:Real},
    loads::LoadData,
    fixed_positions::AbstractMatrix{<:Real},
)
    _solve_geometry!(workspace, q, loads, fixed_positions)
end

function solve_explicit!(
    dest::Matrix{Float64},
    workspace::FDMSolverWorkspace,
    q::AbstractVector{<:Real},
    loads::LoadData,
    fixed_positions::AbstractMatrix{<:Real},
)
    sol = _solve_geometry!(workspace, q, loads, fixed_positions)
    dest .= sol
    dest
end

function _solve_geometry_pullback!(
    workspace::FDMSolverWorkspace,
    q::AbstractVector{<:Real},
    loads::LoadData,
    fixed_positions::AbstractMatrix{<:Real},
    upstream::AbstractMatrix{<:Real},
)
    lambda = workspace.adjoint_rhs
    copy!(lambda, upstream)

    factor = workspace.factor
    factor === nothing && error("solve_explicit pullback requires prior factorization")
    adjoint_dense = SparseArrays.CHOLMOD.solve(1, factor, SparseArrays.CHOLMOD.Dense(lambda))
    copy!(lambda, Array(adjoint_dense))

    Cn_struct = workspace.Cn_struct
    Cf_numeric = workspace.Cf_numeric
    edges_lambda = workspace.qCfNf
    mul!(edges_lambda, Cn_struct, lambda)

    grad_fixed = workspace.grad_fixed
    mul!(grad_fixed, transpose(Cf_numeric), edges_lambda)
    @. grad_fixed = -grad_fixed

    edges_x = workspace.CfNf
    mul!(edges_x, Cn_struct, workspace.solution)

    grad_q = workspace.grad_q
    fill!(grad_q, 0.0)

    edge_ptr = workspace.edge_ptr
    cf_ptr = workspace.cf_edge_ptr
    cf_indices = workspace.cf_edge_indices
    cf_cols = workspace.cf_edge_cols
    base_cf = workspace.base_Cf_nzval
    ndims = workspace.ndims
    ne = size(Cn_struct, 1)

    @inbounds for edge in 1:ne
        lam_row = @view edges_lambda[edge, :]
        x_row = @view edges_x[edge, :]
        dotK = dot(lam_row, x_row)
        dot_rhs = 0.0
        start_cf = cf_ptr[edge]
        stop_cf = cf_ptr[edge + 1] - 1
        for offset in start_cf:stop_cf
            nz_idx = cf_indices[offset]
            col = cf_cols[offset]
            coeff = base_cf[nz_idx]
            fixed_row = @view fixed_positions[col, :]
            for dim in 1:ndims
                dot_rhs += lam_row[dim] * coeff * fixed_row[dim]
            end
        end
        grad_q[edge] = -dotK - dot_rhs
    end

    grad_q, lambda, grad_fixed
end

function solve_geometry_pullback!(
    cache::FDMCache,
    q::AbstractVector{<:Real},
    loads::LoadData,
    upstream::AbstractMatrix{<:Real},
)
    _solve_geometry_pullback!(cache.solver, q, loads, cache.fixed_positions, upstream)
end

solve_geometry_pullback!(
    ctx::FDMContext,
    q::AbstractVector{<:Real},
    loads::LoadData,
    upstream::AbstractMatrix{<:Real},
) = solve_geometry_pullback!(ctx.cache, q, loads, upstream)

function solve_explicit_pullback!(
    workspace::FDMSolverWorkspace,
    q::AbstractVector{<:Real},
    loads::LoadData,
    fixed_positions::AbstractMatrix{<:Real},
    upstream::AbstractMatrix{<:Real},
)
    _solve_geometry_pullback!(workspace, q, loads, fixed_positions, upstream)
end

function ChainRulesCore.rrule(
    ::typeof(solve_geometry),
    ctx::FDMContext,
    q::AbstractVector{<:Real},
    loads::LoadData,
)
    primal = solve_geometry(ctx, q, loads)

    function pullback(ȳ)
        if ChainRulesCore.iszero(ȳ)
            return (
                ChainRulesCore.NoTangent(),
                ChainRulesCore.NoTangent(),
                ChainRulesCore.ZeroTangent(),
                ChainRulesCore.ZeroTangent(),
            )
        end

        upstream = ChainRulesCore.unthunk(ȳ)
        grad_q, load_grad_matrix, _ = solve_geometry_pullback!(ctx, q, loads, upstream)
        load_grad = ChainRulesCore.Tangent{LoadData}(; free_node_loads = load_grad_matrix)

        (
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            grad_q,
            load_grad,
        )
    end

    primal, pullback
end

function Mooncake.rrule!!(
    ::Mooncake.CoDual{typeof(solve_geometry)},
    ctx_cd::Mooncake.CoDual{FDMContext},
    q_cd::Mooncake.CoDual{<:AbstractVector{Float64}},
    loads_cd::Mooncake.CoDual{LoadData},
)
    ctx = Mooncake.primal(ctx_cd)
    q = Mooncake.primal(q_cd)
    loads = Mooncake.primal(loads_cd)

    primal_output = solve_geometry(ctx, q, loads)
    output = Mooncake.CoDual(primal_output, Mooncake.zero_tangent(primal_output))

    function solve_geometry_pullback!!(ȳ)
        upstream = ȳ isa Mooncake.NoRData ? zero(primal_output) : ȳ
        grad_q, load_grad_matrix, _ = solve_geometry_pullback!(ctx, q, loads, upstream)

        load_tangent = Mooncake.Tangent((; free_node_loads = copy(load_grad_matrix)))
        (
            Mooncake.NoRData(),
            Mooncake.NoRData(),
            copy(grad_q),
            load_tangent,
        )
    end

    output, solve_geometry_pullback!!
end

function ChainRulesCore.rrule(
    ::typeof(solve_explicit),
    workspace::FDMSolverWorkspace,
    q::Vector{Float64},
    loads::LoadData,
    fixed_positions::Matrix{Float64},
)
    primal = solve_explicit(workspace, q, loads, fixed_positions)

    function pullback(ȳ)
        if ChainRulesCore.iszero(ȳ)
            return (
                ChainRulesCore.NoTangent(),
                ChainRulesCore.NoTangent(),
                ChainRulesCore.ZeroTangent(),
                ChainRulesCore.ZeroTangent(),
                ChainRulesCore.ZeroTangent(),
            )
        end

        upstream = ChainRulesCore.unthunk(ȳ)
        grad_q, load_grad_matrix, grad_fixed =
            solve_explicit_pullback!(workspace, q, loads, fixed_positions, upstream)
        load_grad = ChainRulesCore.Tangent{LoadData}(; free_node_loads=load_grad_matrix)

        (
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            grad_q,
            load_grad,
            grad_fixed,
        )
    end

    primal, pullback
end


function Mooncake.rrule!!(
    ::Mooncake.CoDual{typeof(solve_explicit)},
    workspace_cd::Mooncake.CoDual{FDMSolverWorkspace},
    q_cd::Mooncake.CoDual{<:AbstractVector{Float64}},
    loads_cd::Mooncake.CoDual{LoadData},
    fixed_cd::Mooncake.CoDual{<:AbstractMatrix{Float64}},
)
    workspace = Mooncake.primal(workspace_cd)
    q = Mooncake.primal(q_cd)
    loads = Mooncake.primal(loads_cd)
    fixed_positions = Mooncake.primal(fixed_cd)

    primal_output = solve_explicit(workspace, q, loads, fixed_positions)
    output = Mooncake.CoDual(primal_output, Mooncake.zero_tangent(primal_output))

    function solve_explicit_pullback!!(ȳ)
        upstream = ȳ isa Mooncake.NoRData ? zero(primal_output) : ȳ
        grad_q, load_grad_matrix, grad_fixed =
            solve_explicit_pullback!(workspace, q, loads, fixed_positions, upstream)

        load_tangent = Mooncake.Tangent((; free_node_loads = copy(load_grad_matrix)))
        (
            Mooncake.NoRData(),
            Mooncake.NoRData(),
            copy(grad_q),
            load_tangent,
            copy(grad_fixed),
        )
    end

    output, solve_explicit_pullback!!
end


