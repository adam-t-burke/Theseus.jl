#######
# contains functions for analyzing an FDM network
#######

```
Explicit solver function
```
function solve_explicit(
    q::AbstractVector{<:Real}, #Vector of force densities
    Cn::SparseMatrixCSC{Int64,Int64}, #Index matrix of free nodes
    Cf::SparseMatrixCSC{Int64,Int64}, #Index matrix of fixed nodes
    Pn::AbstractMatrix{<:Real}, #Matrix of free node loads
    Nf::AbstractMatrix{<:Real}, #Matrix of fixed node positions
    )

    # Scale columns of Cn and Cf by q
    Cnq = Cn .* q
    Cfq = Cf .* q

    # Compute result using scaled matrices
    return (Cn' * Cnq) \ (Pn - Cn' * Cfq * Nf)
end

function update_factorization!(ctx::FDMContext{T}) where {T}
    try
        cholesky!(ctx.factorization, ctx.K)
        ctx.is_chol = true
    catch
        ldlt!(ctx.factorization, ctx.K)
        ctx.is_chol = false
    end
end

function solve_explicit!(
    ctx::FDMContext{T},
    q::AbstractVector{<:Real},
    Cn::SparseMatrixCSC{Int, Int},
    Cf::SparseMatrixCSC{Int, Int},
    Pn::AbstractMatrix{<:Real},
    Nf::AbstractMatrix{<:Real}
    ) where {T}
    
    # Update K
    # Using a slightly allocating but efficient update for now
    ctx.K = Cn' * (Diagonal(q) * Cn)
    
    # Update Factorization
    update_factorization!(ctx)
    
    # Form RHS
    # (Diagonal(q) * Cf) * Nf scales rows of Cf by q, then multiplies by Nf
    ctx.rhs .= Pn .- Cn' * (Diagonal(q) * (Cf * Nf))
    
    # In-place solve
    ldiv!(ctx.xyz_free, ctx.factorization, ctx.rhs)
    
    return ctx.xyz_free
end

