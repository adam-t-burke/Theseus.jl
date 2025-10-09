#######
# contains functions for analyzing an FDM network
#######

```
Explicit solver function
```
function solve_explicit(
    q::Vector{Float64}, #Vector of force densities
    Cn::SparseMatrixCSC{Int64,Int64}, #Index matrix of free nodes
    Cf::SparseMatrixCSC{Int64,Int64}, #Index matrix of fixed nodes
    Pn::Matrix{Float64}, #Matrix of free node loads
    Nf::Matrix{Float64}, #Matrix of fixed node positions
    )

    # Scale columns of Cn and Cf by q
    Cnq = Cn .* q
    Cfq = Cf .* q

    # Compute result using scaled matrices
    return (Cn' * Cnq) \ (Pn - Cn' * Cfq * Nf)
end

