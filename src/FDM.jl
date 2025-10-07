```
Explicit solver function
```
function solve_explicit(
    Q::SparseMatrixCSC{Float64, Int64}, #Vector of force densities
    Cn::SparseMatrixCSC{Int64,Int64}, #Index matrix of free nodes
    Cf::SparseMatrixCSC{Int64,Int64}, #Index matrix of fixed nodes
    Pn::Matrix{Float64}, #Matrix of free node loads
    Nf::Matrix{Float64}, #Matrix of fixed node positions
    )

    return (Cn' * Q * Cn) \ (Pn - Cn' * Q * Cf * Nf)
end

