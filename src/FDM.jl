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
    sp_init::Vector{Int64} #Intialization for sparse matrix
    )

    Q = sparse(sp_init,sp_init,q) # build diagonal force density Matrix

    Qcn = 
    Qcf = 
    
    return (Cn' * Q * Cn) \ (Pn - Cn' * Q * Cf * Nf)
end

