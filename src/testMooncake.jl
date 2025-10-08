#Minimal example demonstrating use of mooncake to minimize objective function. 
#Redevelopment of theseus will use this approach. 

using SparseArrays
using Mooncake
using Optim
import DifferentiationInterface as DI

```
Explicit solver function
```
function solve_explicit(
    qvec::Vector{Float64},           # Vector of force densities
    Cn::SparseMatrixCSC{Int64,Int64},# Index matrix of free nodes
    Cf::SparseMatrixCSC{Int64,Int64},# Index matrix of fixed nodes
    Pn::Matrix{Float64},             # Matrix of free node loads
    Nf::Matrix{Float64}              # Matrix of fixed node positions
)
    # Scale columns of Cn and Cf by qvec
    Cnq = Cn .* qvec
    Cfq = Cf .* qvec

    # Compute result using scaled matrices
    return (Cn' * Cnq) \ (Pn - Cn' * Cfq * Nf)
end

function obj(qvec, Cn, Cf, Pn, Nf, target)
    xyznew = solve_explicit(qvec, Cn, Cf, Pn, Nf)
    loss = sum((xyznew - target).^2)
    return loss
end

begin
    # Problem setup
    Cn = sparse([1, 2], [1, 1], [1, 1], 2, 1)
    Cf = sparse([1, 2], [1, 2], [-1, -1], 2, 2)
    Nf = [0.0 0.0 0.0; 2.0 0.0 0.0]
    Pn = reshape([0.0, 0.0, -1.0], 1, 3)
end

target_xyz = reshape([0.75, 0.0, -0.75], 1, 3)

"""
Mooncake setup
"""
backend = DI.AutoMooncake(; config=nothing)

qvec = [2.0, 2.0]  # Dense vector of force densities
prep = DI.prepare_gradient(
    obj,
    backend,
    qvec,
    DI.Constant(Cn),
    DI.Constant(Cf),
    DI.Constant(Pn),
    DI.Constant(Nf),
    DI.Constant(target_xyz)
)
grad = similar(qvec)

DI.gradient!(
    obj,
    grad,
    prep,
    backend,
    qvec,
    DI.Constant(Cn),
    DI.Constant(Cf),
    DI.Constant(Pn),
    DI.Constant(Nf),
    DI.Constant(target_xyz)
)

result = optimize(
    q -> obj(q, Cn, Cf, Pn, Nf, target_xyz),  # objective as closure
    [2.0, 2.0],                              # initial guess
    LBFGS()
)

println("Optimal qvec: ", Optim.minimizer(result))
println("Minimum loss: ", Optim.minimum(result))

# Final solve with optimal qvec
final_xyz = solve_explicit(
    Optim.minimizer(result),
    Cn,
    Cf,
    Pn,
    Nf
)
println("Final node location: ", final_xyz)



