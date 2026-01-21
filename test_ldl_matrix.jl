using LDLFactorizations
using SparseArrays
using LinearAlgebra

A = sparse(1:5, 1:5, ones(5))
F = ldl(A)
B = rand(5, 3)
X = zeros(5, 3)

try
    ldiv!(X, F, B)
    println("Matrix ldiv! check:")
    display(X)
    println("\nExpected:")
    display(B)
catch e
    println("Matrix ldiv! failed: ", e)
end
