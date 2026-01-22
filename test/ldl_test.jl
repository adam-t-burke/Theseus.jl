using SparseArrays
using LinearAlgebra
using LDLFactorizations
using BenchmarkTools

function test_ldl(N)
    # Create SPD matrix
    I = Int[]
    J = Int[]
    V = Float64[]
    for i in 1:N
        push!(I, i); push!(J, i); push!(V, 2.0)
        if i < N
            push!(I, i); push!(J, i+1); push!(V, -1.0)
            push!(I, i+1); push!(J, i); push!(V, -1.0)
        end
    end
    A = sparse(I, J, V)
    b = randn(N, 3)
    x = zeros(N, 3)
    
    # Init
    F = ldlt(A)
    
    # Warmup
    ldlt!(F, A)
    ldiv!(x, F, b)
    
    allocs = @allocated begin
        ldlt!(F, A)
        ldiv!(x, F, b)
    end
    
    println("N=$N LDLFactorizations Allocs: $allocs bytes")
end

test_ldl(10)
test_ldl(1000)
test_ldl(10000)
