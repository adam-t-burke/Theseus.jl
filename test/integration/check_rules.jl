using Mooncake
using Theseus
using SparseArrays

println("Checking update_factorization! rule with debug_mode=true...")
try
    # Note: Using the exact signature suggested by the error message
    rule = Mooncake.build_rrule(Mooncake.MooncakeInterpreter(Mooncake.ReverseMode), Tuple{typeof(Theseus.update_factorization!), Theseus.FDMContext}; debug_mode=true)
    println("Successfully built rule FOR update_factorization!")
catch e
    println("Failed to build rule for update_factorization!")
    Base.display_error(e, catch_backtrace())
end

println("\nChecking solve_explicit! rule...")
try
    rule = Mooncake.build_rrule(Mooncake.MooncakeInterpreter(Mooncake.ReverseMode), Tuple{typeof(Theseus.solve_explicit!), Theseus.FDMContext, Vector{Float64}, SparseMatrixCSC{Int64, Int64}, SparseMatrixCSC{Int64, Int64}, Matrix{Float64}, Matrix{Float64}})
    println("Successfully built rule for solve_explicit!")
catch e
    println("Failed to build rule for solve_explicit!")
    Base.display_error(e, catch_backtrace())
end
