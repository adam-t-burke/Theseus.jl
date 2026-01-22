using Test
using Theseus

# Example function to test
function add(a::Int, b::Int)
    return a + b
end

@testset "Math Tests" begin
    @test add(1, 2) == 3
    @test add(-1, 1) == 0
    @test add(0, 0) == 0
end