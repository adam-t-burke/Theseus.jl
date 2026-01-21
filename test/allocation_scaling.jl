using Test
using Theseus
using SparseArrays
using LinearAlgebra

function create_random_fdm_problem(ne, nn_free, nn_fixed)
    # Simple mesh: nodes in a line
    nn = nn_free + nn_fixed
    I = Int[]
    J = Int[]
    V = Int[]
    for k in 1:ne
        # Each edge connects node k to k+1
        push!(I, k)
        push!(J, k)
        push!(V, -1)
        push!(I, k)
        push!(J, k+1)
        push!(V, 1)
    end
    incidence = sparse(I, J, V, ne, nn)
    
    free_nodes = collect(1:nn_free)
    fixed_nodes = collect(nn_free+1:nn)
    
    free_incidence = incidence[:, free_nodes]
    fixed_incidence = incidence[:, fixed_nodes]
    
    topo = Theseus.NetworkTopology(
        incidence,
        free_incidence,
        fixed_incidence,
        ne,
        nn,
        free_nodes,
        fixed_nodes
    )
    
    Pn = randn(nn_free, 3)
    loads = Theseus.LoadData(Pn)
    Nf_full = randn(nn, 3)
    geometry = Theseus.GeometryData(Nf_full)
    anchors = Theseus.AnchorInfo(
        variable_indices=Int[], 
        fixed_indices=fixed_nodes, 
        reference_positions=Nf_full, 
        initial_variable_positions=zeros(0, 3)
    )
    params = Theseus.OptimizationParameters(
        Theseus.AbstractObjective[],
        Theseus.default_bounds(ne),
        Theseus.SolverOptions(1e-6, 1e-6, 1, 1, false, 1000.0, 10.0, true),
        Theseus.TracingOptions(false, 1)
    )
    
    return Theseus.OptimizationProblem(topo, loads, geometry, anchors, params)
end

@testset "Allocation Scaling Test" begin
    # Test case 1: Very Small
    p1 = create_random_fdm_problem(10, 5, 6)
    c1 = Theseus.FDMCache(p1)
    copyto!(c1.q, ones(10))
    Theseus.solve_explicit!(c1, p1, zeros(0,3)) # warm up
    a1 = @allocated Theseus.solve_explicit!(c1, p1, zeros(0,3))
    
    # Test case 2: Larger
    p2 = create_random_fdm_problem(1000, 500, 501)
    c2 = Theseus.FDMCache(p2)
    copyto!(c2.q, ones(1000))
    Theseus.solve_explicit!(c2, p2, zeros(0,3)) # warm up
    a2 = @allocated Theseus.solve_explicit!(c2, p2, zeros(0,3))
    
    # Test case 3: Much Larger
    p3 = create_random_fdm_problem(10000, 5000, 5001)
    c3 = Theseus.FDMCache(p3)
    copyto!(c3.q, ones(10000))
    Theseus.solve_explicit!(c3, p3, zeros(0,3)) # warm up
    a3 = @allocated Theseus.solve_explicit!(c3, p3, zeros(0,3))

    @info "Allocations for N=10 edges: $a1 bytes"
    @info "Allocations for N=1000 edges: $a2 bytes"
    @info "Allocations for N=10000 edges: $a3 bytes"
    
    # Check if a3 is significantly larger than a1
    # If it scales with N, a3 should be ~1000x a1. 
    # If metadata, they should be close.
    @test a3 < a1 * 2 # Ideally a small multiple, not scaling linearly
end
