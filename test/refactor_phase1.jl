using Test
using Theseus
using SparseArrays
using LinearSolve
using Mooncake

@testset "Refactor Phase 1: Foundation & OptimizationCache" begin
    # Create a small dummy problem
    # Topology: 2 edges, 2 free nodes, 1 fixed node
    # Edge 1: Node 1 (free) - Node 2 (free)
    # Edge 2: Node 2 (free) - Node 3 (fixed)
    
    ne = 2
    nn = 3
    free_nodes = [1, 2]
    fixed_nodes = [3]
    
    # Incidence matrix (edges x nodes)
    # Edge 1: -1 at node 1, 1 at node 2
    # Edge 2: -1 at node 2, 1 at node 3
    I = [1, 1, 2, 2]
    J = [1, 2, 2, 3]
    V = [-1, 1, -1, 1]
    incidence = sparse(I, J, V, ne, nn)
    
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
    
    loads = Theseus.LoadData(zeros(length(free_nodes), 3))
    geometry = Theseus.GeometryData(reshape([0.0, 0.0, 0.0], 1, 3))
    anchors = Theseus.AnchorInfo(collect(1:1), Int[], zeros(1, 3), zeros(0, 3))
    
    params = Theseus.OptimizationParameters(
        Theseus.AbstractObjective[],
        Theseus.default_bounds(ne),
        Theseus.SolverOptions(1e-6, 1e-6, 1, 1, false, 1000.0, 10.0, true),
        Theseus.TracingOptions(false, 1)
    )
    
    problem = Theseus.OptimizationProblem(topo, loads, geometry, anchors, params)
    
    @testset "OptimizationCache Initialization" begin
        cache = Theseus.OptimizationCache(problem)
        
        @test size(cache.A) == (2, 2)
        @test length(cache.q_to_nz) == ne
        
        # Edge 1 (free-free) should map to 4 entries: (1,1), (2,2), (1,2), (2,1)
        @test length(cache.q_to_nz[1]) == 4
        # Edge 2 (free-fixed) should map to 1 entry: (2,2)
        @test length(cache.q_to_nz[2]) == 1
        
        @test cache.integrator isa LinearSolve.LinearCache
        
        # Verify buffers
        @test size(cache.x) == (2, 3)
        @test size(cache.λ) == (2, 3)
        @test size(cache.q) == (2,)
        
        # Verify Mooncake fdata buffers
        @test size(cache.x_fdata) == (2, 3)
        @test size(cache.q_fdata) == (2,)
    end
end
