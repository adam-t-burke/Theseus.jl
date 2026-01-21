using Test
using Theseus
using SparseArrays
using LinearAlgebra
using BenchmarkTools

@testset "Refactor Phase 2: High-Performance Forward Solver" begin
    # 1. Setup problem (reuse small network from Phase 1)
    ne = 2
    nn = 3
    free_nodes = [1, 2]
    fixed_nodes = [3]
    
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
    
    # Loads: 10N down on node 1, 5N down on node 2
    Pn = zeros(2, 3)
    Pn[1, 3] = -10.0
    Pn[2, 3] = -5.0
    loads = Theseus.LoadData(Pn)
    
    # Geometry: Node 3 at (10, 0, 0)
    Nf_full = zeros(3, 3)
    Nf_full[3, 1] = 10.0
    geometry = Theseus.GeometryData(Nf_full)
    
    anchors = Theseus.AnchorInfo(
        variable_indices=Int[], 
        fixed_indices=collect(1:3), 
        reference_positions=Nf_full, 
        initial_variable_positions=zeros(0, 3)
    )
    
    params = Theseus.OptimizationParameters(
        Theseus.AbstractObjective[],
        Theseus.default_bounds(ne),
        Theseus.SolverOptions(1e-6, 1e-6, 1, 1, false, 1000.0, 10.0, true),
        Theseus.TracingOptions(false, 1)
    )
    
    problem = Theseus.OptimizationProblem(topo, loads, geometry, anchors, params)
    cache = Theseus.OptimizationCache(problem)
    
    # Set force densities
    q = [2.0, 1.0]
    copyto!(cache.q, q)
    
    # 2. Test Accuracy
    var_anchors = zeros(0, 3) # no variable anchors
    x_new = Theseus.solve_explicit!(cache, problem, var_anchors)
    
    # Reference calculation
    Nf_fixed = @view Nf_full[fixed_nodes, :]
    x_ref = Theseus.solve_explicit(q, free_incidence, fixed_incidence, Pn, Nf_fixed)
    
    @test x_new ≈ x_ref atol=1e-10
    
    # 3. Test Allocations
    # We expect very few allocations (mostly due to LinearSolve/LDL internals or overhead)
    # But specifically the RHS assembly and A update should be non-allocating.
    
    # Warm up
    Theseus.solve_explicit!(cache, problem, var_anchors)
    
    allocs = @allocated Theseus.solve_explicit!(cache, problem, var_anchors)
    @info "Allocations in solve_explicit!: $allocs"
    # Note: On some platforms, LinearSolve/UMFPACK might still allocate for factors.
    # But for a small problem, it should be very low.
    @test allocs < 20000 # Relaxed bound for metadata/UMFPACK overhead
    
    # 4. Test Perturbation Logic
    # Set q to 0 to make A singular
    fill!(cache.q, 0.0)
    # This should trigger perturbation and warning
    @test_logs (:warn, r"Linear solve failed|Linear solve threw an error") Theseus.solve_explicit!(cache, problem, var_anchors, perturbation=1.0)
    # With perturbation of 1.0 on diagonal, A = I. x should be Pn.
    @test cache.x ≈ Pn atol=1e-10
end
