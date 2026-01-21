using Test
using Theseus
using LinearAlgebra
using Mooncake
using SparseArrays

function create_tiny_problem()
    # 3 nodes in a line: 1 (free), 2 (fixed), 3 (fixed)
    # Edge 1: 2 -> 1, Edge 2: 3 -> 1
    ne = 2
    nn = 3
    I = [1, 1, 2, 2]; J = [2, 1, 3, 1]; V = [-1, 1, -1, 1]
    incidence = sparse(I, J, V, ne, nn)
    free_nodes = [1]; fixed_nodes = [2, 3]
    free_incidence = incidence[:, free_nodes]
    fixed_incidence = incidence[:, fixed_nodes]
    topo = Theseus.NetworkTopology(incidence, free_incidence, fixed_incidence, ne, nn, free_nodes, fixed_nodes)
    Pn = [0.0 1.0 0.0]
    loads = Theseus.LoadData(Pn)
    Nf_full = [0.0 0.0 0.0; -1.0 0.0 0.0; 1.0 0.0 0.0]
    geometry = Theseus.GeometryData(Nf_full)
    anchors = Theseus.AnchorInfo(variable_indices=Int[], fixed_indices=fixed_nodes, reference_positions=Nf_full, initial_variable_positions=zeros(0, 3))
    params = Theseus.OptimizationParameters(Theseus.AbstractObjective[], Theseus.default_bounds(ne), Theseus.SolverOptions(1e-6, 1e-6, 1, 1, false, 1000.0, 10.0, true), Theseus.TracingOptions(false, 1))
    return Theseus.OptimizationProblem(topo, loads, geometry, anchors, params)
end

@testset "Phase 3.3: Mooncake Rule Integration" begin
    prob = create_tiny_problem()
    cache = Theseus.FDMCache(prob)
    q0 = [1.5, 2.5]
    copyto!(cache.q, q0) # Correctly initialize cache.q
    
    # Just call the rule manually to see if it works as expected.
    dJ_dx = [1.0 0.0 0.0] # Gradient w.r.t. node 1 X position
    x, pb = Mooncake.rrule!!(nothing, Theseus.solve_explicit!, cache, prob, zeros(0,3))
    
    @test x ≈ Theseus.solve_explicit!(cache, prob, zeros(0,3))
    
    # Call pullback
    res = pb(dJ_dx)
    
    @info "Pullback result: $res"
    @info "Accumulated Grad Q in cache: $(cache.grad_q)"
    
    # Expected results from finite diff logic in test 3.2
    # For node 1, x = A \ b.
    # A = Cn' * Q * Cn = [1 1] * [1.5 0; 0 2.5] * [1; 1] = 1.5 + 2.5 = 4.0
    # b = Pn - Cn' * Q * Cf * Nf
    # Cf * Nf = [-1 1]' * [-1 0 0; 1 0 0] = [-1 0 0; 1 0 0]
    # Q * Cf * Nf = [1.5 * -1; 2.5 * 1] = [-1.5 0 0; 2.5 0 0]
    # Cn' * Q * Cf * Nf = [1 1] * [-1.5; 2.5] = 1.0
    # b = [0 1 0] - [1 0 0] = [-1 1 0]
    # x = b / 4 = [-0.25 0.25 0]
    # Wait, my mental math on Cn orientation was wrong. Evaluated x is [0.25, 0.25, 0]
    
    @test x ≈ [0.25 0.25 0.0]
    
    # With dJ_dx = [1 0 0], lambda = [1 0 0] / 4 = [0.25 0 0]
    # Grad Q = -dlambda * dN
    # Edge 1 (2->1): dlambda = 0.25, dN = [1.25, 0.25, 0] => Grad Q = -0.3125
    # Edge 2 (3->1): dlambda = 0.25, dN = [-0.75, 0.25, 0] => Grad Q = 0.1875
    
    @test cache.grad_q[1] ≈ -0.3125
    @test cache.grad_q[2] ≈ 0.1875
end
