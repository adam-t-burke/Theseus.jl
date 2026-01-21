using Test
using Theseus
using LinearAlgebra
using SparseArrays

function create_scaling_fdm_problem(N)
    nn_free = Int(N/2)
    nn_fixed = Int(N/2) + 1
    nn = nn_free + nn_fixed
    ne = N
    I = Int[]; J = Int[]; V = Int[]
    for k in 1:ne
        push!(I, k); push!(J, k); push!(V, -1)
        push!(I, k); push!(J, k+1); push!(V, 1)
    end
    incidence = sparse(I, J, V, ne, nn)
    free_nodes = collect(1:nn_free)
    fixed_nodes = collect(nn_free+1:nn)
    free_incidence = incidence[:, free_nodes]
    fixed_incidence = incidence[:, fixed_nodes]
    topo = Theseus.NetworkTopology(incidence, free_incidence, fixed_incidence, ne, nn, free_nodes, fixed_nodes)
    Pn = randn(nn_free, 3)
    loads = Theseus.LoadData(Pn)
    Nf_full = randn(nn, 3)
    geometry = Theseus.GeometryData(Nf_full)
    anchors = Theseus.AnchorInfo(variable_indices=Int[], fixed_indices=fixed_nodes, reference_positions=Nf_full, initial_variable_positions=zeros(0, 3))
    params = Theseus.OptimizationParameters(Theseus.AbstractObjective[], Theseus.default_bounds(ne), Theseus.SolverOptions(1e-6, 1e-6, 1, 1, false, 1000.0, 10.0, true), Theseus.TracingOptions(false, 1))
    return Theseus.OptimizationProblem(topo, loads, geometry, anchors, params)
end

@testset "Phase 3.1: Adjoint Factor Reuse" begin
    N = 1000
    prob = create_scaling_fdm_problem(N)
    cache = Theseus.FDMCache(prob)
    copyto!(cache.q, ones(N))
    
    # 1. Forward Pass (Factorizes matrix)
    Theseus.solve_explicit!(cache, prob, zeros(0,3))
    
    # 2. Adjoint Pass (Reuse factors)
    dJ_dx = randn(size(cache.x))
    
    # Warmup adjoint
    Theseus.Adjoints.solve_adjoint!(cache, dJ_dx)
    
    # Measure allocations
    allocs = @allocated Theseus.Adjoints.solve_adjoint!(cache, dJ_dx)
    
    @info "Adjoint solve allocations for N=$N: $allocs bytes"
    
    # We expect very low allocations (metadata only, no re-factorization)
    # Forward pass was ~340KB for N=1000. Adjoint should be < 10KB.
    @test allocs < 20000 
end
