using Theseus
using SparseArrays
using LinearAlgebra
using BenchmarkTools
using Mooncake

function create_scaling_problem(N)
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

function benchmark_size(N)
    prob = create_scaling_problem(N)
    cache = Theseus.FDMCache(prob)
    copyto!(cache.q, ones(N))
    var_anchors = zeros(0, 3)
    
    println("\n--- $N Edges ---")
    
    # 1. Forward Pass Bench
    # Warmup
    Theseus.solve_explicit!(cache, prob, var_anchors)
    fwd_time = @belapsed Theseus.solve_explicit!($cache, $prob, $var_anchors)
    fwd_allocs = @allocated Theseus.solve_explicit!(cache, prob, var_anchors)
    
    # 2. Mooncake Rule Initialization (Creating the pullback)
    # This might have small constant allocations due to closure creation
    _, pb = Mooncake.rrule!!(nothing, Theseus.solve_explicit!, cache, prob, var_anchors)
    
    # 3. Pullback Bench
    dx = randn(size(cache.x))
    # Warmup
    pb(dx)
    adj_time = @belapsed $pb($dx)
    adj_allocs = @allocated pb(dx)
    
    println("Forward   : $(round(fwd_time*1000, digits=2)) ms, $(fwd_allocs) bytes")
    println("Adjoint   : $(round(adj_time*1000, digits=2)) ms, $(adj_allocs) bytes")
end

sizes = [10, 1000, 10000, 100000]
for N in sizes
    benchmark_size(N)
end
