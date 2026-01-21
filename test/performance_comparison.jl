using Theseus
using SparseArrays
using LinearAlgebra
using BenchmarkTools

function create_benchmark_problem(N)
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

function old_solve_explicit(q, Cn, Cf, Pn, Nf)
    # Correct broadcasting: q is length ne, Cn is ne x nn_free.
    # We want to scale each ROW of Cn by q. 
    # In Julia, A .* v scales ROWS if v is a vector? No, that's columns.
    # To scale rows, we do q .* Cn.
    Cnq = q .* Cn
    Cfq = q .* Cf
    return (Cn' * Cnq) \ (Pn - Cn' * (Cfq * Nf))
end

function compare_performance(N)
    prob = create_benchmark_problem(N)
    cache = Theseus.FDMCache(prob)
    q = ones(N)
    copyto!(cache.q, q)
    var_anchors = zeros(0, 3)
    
    Cn = convert(SparseMatrixCSC{Float64, Int64}, prob.topology.free_incidence)
    Cf = convert(SparseMatrixCSC{Float64, Int64}, prob.topology.fixed_incidence)
    Pn = prob.loads.free_node_loads
    fixed_indices = prob.topology.fixed_node_indices
    Nf_fixed = prob.geometry.fixed_node_positions[fixed_indices, :]

    println("\n--- Performance Comparison N=$N ---")
    
    # 1. Old Implementation
    # Note: `Cn .* q` works if q is a vector and Cn is ne x nn_free? 
    # Cn is ne x nn_free. q is ne. Julia 1.12+ supports broadcasting Cn .* q.
    # In old code: Cnq = Cn .* q. 
    # Warmup
    old_solve_explicit(q, Cn, Cf, Pn, Nf_fixed)
    t_old = @belapsed old_solve_explicit($q, $Cn, $Cf, $Pn, $Nf_fixed)
    a_old = @allocated old_solve_explicit(q, Cn, Cf, Pn, Nf_fixed)
    
    # 2. New Implementation
    # Warmup
    Theseus.solve_explicit!(cache, prob, var_anchors)
    t_new = @belapsed Theseus.solve_explicit!($cache, $prob, $var_anchors)
    a_new = @allocated Theseus.solve_explicit!(cache, prob, var_anchors)
    
    println("Old (Backslash): $(round(t_old*1000, digits=2)) ms, $a_old bytes")
    println("New (In-place) : $(round(t_new*1000, digits=2)) ms, $a_new bytes")
end

compare_performance(1000)
compare_performance(10000)
