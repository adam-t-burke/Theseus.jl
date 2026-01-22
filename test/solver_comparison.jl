using Theseus
using SparseArrays
using LinearAlgebra
using BenchmarkTools
using LinearSolve

function create_random_fdm_problem(ne, nn_free, nn_fixed)
    nn = nn_free + nn_fixed
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

function test_solver(N, alg)
    p = create_random_fdm_problem(N, Int(N/2), Int(N/2)+1)
    # Manually create cache with specific alg
    A = p.topology.free_incidence' * p.topology.free_incidence
    b = zeros(size(A, 1), 3)
    x = zeros(size(A, 1), 3)
    prob = LinearProblem(A, b; u0=x)
    integrator = init(prob, alg)
    
    # Re-wrap in a mock cache object for testing
    # (Just enough to call LinearSolve.solve!)
    # Warmup
    LinearSolve.solve!(integrator)
    
    allocs = @allocated begin
        integrator.A = A # simulate update
        LinearSolve.solve!(integrator)
    end
    return allocs
end

println("Algorithm: UMFPACKFactorization()")
println("N=10: $(test_solver(10, UMFPACKFactorization())) bytes")
println("N=1000: $(test_solver(1000, UMFPACKFactorization())) bytes")

println("\nAlgorithm: KrylovJL_CG()")
println("N=10: $(test_solver(10, KrylovJL_CG())) bytes")
println("N=1000: $(test_solver(1000, KrylovJL_CG())) bytes")
