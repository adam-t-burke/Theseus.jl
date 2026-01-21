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

function debug_allocs(N)
    p = create_random_fdm_problem(N, Int(N/2), Int(N/2)+1)
    cache = Theseus.FDMCache(p)
    copyto!(cache.q, ones(N))
    var_anchors = zeros(0,3)
    
    println("\n--- Debugging Allocs for N=$N ---")
    
    # Warmup
    Theseus.solve_explicit!(cache, p, var_anchors)

    # 1. Nzval update
    a1 = @allocated begin
        fill!(cache.A.nzval, 0.0)
        for k in 1:length(cache.q)
            qk = cache.q[k]
            for (nz_idx, coeff) in cache.q_to_nz[k]
                cache.A.nzval[nz_idx] += qk * coeff
            end
        end
    end
    println("Update nzval: $a1 bytes")

    # 2. current_fixed_positions
    a2 = @allocated Theseus.current_fixed_positions!(cache.Nf, p, var_anchors)
    println("current_fixed_positions: $a2 bytes")

    # 3. View and mul! (Cf)
    fixed_indices = p.topology.fixed_node_indices
    a3 = @allocated begin
        Nf_fixed = @view cache.Nf[fixed_indices, :]
        mul!(cache.Cf_Nf, cache.Cf, Nf_fixed)
    end
    println("Cf_Nf mul!: $a3 bytes")

    # 4. Q_Cf_Nf loop
    a4 = @allocated begin
        for j in 1:3
            for i in 1:length(cache.q)
                cache.Q_Cf_Nf[i, j] = cache.q[i] * cache.Cf_Nf[i, j]
            end
        end
    end
    println("Q_Cf_Nf loop: $a4 bytes")

    # 5. RHS assembly
    rhs = cache.integrator.b
    a5 = @allocated begin
        copyto!(rhs, cache.Pn)
        mul!(rhs, p.topology.free_incidence', cache.Q_Cf_Nf, -1.0, 1.0)
    end
    println("RHS assembly: $a5 bytes")

    # 6. LinearSolve update and solve
    a6 = @allocated begin
        cache.integrator.A = cache.A
    end
    println("Integrator A update: $a6 bytes")

    a7 = @allocated begin
        LinearSolve.solve!(cache.integrator)
    end
    println("LinearSolve.solve!: $a7 bytes")
end

debug_allocs(10)
debug_allocs(1000)
debug_allocs(10000)
