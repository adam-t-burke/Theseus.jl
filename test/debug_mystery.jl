using Theseus
using SparseArrays
using LinearAlgebra
using BenchmarkTools
using LDLFactorizations

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

function compare(N)
    prob = create_benchmark_problem(N)
    cache = Theseus.OptimizationCache(prob)
    copyto!(cache.q, ones(N))
    var_anchors = zeros(0, 3)

    println("\nN=$N Comparison")

    # 1. Julia Backslash
    Cn = convert(SparseMatrixCSC{Float64, Int64}, prob.topology.free_incidence)
    q = cache.q
    fixed_indices = prob.topology.fixed_node_indices
    Nf_fixed = prob.geometry.fixed_node_positions[fixed_indices, :]
    
    t_backslash = @belapsed begin
        A = $Cn' * ($q .* $Cn)
        A \ randn(size($Cn, 2), 3)
    end
    println("Backslash: $(round(t_backslash*1000, digits=3)) ms")

    # 2. Theseus solve_explicit!
    # Warmup
    Theseus.solve_explicit!(cache, prob, var_anchors)
    
    println("Breakdown for Theseus:")
    
    # nzval
    t_nz = @belapsed begin
        fill!($cache.A.nzval, 0.0)
        for k in 1:length($cache.q)
            qk = $cache.q[k]
            for (nz_idx, coeff) in $cache.q_to_nz[k]
                $cache.A.nzval[nz_idx] += qk * coeff
            end
        end
    end
    println("  nzval update: $(round(t_nz*1000, digits=3)) ms")
    
    # RHS
    println("  RHS Prep Breakdown:")
    t1 = @belapsed Theseus.current_fixed_positions!($cache.Nf, $prob, $var_anchors)
    println("    current_fixed_positions: $(round(t1*1000, digits=3)) ms")
    
    fi = prob.topology.fixed_node_indices
    nf_f_dense = zeros(length(fi), 3)
    t2 = @belapsed begin
        copyto!($nf_f_dense, @view $cache.Nf[$fi, :])
        mul!($cache.Cf_Nf, $cache.Cf, $nf_f_dense)
    end
    println("    copy+mul!(Cf, Nf_f_dense): $(round(t2*1000, digits=3)) ms")
    
    t3 = @belapsed $cache.Q_Cf_Nf .= $cache.q .* $cache.Cf_Nf
    println("    broadcasting Q         : $(round(t3*1000, digits=3)) ms")
    
    t4 = @belapsed copyto!($cache.grad_x, $cache.Pn)
    println("    copyto!(Pn)            : $(round(t4*1000, digits=3)) ms")
    
    t5 = @belapsed mul!($cache.grad_x, $cache.Cn', $cache.Q_Cf_Nf, -1.0, 1.0)
    println("    mul!(Cn', Q_Cf_Nf)     : $(round(t5*1000, digits=3)) ms")
    
    # Solve
    t_sol = @belapsed begin
        LDLFactorizations.ldl_factorize!($cache.A, $cache.factor)
        ldiv!($cache.x, $cache.factor, $cache.grad_x)
    end
    println("  LDL solve   : $(round(t_sol*1000, digits=3)) ms")
    
    # Total
    t_total = @belapsed Theseus.solve_explicit!($cache, $prob, $var_anchors)
    println("Total Theseus : $(round(t_total*1000, digits=3)) ms")
end

compare(10000)
