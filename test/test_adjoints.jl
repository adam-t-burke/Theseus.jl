using Theseus
using Mooncake
using LinearAlgebra
using SparseArrays
using Test

function test_adjoints_precision()
    println("Setting up small problem for adjoint validation...")
    
    # 3 nodes: 1, 2 free, 3 fixed.
    # Edge 1: 1->2
    # Edge 2: 2->3
    ne = 2
    nn = 3
    free_nodes = [1, 2]
    fixed_nodes = [3]
    
    I_idx = [1, 1, 2, 2]
    J_idx = [1, 2, 2, 3]
    V = [-1.0, 1.0, -1.0, 1.0]
    incidence = sparse(I_idx, J_idx, V, ne, nn)
    
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
    
    # Apply a load in Z to Node 1
    free_loads = zeros(2, 3)
    free_loads[1, 3] = -5.0
    loads = Theseus.LoadData(free_loads)
    
    # Reference positions for all nodes
    ref_pos = zeros(3, 3)
    ref_pos[3, :] = [2.0, 1.0, 0.5]
    geometry = Theseus.GeometryData(ref_pos)
    
    # Anchors - node 3 is fixed for now
    anchors = Theseus.AnchorInfo(
        variable_indices = Int[],
        fixed_indices = [3],
        reference_positions = ref_pos,
        initial_variable_positions = zeros(0, 3)
    )
    
    params = Theseus.OptimizationParameters(
        Theseus.AbstractObjective[],
        Theseus.default_bounds(ne),
        Theseus.SolverOptions(1e-6, 1e-6, 1, 1, false, 1000.0, 10.0, true),
        Theseus.TracingOptions(false, 1)
    )
    
    problem = Theseus.OptimizationProblem(topo, loads, geometry, anchors, params)
    cache = Theseus.OptimizationCache(problem)
    
    q_init = [1.5, 2.5]
    anchor_pos = zeros(0, 3)
    
    # Define scalar objective function
    function objective(q_val)
        xyz_free = Theseus.solve_FDM!(cache, q_val, problem, anchor_pos)
        # Loss: sum of squared y and z coordinates of free nodes
        return sum(abs2, xyz_free[:, 2:3])
    end

    println("Computing Finite Difference Gradient...")
    eps = 1e-6
    v0 = objective(q_init)
    g_fd = zeros(length(q_init))
    for i in 1:length(q_init)
        q_eps = copy(q_init)
        q_eps[i] += eps
        g_fd[i] = (objective(q_eps) - v0) / eps
    end
    
    println("Computing Mooncake Gradient...")
    try
        rule = Mooncake.prepare_gradient_cache(objective, q_init)
        v_mc, grads_mc = Mooncake.value_and_gradient!!(rule, objective, q_init)
        g_mc_q = grads_mc[2]
        
        println("Objective value (MC): ", v_mc)
        println("Objective value (FD): ", v0)
        println("Mooncake Gradient w.r.t. q: ", g_mc_q)
        println("FD Gradient w.r.t. q:       ", g_fd)
        
        @test v_mc ≈ v0 atol=1e-6
        @test isapprox(g_mc_q, g_fd, rtol=1e-3)
        println("Adjoint validation successful for q!")
    catch e
        @error "Mooncake rule failed" exception=(e, catch_backtrace())
        rethrow(e)
    end

    # Also test anchors if any are variable
    println("\nValidating Anchor Gradients...")
    # Create a new problem with node 3 as a variable anchor
    anchors_var = Theseus.AnchorInfo(
        variable_indices = [3],
        fixed_indices = Int[],
        reference_positions = ref_pos,
        initial_variable_positions = reshape([2.0, 1.0, 0.5], 1, 3)
    )
    problem_var = Theseus.OptimizationProblem(topo, loads, geometry, anchors_var, params)
    cache_var = Theseus.OptimizationCache(problem_var)
    
    a0 = reshape([2.0, 1.0, 0.5], 1, 3)
    
    function objective_anchors(a_pos)
        xyz_free = Theseus.solve_FDM!(cache_var, q_init, problem_var, a_pos)
        return sum(abs2, xyz_free)
    end
    
    v0_a = objective_anchors(a0)
    g_fd_a = zeros(size(a0))
    for i in 1:length(a0)
        a_eps = copy(a0)
        a_eps[i] += eps
        g_fd_a[i] = (objective_anchors(a_eps) - v0_a) / eps
    end
    
    rule_a = Mooncake.prepare_gradient_cache(objective_anchors, a0)
    v_mc_a, grads_mc_a = Mooncake.value_and_gradient!!(rule_a, objective_anchors, a0)
    g_mc_a = grads_mc_a[2]
    
    println("Mooncake Anchor Gradient: ", g_mc_a)
    println("FD Anchor Gradient:       ", g_fd_a)
    @test isapprox(g_mc_a, g_fd_a, rtol=1e-3)
    println("Adjoint validation successful for anchors!")
end

if abspath(PROGRAM_FILE) == @__FILE__
    test_adjoints_precision()
end
