using Theseus
using LinearAlgebra
using SparseArrays
using Mooncake
using Test

@testset "Full Pipeline Validation" begin
    # 1. Setup a simple 3-node problem
    # Node 1: (0,0,0) fixed, Node 2: (1,0,0) free, Node 3: (2,0,0) fixed
    nn = 3
    ne = 2
    free_nodes = [2]
    fixed_nodes = [1, 3]
    
    # Edges: 1->2 (edge 1), 2->3 (edge 2)
    # Edge 1: start 1, end 2. C[1,1]=-1, C[1,2]=1
    # Edge 2: start 2, end 3. C[2,2]=-1, C[2,3]=1
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
    
    nodes = [0.0 0.0 0.0; 1.0 0.0 0.0; 2.0 0.0 0.0]
    q_init = [1.0, 1.0]

    loads = Theseus.LoadData(zeros(1, 3)) # One free node
    loads.free_node_loads[1, 3] = -1.0
    
    ref_pos = nodes
    geometry = Theseus.GeometryData(ref_pos)
    anchors = Theseus.AnchorInfo(
        variable_indices = Int[],
        fixed_indices = [1, 3],
        reference_positions = ref_pos,
        initial_variable_positions = zeros(0, 3)
    )
    
    # Define an objective: match node 2 target (1, 0, -0.5)
    target_pos = [1.0 0.0 -0.5]
    obj = Theseus.TargetXYZObjective(weight=1.0, node_indices=[2], target=target_pos)
    # Add a length objective too
    obj2 = Theseus.TargetLengthObjective(weight=0.1, edge_indices=[1, 2], target=[1.0, 1.0])
    
    params = Theseus.OptimizationParameters(
        objectives = [obj, obj2],
        bounds = Theseus.Bounds(fill(0.1, 2), fill(10.0, 2)),
        solver = Theseus.SolverOptions(1e-8, 1e-8, 100, 1, false, 0.0, 1.0, false),
        tracing = Theseus.TracingOptions(false, 1)
    )
    
    problem = Theseus.OptimizationProblem(topo, loads, geometry, anchors, params)
    cache = Theseus.OptimizationCache(problem)

    function full_objective(q_val)
        # Use ref_pos as-is since no variable anchors
        snapshot = Theseus.evaluate_geometry(problem, q_val, zeros(0,3), cache)
        
        loss = 0.0
        for o in problem.parameters.objectives
            loss += Theseus.objective_loss(o, snapshot)
        end
        return loss
    end

    # Finite Difference
    eps = 1e-6
    v0 = full_objective(q_init)
    g_fd = zeros(2)
    for i in 1:2
        q_eps = copy(q_init)
        q_eps[i] += eps
        g_fd[i] = (full_objective(q_eps) - v0) / eps
    end

    # Mooncake
    println("Validating Pipeline with Mooncake...")
    rule = Mooncake.prepare_gradient_cache(full_objective, q_init)
    v_mc, grads_mc = Mooncake.value_and_gradient!!(rule, full_objective, q_init)
    g_mc = grads_mc[2]

    println("Objective (MC): ", v_mc)
    println("Objective (FD): ", v0)
    println("Gradient (MC):  ", g_mc)
    println("Gradient (FD):  ", g_fd)

    @test v_mc ≈ v0 atol=1e-6
    @test isapprox(g_mc, g_fd, rtol=1e-3)

    # Test variable anchors too
    println("\nTesting Variable Anchors...")
    anchors_var = Theseus.AnchorInfo(
        variable_indices = [3],
        fixed_indices = [1],
        reference_positions = ref_pos,
        initial_variable_positions = reshape([2.0, 0.0, 0.0], 1, 3)
    )
    problem_var = Theseus.OptimizationProblem(topo, loads, geometry, anchors_var, params)
    cache_var = Theseus.OptimizationCache(problem_var)
    anchor_init = reshape([2.0, 0.0, 0.0], 1, 3)

    function anchor_objective(θ)
        q = θ[1:2]
        a = reshape(θ[3:5], 1, 3)
        snapshot = Theseus.evaluate_geometry(problem_var, q, a, cache_var)
        loss = 0.0
        for o in problem_var.parameters.objectives
            loss += Theseus.objective_loss(o, snapshot)
        end
        return loss
    end

    θ_init = vcat(q_init, vec(anchor_init))
    v0_a = anchor_objective(θ_init)
    g_fd_a = zeros(5)
    for i in 1:5
        θ_eps = copy(θ_init)
        θ_eps[i] += eps
        g_fd_a[i] = (anchor_objective(θ_eps) - v0_a) / eps
    end

    rule_a = Mooncake.prepare_gradient_cache(anchor_objective, θ_init)
    v_mc_a, grads_mc_a = Mooncake.value_and_gradient!!(rule_a, anchor_objective, θ_init)
    g_mc_a = grads_mc_a[2]

    println("Anchor Objective (MC): ", v_mc_a)
    println("Anchor Objective (FD): ", v0_a)
    println("Anchor Gradient (MC):  ", g_mc_a)
    println("Anchor Gradient (FD):  ", g_fd_a)

    @test v_mc_a ≈ v0_a atol=1e-6
    @test isapprox(g_mc_a, g_fd_a, rtol=1e-3)
end
