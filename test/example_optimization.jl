using Theseus
using LinearAlgebra
using SparseArrays
using Test

function run_example()
    # 1. Setup small problem
    # Two nodes bridged between two fixed points
    # Fixed: 1 (0,0,0), 4 (3,0,0)
    # Free: 2 (1,0,0), 3 (2,0,0)
    nn = 4
    ne = 3
    free_nodes = [2, 3]
    fixed_nodes = [1, 4]
    
    # Edges: 1-2, 2-3, 3-4
    I_idx = [1, 1, 2, 2, 3, 3]
    J_idx = [1, 2, 2, 3, 3, 4]
    V = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
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
    
    # Loads (gravity)
    loads = Theseus.LoadData(zeros(2, 3))
    loads.free_node_loads[:, 3] .= -1.0
    
    # Geometry
    ref_pos = [0.0 0.0 0.0; 1.0 0.0 0.0; 2.0 0.0 0.0; 3.0 0.0 0.0]
    geometry = Theseus.GeometryData(ref_pos)
    
    # Anchors
    anchors = Theseus.AnchorInfo(
        variable_indices = Int[],
        fixed_indices = [1, 4],
        reference_positions = ref_pos,
        initial_variable_positions = zeros(0, 3)
    )
    
    # 2. Objectives
    # We want node 2 and 3 to be at Z = -1.0
    target_nodes = [2, 3]
    target_pos = [1.0 0.0 -1.0; 2.0 0.0 -1.0]
    obj1 = Theseus.TargetXYZObjective(weight=10.0, node_indices=target_nodes, target=target_pos)
    
    # We want to minimize length variation (uniform spacing)
    obj2 = Theseus.LengthVariationObjective(weight=1.0, edge_indices=[1, 2, 3])
    
    # 3. Parameters
    params = Theseus.OptimizationParameters(
        objectives = [obj1, obj2],
        bounds = Theseus.Bounds(fill(0.1, ne), fill(10.0, ne)),
        solver = Theseus.SolverOptions(
            absolute_tolerance = 1e-8,
            relative_tolerance = 1e-8,
            max_iterations = 100,
            report_frequency = 10,
            show_progress = true,
            barrier_weight = 1e-3,
            barrier_sharpness = 10.0,
            use_auto_scaling = true
        ),
        tracing = Theseus.TracingOptions(record_nodes=true, emit_frequency=1)
    )
    
    # 4. Optimization
    problem = Theseus.OptimizationProblem(topo, loads, geometry, anchors, params)
    
    initial_q = fill(1.0, ne)
    state = Theseus.OptimizationState(initial_q, zeros(0, 3))
    
    println("Starting optimization loop...")
    println("Initial loss: ", full_objective_eval(problem, initial_q))
    
    result, final_snapshot = Theseus.optimize_problem!(problem, state, on_iteration = (s, snap, loss) -> begin
        if s.iterations % 10 == 0
            println("Iteration $(s.iterations): Loss = $loss")
        end
    end)
    
    println("\nOptimization finished!")
    println("Final q: ", state.force_densities)
    println("Final loss: ", state.loss_trace[end])
    println("Final Node Positions (Z): ", final_snapshot.xyz_full[:, 3])
    
    # Verify results
    @test isapprox(final_snapshot.xyz_full[2, 3], -1.0, atol=0.1)
    @test isapprox(final_snapshot.xyz_full[3, 3], -1.0, atol=0.1)
end

function full_objective_eval(problem, q)
    snapshot = Theseus.evaluate_geometry(problem, q, zeros(0,3))
    loss = 0.0
    for obj in problem.parameters.objectives
        loss += Theseus.objective_loss(obj, snapshot)
    end
    return loss
end

run_example()
