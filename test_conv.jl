using Theseus
using SparseArrays
using LinearAlgebra
using Optim

function test_convergence()
    # 3 nodes in a line: 1 (fixed), 2 (free), 3 (fixed)
    # Node 1 at (0,0,0), Node 2 at (5,0,0), Node 3 at (10,0,0)
    ne = 2
    nn = 3
    
    # Edges: (1-2) and (2-3)
    # Incidence:
    # E1: -1 at N1, +1 at N2
    # E2: -1 at N2, +1 at N3
    I = [1, 1, 2, 2]
    J = [1, 2, 2, 3]
    V = [-1, 1, -1, 1]
    incidence = sparse(I, J, V, ne, nn)
    
    free_nodes = [2]
    fixed_nodes = [1, 3]
    
    topo = Theseus.NetworkTopology(
        incidence, 
        incidence[:, free_nodes], 
        incidence[:, fixed_nodes], 
        ne, nn, free_nodes, fixed_nodes
    )
    
    # Target Node 2 to move in Z, starting from a flat line
    obj = Theseus.TargetXYZObjective(weight=1.0, node_indices=[2], target=[5.0 0.0 -5.0])
    
    solver = Theseus.SolverOptions(
        absolute_tolerance = 1e-12, # Extremely tight to avoid gradient stop
        relative_tolerance = 1e-2,  # Loose reltol to see if it stops
        max_iterations = 100,
        report_frequency = 1,
        show_progress = true,
        barrier_weight = 0.0,
        barrier_sharpness = 1.0,
        use_auto_scaling = false
    )
    
    params = Theseus.OptimizationParameters(
        objectives = [obj],
        bounds = Theseus.Bounds(fill(0.1, ne), fill(10.0, ne)),
        solver = solver,
        tracing = Theseus.TracingOptions(false, 1)
    )
    
    # Reference positions for fixed nodes
    fixed_pos = [0.0 0.0 0.0; 10.0 0.0 0.0]
    geometry = Theseus.GeometryData(fixed_pos)
    
    anchors = Theseus.AnchorInfo(
        variable_indices = Int[],
        fixed_indices = fixed_nodes,
        reference_positions = fixed_pos,
        initial_variable_positions = zeros(0, 3)
    )
    
    # Gravity-like load on Node 2 to make it move
    loads = Theseus.LoadData([0.0 0.0 -1.0])
    
    problem = Theseus.OptimizationProblem(topo, loads, geometry, anchors, params)
    state = Theseus.OptimizationState(fill(1.0, ne), zeros(0, 3))
    
    println("Running optimization with relative_tolerance = 0.01")
    result, _ = Theseus.optimize_problem!(problem, state)
    
    display(result)
    println("Iterations: ", Optim.iterations(result))
    println("Converged: ", Optim.converged(result))
end

test_convergence()
