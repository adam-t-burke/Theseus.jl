using Theseus
using Mooncake
using LinearAlgebra
using SparseArrays

# Setup a small problem
function create_tiny_problem()
    n_free = 1
    n_fixed = 2
    n_nodes = 3
    n_edges = 2
    
    # 0 --(1)-- 1 --(2)-- 2
    # Node 0 fixed at (0,0,0), Node 2 fixed at (2,0,0)
    # Node 1 free at (1,1,0)
    
    II = [1, 1, 2, 2]
    JJ = [1, 2, 2, 3] # Edge 1: 0-1, Edge 2: 1-2
    VV = [-1.0, 1.0, -1.0, 1.0]
    C_total = sparse(II, JJ, VV, n_edges, n_nodes)
    
    free_indices = [2]
    fixed_indices = [1, 3]
    
    topo = Theseus.NetworkTopology(
        C_total,
        C_total[:, free_indices],
        C_total[:, fixed_indices],
        n_edges,
        n_nodes,
        free_indices,
        fixed_indices
    )
    
    loads = Theseus.LoadData(zeros(n_free, 3))
    # Fixed positions: Node 1 at (0,0,0), Node 3 at (2,0,0)
    # ref_pos[1] -> Node 1, ref_pos[2] -> Node 3
    ref_pos = [0.0 0.0 0.0; 2.0 0.0 0.0]
    geom = Theseus.GeometryData(ref_pos)
    
    anchors = Theseus.AnchorInfo(ref_pos)
    
    params = Theseus.OptimizationParameters(
        Theseus.AbstractObjective[],
        Theseus.Bounds(fill(0.1, n_edges), fill(10.0, n_edges)),
        Theseus.SolverOptions(1e-4, 1e-4, 10, 1, false, 0.0, 10.0, false),
        Theseus.TracingOptions(false, 1)
    )
    
    return Theseus.OptimizationProblem(topo, loads, geom, anchors, params)
end

println("--- Testing Mooncake integration (Zygote-free) ---")
problem = create_tiny_problem()
q_init = [1.0, 1.0]
state = Theseus.OptimizationState(q_init, problem.anchors.initial_variable_positions)
state.cache = Theseus.OptimizationCache(problem)

# Define a simple objective function
function my_objective_logic(θ)
    q = θ[1:2]
    # In evaluate_geometry, it will use solve_explicit! which has our Mooncake rule
    snap = Theseus.evaluate_geometry(problem, q, state.variable_anchor_positions, state.cache)
    return sum(snap.xyz_free.^2) # Minimize distance from origin
end

println("Computing value and gradient with Mooncake...")
gradient_cache = Mooncake.prepare_gradient_cache(my_objective_logic, q_init)
val, grad = Mooncake.value_and_gradient!!(gradient_cache, my_objective_logic, q_init)

println("Value: ", val)
println("Gradient: ", grad)

# Verify gradient is not zero
if all(grad .== 0)
    error("Gradient is zero! AD link might be broken.")
else
    println("Success: Gradient is non-zero.")
end

println("--- Result ---")
