using Theseus
using Mooncake
using LinearAlgebra
using Statistics
using SparseArrays

function test_mooncake_only()
    println("Setting up problem...")
    
    ne = 2
    nn = 3
    free_nodes = [1, 2]
    fixed_nodes = [3]
    
    I = [1, 1, 2, 2]
    J = [1, 2, 2, 3]
    V = [-1, 1, -1, 1]
    incidence = sparse(I, J, V, ne, nn)
    
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
    
    loads = Theseus.LoadData(zeros(length(free_nodes), 3))
    geometry = Theseus.GeometryData(reshape([0.0, 0.0, 1.0], 1, 3))
    anchors = Theseus.AnchorInfo(Int[], collect(1:1), reshape([0.0, 0.0, 1.0], 1, 3), zeros(0,3))
    
    params = Theseus.OptimizationParameters(
        Theseus.AbstractObjective[],
        Theseus.default_bounds(ne),
        Theseus.SolverOptions(1e-6, 1e-6, 1, 1, false, 1000.0, 10.0, true),
        Theseus.TracingOptions(false, 1)
    )
    
    problem = Theseus.OptimizationProblem(topo, loads, geometry, anchors, params)
    cache = Theseus.OptimizationCache(problem)
    
    # Define a simple objective
    function objective(q_scaled, anchors)
        curr_q = ones(ne) .* q_scaled
        state = Theseus.OptimizationState(curr_q, anchors)
        state.cache = cache # Inject the cache
        coords = Theseus.evaluate_geometry(problem, curr_q, anchors, cache)
        return sum(abs2, coords)
    end
    
    q_scaled = ones(ne)
    anchors_pos = geometry.fixed_node_positions
    
    println("Preparing gradient cache...")
    rule = Mooncake.prepare_gradient_cache(objective, q_scaled, anchors_pos)
    
    println("Running value_and_gradient!!...")
    v, g = Mooncake.value_and_gradient!!(rule, objective, q_scaled, anchors_pos)
    
    println("Success!")
    println("Value: ", v)
    println("Gradient length: ", length(g[2])) 
end

try
    test_mooncake_only()
catch e
    @error "Test failed" exception=(e, catch_backtrace())
    # Re-throw to see the full error in terminal output
    rethrow(e)
end
