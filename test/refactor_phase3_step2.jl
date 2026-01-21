using Test
using Theseus
using LinearAlgebra
using SparseArrays

function create_tiny_problem()
    # 3 nodes in a line: 1 (free), 2 (fixed), 3 (fixed)
    # Edge 1: 2 -> 1, Edge 2: 3 -> 1
    ne = 2
    nn = 3
    I = [1, 1, 2, 2]
    J = [2, 1, 3, 1]
    V = [-1, 1, -1, 1]
    incidence = sparse(I, J, V, ne, nn)
    
    free_nodes = [1]
    fixed_nodes = [2, 3]
    
    free_incidence = incidence[:, free_nodes]
    fixed_incidence = incidence[:, fixed_nodes]
    
    topo = Theseus.NetworkTopology(incidence, free_incidence, fixed_incidence, ne, nn, free_nodes, fixed_nodes)
    
    # Loads
    Pn = [0.0 1.0 0.0] # 1.0 load in Y dir on node 1
    loads = Theseus.LoadData(Pn)
    
    # Fixed positions
    Nf_full = [0.0 0.0 0.0;  # node 1 (placeholder)
               -1.0 0.0 0.0; # node 2
               1.0 0.0 0.0]  # node 3
    geometry = Theseus.GeometryData(Nf_full)
    
    anchors = Theseus.AnchorInfo(variable_indices=Int[], fixed_indices=fixed_nodes, reference_positions=Nf_full, initial_variable_positions=zeros(0, 3))
    params = Theseus.OptimizationParameters(Theseus.AbstractObjective[], Theseus.default_bounds(ne), Theseus.SolverOptions(1e-6, 1e-6, 1, 1, false, 1000.0, 10.0, true), Theseus.TracingOptions(false, 1))
    
    return Theseus.OptimizationProblem(topo, loads, geometry, anchors, params)
end

@testset "Phase 3.2: Gradient Correctness" begin
    prob = create_tiny_problem()
    cache = Theseus.FDMCache(prob)
    q0 = [1.5, 2.5]
    copyto!(cache.q, q0)
    
    # 1. Forward Pass
    Theseus.solve_explicit!(cache, prob, zeros(0,3))
    x0 = copy(cache.x)
    
    # 2. Objective: J = sum(x.^2)
    # dJ/dx = 2*x
    dJ_dx = 2 .* x0
    
    # 3. Adjoint Pass
    Theseus.Adjoints.solve_adjoint!(cache, dJ_dx)
    Theseus.Adjoints.accumulate_gradients!(cache, prob)
    adj_grad_q = copy(cache.grad_q)
    
    # 4. Finite Difference for q
    eps = 1e-7
    fd_grad_q = zeros(2)
    for i in 1:2
        q_eps = copy(q0)
        q_eps[i] += eps
        copyto!(cache.q, q_eps)
        Theseus.solve_explicit!(cache, prob, zeros(0,3))
        J_plus = sum(cache.x .^ 2)
        
        q_eps[i] -= 2*eps
        copyto!(cache.q, q_eps)
        Theseus.solve_explicit!(cache, prob, zeros(0,3))
        J_minus = sum(cache.x .^ 2)
        
        fd_grad_q[i] = (J_plus - J_minus) / (2*eps)
    end
    
    @info "Adjoint Grad Q: $adj_grad_q"
    @info "Finite Diff Grad Q: $fd_grad_q"
    
    @test adj_grad_q ≈ fd_grad_q rtol=1e-5

    # 5. Finite Difference for Nf
    # Node 2 is at [-1, 0, 0]. Let's nudge its X.
    fd_grad_Nf2x = 0.0
    
    # Original Nf
    nudge_node = 2
    nudge_dim = 1
    
    old_val = prob.geometry.fixed_node_positions[nudge_node, nudge_dim]
    
    prob.geometry.fixed_node_positions[nudge_node, nudge_dim] = old_val + eps
    copyto!(cache.q, q0)
    Theseus.solve_explicit!(cache, prob, zeros(0,3))
    J_plus = sum(cache.x .^ 2)
    
    prob.geometry.fixed_node_positions[nudge_node, nudge_dim] = old_val - eps
    copyto!(cache.q, q0)
    Theseus.solve_explicit!(cache, prob, zeros(0,3))
    J_minus = sum(cache.x .^ 2)
    
    fd_grad_Nf2x = (J_plus - J_minus) / (2*eps)
    prob.geometry.fixed_node_positions[nudge_node, nudge_dim] = old_val # restore
    
    @info "Adjoint Grad Nf[2,1]: $(cache.grad_Nf[2,1])"
    @info "Finite Diff Grad Nf[2,1]: $fd_grad_Nf2x"
    
    @test cache.grad_Nf[2,1] ≈ fd_grad_Nf2x rtol=1e-5
end
