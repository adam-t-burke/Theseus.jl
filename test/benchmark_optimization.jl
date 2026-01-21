using Theseus
using SparseArrays
using LinearAlgebra
using BenchmarkTools
using ADTypes
using DifferentiationInterface
import Mooncake
using TimerOutputs

function create_grid_problem(M, N)
    # M x N grid
    num_nodes = M * N
    num_edges = (M-1)*N + M*(N-1)
    
    I_idx = Int[]
    J_idx = Int[]
    V = Float64[]
    
    edge_idx = 1
    # Horizontal edges
    for i in 1:M
        for j in 1:N-1
            n1 = (i-1)*N + j
            n2 = (i-1)*N + j + 1
            push!(I_idx, edge_idx); push!(J_idx, n1); push!(V, -1.0)
            push!(I_idx, edge_idx); push!(J_idx, n2); push!(V, 1.0)
            edge_idx += 1
        end
    end
    # Vertical edges
    for i in 1:M-1
        for j in 1:N
            n1 = (i-1)*N + j
            n2 = i*N + j
            push!(I_idx, edge_idx); push!(J_idx, n1); push!(V, -1.0)
            push!(I_idx, edge_idx); push!(J_idx, n2); push!(V, 1.0)
            edge_idx += 1
        end
    end
    
    incidence = sparse(I_idx, J_idx, V, num_edges, num_nodes)
    
    # Boundary nodes are fixed
    fixed_nodes = Int[]
    for i in 1:M
        push!(fixed_nodes, (i-1)*N + 1)
        push!(fixed_nodes, (i-1)*N + N)
    end
    for j in 2:N-1
        push!(fixed_nodes, j)
        push!(fixed_nodes, (M-1)*N + j)
    end
    fixed_nodes = sort(unique(fixed_nodes))
    free_nodes = setdiff(1:num_nodes, fixed_nodes)
    
    free_incidence = incidence[:, free_nodes]
    fixed_incidence = incidence[:, fixed_nodes]
    
    topo = Theseus.NetworkTopology(
        incidence,
        free_incidence,
        fixed_incidence,
        num_edges,
        num_nodes,
        free_nodes,
        fixed_nodes
    )
    
    # Loads: Gravity
    Pn = zeros(length(free_nodes), 3)
    Pn[:, 3] .= -1.0
    loads = Theseus.LoadData(Pn)
    
    # Geometry: Flat grid
    ref_pos = zeros(num_nodes, 3)
    for i in 1:M
        for j in 1:N
            idx = (i-1)*N + j
            ref_pos[idx, 1] = Float64(i)
            ref_pos[idx, 2] = Float64(j)
        end
    end
    geometry = Theseus.GeometryData(ref_pos)
    
    # Anchors
    anchors = Theseus.AnchorInfo(
        variable_indices = Int[],
        fixed_indices = fixed_nodes,
        reference_positions = ref_pos,
        initial_variable_positions = zeros(0, 3)
    )
    
    # Objective: Target center node Z
    center_node = (M ÷ 2) * N + (N ÷ 2)
    center_idx = findfirst(==(center_node), free_nodes)
    if isnothing(center_idx)
         # fallback to some free node if center is fixed (unlikely for large grids)
         center_node = free_nodes[length(free_nodes)÷2]
    end
    
    target_pos = copy(ref_pos[center_node:center_node, :])
    target_pos[1, 3] = -5.0
    obj1 = Theseus.TargetXYZObjective(weight=100.0, node_indices=[center_node], target=target_pos)
    
    # Uniform lengths
    obj2 = Theseus.LengthVariationObjective(weight=1.0, edge_indices=collect(1:num_edges))
    
    params = Theseus.OptimizationParameters(
        objectives = [obj1, obj2],
        bounds = Theseus.Bounds(fill(0.1, num_edges), fill(20.0, num_edges)),
        solver = Theseus.SolverOptions(
            absolute_tolerance = 1e-5,
            relative_tolerance = 1e-5,
            max_iterations = 20, # Keep it short for benchmarking
            report_frequency = 100,
            show_progress = false,
            barrier_weight = 1e-3,
            barrier_sharpness = 10.0,
            use_auto_scaling = false
        ),
        tracing = Theseus.TracingOptions(record_nodes=false, emit_frequency=1)
    )
    
    return Theseus.OptimizationProblem(topo, loads, geometry, anchors, params)
end

function benchmark_loop(M, N)
    println("\n=== Benchmarking Grid $(M)x$(N) ($( (M-1)*N + M*(N-1) ) edges) ===")
    
    problem = create_grid_problem(M, N)
    initial_q = fill(1.0, problem.topology.num_edges)
    state = Theseus.OptimizationState(initial_q, zeros(0, 3))
    state.cache = Theseus.OptimizationCache(problem)
    
    θ0 = Theseus.pack_parameters(problem, state)
    
    # 1. Benchmark Objective Function
    lower_bounds, upper_bounds = Theseus.parameter_bounds(problem)
    lb_idx = findall(isfinite, lower_bounds)
    ub_idx = findall(isfinite, upper_bounds)
    objective = Theseus.form_finding_objective(
        problem, state.cache, lower_bounds, upper_bounds, 
        lb_idx, ub_idx, problem.parameters.solver.barrier_weight, 
        problem.parameters.solver.barrier_sharpness
    )
    
    t_obj = @belapsed $objective($θ0)
    a_obj = @allocated objective(θ0)
    println("Objective: $(round(t_obj*1000, digits=3)) ms ($(round(a_obj/1024, digits=1)) KB)")
    
    # 2. Benchmark Gradient (Mooncake)
    backend = ADTypes.AutoMooncake(config=Mooncake.Config(friendly_tangents=true))
    prep = DifferentiationInterface.prepare_gradient(objective, backend, θ0)
    G = similar(θ0)
    
    t_grad = @belapsed DifferentiationInterface.value_and_gradient!($objective, $G, $prep, $backend, $θ0)
    a_grad = @allocated DifferentiationInterface.value_and_gradient!(objective, G, prep, backend, θ0)
    println("Gradient:  $(round(t_grad*1000, digits=3)) ms ($(round(a_grad/1024, digits=1)) KB)")
    
    # Show breakdown for ONE gradient call
    reset_timer!(state.cache.to)
    DifferentiationInterface.value_and_gradient!(objective, G, prep, backend, θ0)
    println("\nBreakdown for ONE Gradient call:")
    show(state.cache.to)
    println("\n")
    
    # 3. Full Loop (20 iterations)
    # Warmup once
    Theseus.optimize_problem!(problem, state)
    
    # Use a fresh state for each benchmark trial to avoid compounding
    q_init = copy(initial_q)
    t_loop = @belapsed Theseus.optimize_problem!($problem, Theseus.OptimizationState(copy($q_init), zeros(0, 3)))
    println("Full Loop: $(round(t_loop*1000, digits=3)) ms (20 iters)")
    
    # Capture one granular run separately to show the breakdown
    granular_state = Theseus.OptimizationState(copy(q_init), zeros(0, 3))
    Theseus.optimize_problem!(problem, granular_state)
    println("\nGranular Breakdown (per optimize_problem! call):")
    show(granular_state.cache.to)
    println()
end

# Warmup everything first with a tiny problem
println("Warming up...")
benchmark_loop(5, 5)

# Run benchmarks
benchmark_loop(10, 10)
benchmark_loop(20, 20)
benchmark_loop(50, 50)
