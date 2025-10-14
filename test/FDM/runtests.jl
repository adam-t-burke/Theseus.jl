using Test
using Theseus
using Optim
using SparseArrays
using ChainRulesCore
using JSON3
using HTTP

mutable struct DummyWebSocket
    messages::Vector{String}
end

@testset "optimize_problem! decreases loss" begin
    json_path = joinpath(@__DIR__, "..", "..", "examples", "test_network.json")
    problem_json = JSON3.read(read(json_path, String))
    problem, state = Theseus.build_problem(problem_json)

    θ0 = Theseus.pack_parameters(problem, state)
    initial_loss = Theseus.objective_core!(state.workspace, problem, θ0)

    result, _ = Theseus.optimize_problem!(problem, state)
    final_loss = Optim.minimum(result)

    @test final_loss <= initial_loss
    @test !isempty(state.loss_trace)
    @test state.loss_trace[end] ≈ final_loss atol=1e-8
end

DummyWebSocket() = DummyWebSocket(String[])

function HTTP.WebSockets.send(ws::DummyWebSocket, data)
    push!(ws.messages, String(data))
    nothing
end

@testset "solve_explicit gradient" begin
    incidence = sparse([1, 1], [1, 2], Int[1, -1], 1, 2)
    free_incidence = sparse([1], [1], Int[1], 1, 1)
    fixed_incidence = sparse([1], [1], Int[-1], 1, 1)
    topo = Theseus.NetworkTopology(
        incidence,
        free_incidence,
        fixed_incidence,
        1,
        2,
        [1],
        [2],
    )

    loads = Theseus.LoadData(reshape([0.0, 0.0, -1.0], 1, 3))
    fixed_positions = reshape([0.0, 0.0, 1.0], 1, 3)
    q0 = [2.0]

    solver = Theseus.create_fdmsolver_workspace(topo; ndims=3)
    primal, pullback = ChainRulesCore.rrule(Theseus.solve_explicit, solver, q0, loads, fixed_positions)
    cotangent = 2 .* primal
    _, _, grad_q, _, _ = pullback(cotangent)

    h = 1e-6
    function objective_scalar(qval)
        local_solver = Theseus.create_fdmsolver_workspace(topo; ndims=3)
        Theseus.solve_explicit(local_solver, [qval], loads, fixed_positions)
        sum(abs2, local_solver.solution)
    end

    fp = objective_scalar(q0[1] + h)
    fm = objective_scalar(q0[1] - h)
    fd = (fp - fm) / (2h)

    @test isapprox(grad_q[1], fd; atol=1e-6, rtol=1e-6)
end

