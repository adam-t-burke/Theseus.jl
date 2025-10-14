import HTTP.WebSockets

function assemble_xyz(snapshot::GeometrySnapshot)
    snapshot.xyz_full
end

function loss_trace_or_default(trace::Vector{Float64})
    isempty(trace) ? Float64[] : trace
end

function send_message(ws, problem::OptimizationProblem, state::OptimizationState, snapshot::GeometrySnapshot; finished::Bool, iteration::Int, loss::Float64, include_trace::Bool)
    xyz = assemble_xyz(snapshot)
    payload = Dict(
        "Finished" => finished,
        "Iter" => iteration,
        "Loss" => loss,
        "Q" => state.force_densities,
        "X" => xyz[:, 1],
        "Y" => xyz[:, 2],
        "Z" => xyz[:, 3],
        "Losstrace" => copy(loss_trace_or_default(state.loss_trace)),
    )
    if include_trace
        payload["NodeTrace"] = copy(state.node_trace)
    end
    HTTP.WebSockets.send(ws, JSON3.write(payload))
end

function direct_solution!(problem::OptimizationProblem, state::OptimizationState, ws)
    empty!(state.loss_trace)
    empty!(state.node_trace)
    state.iterations = 1
    snapshot = evaluate_geometry!(state.workspace, problem, state.force_densities, state.variable_anchor_positions)
    push!(state.loss_trace, 0.0)
    send_message(ws, problem, state, snapshot;
        finished = true,
        iteration = state.iterations,
        loss = 0.0,
        include_trace = false,
    )
    return snapshot
end

function FDMoptim!(problem::OptimizationProblem, state::OptimizationState, ws; max_norm::Float64 = 1.0)
    if isempty(problem.parameters.objectives)
        return direct_solution!(problem, state, ws)
    end

    solver = problem.parameters.solver
    trace_opts = problem.parameters.tracing

    function iteration_callback(current_state, snapshot, loss)
        if solver.show_progress && (current_state.iterations % max(1, solver.report_frequency) == 0)
            send_message(ws, problem, current_state, snapshot;
                finished = false,
                iteration = current_state.iterations,
                loss = loss,
                include_trace = trace_opts.record_nodes,
            )
        end
    end

    result, snapshot = optimize_problem!(problem, state; on_iteration = iteration_callback)

    final_loss = Optim.minimum(result)
    total_iters = Optim.iterations(result)

    send_message(ws, problem, state, snapshot;
        finished = true,
        iteration = total_iters,
        loss = final_loss,
        include_trace = trace_opts.record_nodes,
    )

    return snapshot
end