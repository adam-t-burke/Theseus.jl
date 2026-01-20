import HTTP.WebSockets

function assemble_xyz(snapshot::GeometrySnapshot)
    snapshot.xyz_full
end

function loss_trace_or_default(trace::Vector{Float64})
    isempty(trace) ? Float64[] : trace
end

function send_message(ws, problem::OptimizationProblem{T}, state::OptimizationState, snapshot::GeometrySnapshot; finished::Bool, iteration::Int, loss::Float64, include_trace::Bool) where {T}
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
        "Penaltytrace" => copy(loss_trace_or_default(state.penalty_trace)),
    )
    if include_trace
        payload["NodeTrace"] = copy(state.node_trace)
    end
    HTTP.WebSockets.send(ws, JSON3.write(payload))
end

function direct_solution!(problem::OptimizationProblem{T}, state::OptimizationState, ws) where {T}
    snapshot = evaluate_geometry!(problem, state.force_densities, state.variable_anchor_positions)
    empty!(state.loss_trace)
    push!(state.loss_trace, 0.0)
    send_message(ws, problem, state, snapshot;
        finished = true,
        iteration = 1,
        loss = 0.0,
        include_trace = false,
    )
    return snapshot
end

function FDMoptim!(problem::OptimizationProblem{T}, state::OptimizationState, ws; max_norm::Float64 = 1.0) where {T}
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

    callback_fn = solver.show_progress ? iteration_callback : nothing

    result, snapshot = optimize_problem!(problem, state; on_iteration = callback_fn)

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