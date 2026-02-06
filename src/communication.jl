import HTTP.WebSockets

cancel = false
simulating = false
counter = 0

function start!(;host = "127.0.0.1", port = 2000)
    #start server
    @info "Theseus server listening" host port

    ## PERSISTENT LOOP
    WebSockets.listen!(host, port) do ws
        # FOR EACH MESSAGE SENT FROM CLIENT
        
        for msg in ws
            try
                @async readMSG(msg, ws)
            catch error
                @error "WebSocket handler failure" exception=(error, catch_backtrace())
            end
        end
    end
end

function readMSG(msg, ws)
    # ACKNOWLEDGE
    @debug "Message received"

    # FIRST MESSAGE
    if msg == "init"
        @info "Client connection initialized"
        return
    end

    if msg == "cancel"
        @info "Cancellation requested"
        global cancel = true
        return
    end

    if simulating
        @warn "Simulation already in progress"
        return
    end

    # ANALYSIS
    try
        problem_json = JSON3.read(msg)
        problem, state = build_problem(problem_json)

        if isempty(problem.parameters.objectives)
            @info "Running direct solution"
        else
            @info "Running optimization"
        end

        if counter == 0
            @info "Initial compile will add latency"
        end
        
        # OPTIMIZATION
        global simulating = true
        elapsed = @elapsed FDMoptim!(problem, state, ws)
        @info "Simulation finished" elapsed=elapsed
       
    catch error
        @error "Invalid input" exception=(error, catch_backtrace())
    end

    global simulating = false
    global counter += 1
    @info "Session complete" counter=counter
end
