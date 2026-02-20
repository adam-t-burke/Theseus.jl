import HTTP.WebSockets

cancel = false
simulating = false
counter = 0

"""
    start!(; host="127.0.0.1", port=2000)

Start the Theseus WebSocket server.

This function starts a persistent WebSocket server that listens for connections from
the Ariadne Grasshopper plugin. The server handles incoming messages containing
network topology and optimization parameters, runs form-finding or optimization,
and returns the resulting geometry.

# Arguments
- `host::String="127.0.0.1"`: The hostname to bind the server to.
- `port::Int=2000`: The port number to listen on.

# Example
```julia
using Theseus
start!()  # Starts server on localhost:2000
start!(port=3000)  # Use a different port
```

# Protocol
The server expects JSON messages from the client with the following structure:
- `"init"`: Initialize the connection
- `"cancel"`: Cancel the current optimization
- JSON object: Network topology and optimization parameters

Results are sent back as JSON containing node positions, member forces, and other
optimization results.

See also: [`Ariadne`](https://github.com/fibrous-tendencies/Ariadne)
"""
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

"""
    readMSG(msg, ws)

Internal function to process incoming WebSocket messages.

Handles three types of messages:
- `"init"`: Connection initialization acknowledgment
- `"cancel"`: Request to cancel current optimization
- JSON object: Problem definition to solve

When a valid JSON problem is received, constructs an [`OptimizationProblem`](@ref)
and [`OptimizationState`](@ref), then runs form-finding or optimization via
[`FDMoptim!`](@ref).
"""
function readMSG(msg, ws)
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
