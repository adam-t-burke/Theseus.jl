import HTTP.WebSockets

const MAX_STACKTRACE_LINES = 6

function format_user_hint(error::Exception)
    if error isa ErrorException
        msg = getfield(error, :msg)
        if msg == "Unhandled field storage"
            return "Automatic differentiation failed because Mooncake doesn't yet handle the `Task.storage` field introduced in Julia 1.11. Update Mooncake to a release that supports Julia 1.11 or switch Theseus to a finite-difference backend for gradients."
        end
        if startswith(msg, "Unhandled field ")
            field = strip(msg[length("Unhandled field ")+1:end])
            return "Unexpected field `$(field)` found in request JSON. Please remove or rename this field to match Theseus' schema."
        end
    end
    return nothing
end

function short_backtrace(bt)
    frames = stacktrace(bt)
    limit = min(MAX_STACKTRACE_LINES, length(frames))
    io = IOBuffer()
    for idx in 1:limit
        Base.show(io, frames[idx])
        if idx < limit
            write(io, '\n')
        end
    end
    String(take!(io))
end

function respond_with_error(ws, error, bt)
    message = sprint(showerror, error)
    hint = format_user_hint(error)
    backtrace = short_backtrace(bt)
    payload = Dict(
        "Event" => "Error",
        "Type" => string(typeof(error)),
        "Message" => message,
        "Backtrace" => backtrace,
    )
    if hint !== nothing
        payload["Hint"] = hint
    end
    HTTP.WebSockets.send(ws, JSON3.write(payload))
end

cancel = false
simulating = false
counter = 0

function start!(;host = "127.0.0.1", port = 2000)
    #start server
    println("###############################################")
    println("###############SERVER OPENED###################")
    println("###############################################")

    ## PERSISTENT LOOP
    WebSockets.listen!(host, port) do ws
        # FOR EACH MESSAGE SENT FROM CLIENT
        
        for msg in ws
            try
                @async readMSG(msg, ws)
            catch error
                println(error)
            end
        end
    end
end

function readMSG(msg, ws)
    # ACKNOWLEDGE
    println("MSG RECEIVED")

    # FIRST MESSAGE
    if msg == "init"
        println("CONNECTION INITIALIZED")
        return
    end

    if msg == "cancel"
        println("Operation Cancelled")
        global cancel = true
        return
    end

    if simulating
        println("Simulation in progress")
        return
    end

    # ANALYSIS
    try
        problem_json = JSON3.read(msg)
        problem, state = build_problem(problem_json)

        if isempty(problem.parameters.objectives)
            println("SOLVING")
        else
            println("OPTIMIZING")
        end

        if counter == 0
            println("First run will take a while.")
            println("Julia needs to compile the code for the first run.")
        end
        
        # OPTIMIZATION
        global simulating = true
        @time FDMoptim!(problem, state, ws)
       
    catch error
        bt = catch_backtrace()
        println("REQUEST FAILED")
        println("Message: ", sprint(showerror, error))
        hint = format_user_hint(error)
        if hint !== nothing
            println("Hint: $hint")
        end
        println("Stacktrace:")
        println(short_backtrace(bt))
        respond_with_error(ws, error, bt)
    end

    println("DONE")
    global simulating = false
    global counter += 1
    println("Counter $counter")
end
