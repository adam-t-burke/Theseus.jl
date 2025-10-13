import HTTP.WebSockets

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
        println("INVALID INPUT")
        println("CHECK PARAMETER BOUNDS")
        println(error)
    end

    println("DONE")
    global simulating = false
    global counter += 1
    println("Counter $counter")
end
