cancel = false
simulating = false
counter = 0
shutdown = false  # Add shutdown flag
server_ref = Ref{Any}()  # Store server reference

function start!(;host = "127.0.0.1", port = 2000)
    #start server
    println("###############################################")
    println("###############SERVER OPENED###################")
    println("###############################################")

    global shutdown = false

    ## PERSISTENT LOOP
    server = WebSockets.listen!(host, port) do ws
        # FOR EACH MESSAGE SENT FROM CLIENT
        
        for msg in ws
            # Check shutdown flag
            if shutdown
                println("Server shutting down...")
                break
            end

            try
            @async readMSG(msg, ws)
            catch error
                println(error)
            end
        end
    end

    # Store server reference for shutdown
    server_ref[] = server
    return server
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

        if simulating == true
            println("Simulation in progress")
            return
        end

        # ANALYSIS
        try
            # DESERIALIZE MESSAGE
            problem = J3.read(msg)
            println(J3.pretty(problem))

            # MAIN ALGORITHM
            println("READING DATA")

            # CONVERT MESSAGE TO RECEIVER TYPE
            receiver = Receiver(problem)

            # SOLVE
            if counter == 0
                println("First run will take a while.")
                println("Julia needs to compile the code for the first run.")
            end
            
            # OPTIMIZATION
            global simulating = true
            @time FDMoptim!(receiver, ws)
           
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

# Add shutdown function
function shutdown_server!()
    println("Initiating server shutdown...")
    global shutdown = true
    
    # Close the server if it exists
    if isassigned(server_ref) && server_ref[] !== nothing
        try
            close(server_ref[])
            println("Server closed successfully")
        catch e
            println("Error closing server: $e")
        end
    end
    
    # Reset flags
    global cancel = false
    global simulating = false
    global counter = 0
end

