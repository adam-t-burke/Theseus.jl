struct Objective
    ID::Int64
    W::Float64
    Indices::Union{Vector{Int64}, Int64}
    Values::Union{Vector{Float64}, Matrix{Float64}, Float64, Nothing}

    function Objective(obj::JSON3.Object, ne::Int64, nn::Int64, N, F)
        id = obj["OBJID"]
        w = Float64(obj["Weight"])

        #If the objective is a global objective, then the indices are -1
        if  obj["Indices"][1] == -1
            if id == 1
                indices = N
            elseif id == 0
                indices = F
            else
                indices = collect(Int64, range(1, ne))
            end        
        else
            indices = Int64.(obj["Indices"]) .+ 1
        end

        #Vector form of values
        if haskey(obj, "Values")
            if obj["Indices"][1] == -1
                values = ones(ne) .* Float64.(obj["Values"])
            elseif length(obj["Values"]) == length(obj["Indices"])
                values = Float64.(obj["Values"])
            elseif length(obj["Values"]) == 1 && length(obj["Indices"]) > 1
                values = ones(length(obj["Indices"])) .* Float64.(obj["Values"][1])
            else
                println("Warning: Number of values must match number of indices or be 1.")
                println("Using first value for all values.")
                values = ones(ne) .* Float64.(obj["Values"][1])
            end
        
        #Matrix form of values. All point based objectives have more than one value and 
        #typically not repeated values, so we assume that the values are given.
        elseif haskey(obj, "Points")
            values = obj["Points"]
            values = reduce(hcat, values)
            values = convert(Matrix{Float64}, values')
        
            
        #Some objectives don't have values.
        else
                values = nothing
        end


        new(id, w, indices, values)
    end        
end

struct Parameters
    #OBJECTIVES
    Objectives::Vector{Objective}

    #PARAMETERS
    AbsTol::Float64
    RelTol::Float64
    Freq::Int64
    MaxIter::Int64
    Show::Bool

    #DERIVED VALUES
    UB::Vector{Float64}
    LB::Vector{Float64}

    NodeTrace::Bool

    function Objectives(objs::JSON3.Array, ne, nn, N, F)
        objectives = [Objective(obj, ne, nn, N, F) for obj in objs]
        return objectives
    end

    function Parameters(parameters::JSON3.Object, ne::Int64, nn::Int64, N::Vector{Int64}, F::Vector{Int64})
        objectives = Objectives(parameters["Objectives"], ne, nn, N, F)
        abstol = Float64(parameters["AbsTol"])
        reltol = Float64(parameters["RelTol"])
        freq = Int64(parameters["UpdateFrequency"])
        maxiter = Int64(parameters["MaxIterations"])
        show = Bool(parameters["ShowIterations"])

        ub = Float64.(parameters["UpperBound"])
        lb = Float64.(parameters["LowerBound"])

        nodeTrace = parameters["NodeTrace"]

        return new(
            objectives,
            abstol,
            reltol,
            freq,
            maxiter,
            show,
            ub,
            lb,
            nodeTrace)
    end
end

struct AnchorParameters
    VAI::Vector{Int64}
    FAI::Vector{Int64}

    Init::Vector{Float64}

    function AnchorParameters(anchors::Vector{Any}, var::Vector{Any}, fix::Vector{Any})
        variableAnchors = Int64.(var) .+ 1
        fixedAnchors = Int64.(fix) .+ 1

        init = []

        for anchor in anchors
            push!(init, anchor["InitialX"])
            push!(init, anchor["InitialY"])
            push!(init, anchor["InitialZ"])
        end

        return new(
            variableAnchors,
            fixedAnchors,
            init)
    end
end

struct Receiver

    #FORCE DENSITY
    Q::Vector{Float64}

    #NETWORK INFORMATION
    N::Vector{Int64}
    F::Vector{Int64}

    XYZf::Matrix{Float64}
    Pn::Matrix{Float64}

    C::SparseMatrixCSC{Int64, Int64}
    Cn::SparseMatrixCSC{Int64, Int64}
    Cf::SparseMatrixCSC{Int64, Int64}

    ne::Int64
    nn::Int64

    Params::Union{Parameters, Nothing}
    
    AnchorParams::Union{AnchorParameters, Nothing}


    #constructor
    function Receiver(problem::JSON3.Object)
        #anchor geometry
        xyzf = problem["XYZf"]
        xyzf = reduce(hcat, xyzf)
        xyzf = convert(Matrix{Float64}, xyzf')

        # global info
        ne = Int(problem["Network"]["Graph"]["Ne"])
        nn = Int(problem["Network"]["Graph"]["Nn"])

        # initial force densities
        if length(problem["Q"]) == 1
            q = Float64.(repeat(problem["Q"], ne))
        elseif length(problem["Q"]) == ne
            q = Float64.(problem["Q"])
        else
            q = repeat(1.0, ne)
        end        

        # free/fixed
        N = Int.(problem["Network"]["FreeNodes"]) .+ 1
        N = collect(range(1, length = length(N)))
        F = Int.(problem["Network"]["FixedNodes"]) .+ 1 
        F = collect(range(length(N)+1, length = length(F)))   

        # loads
        GH_p = problem["P"]
        GH_p = reduce(hcat, GH_p)
        GH_p = convert(Matrix{Float64}, GH_p')

        LN = Int[]
        # load nodes
        if haskey(problem, "LoadNodes")
            LN = Int.(problem["LoadNodes"]) .+ 1     
            #If the number of given load vectors matches the number of 
            #node indices given in the problem, then we assume that the
            #load vectors are given in the same order as the node indices.
            if length(LN) == size(GH_p, 1)
                p = zeros(nn, 3)
                p[LN, :] = GH_p
            elseif length(LN) > 1 && size(GH_p, 1) == 1         
                p = zeros(nn, 3)
                p[LN, :] = repeat(GH_p, length(LN))
            else
                p = zeros(nn, 3)
                println("Warning: Number of load vectors must match number of load nodes or be 1.")
                println("Using zero loads.")
            end
            Pn = p[N, :]
        else
            if size(GH_p, 1) == 1
                p = repeat(GH_p, length(N))
            elseif size(GH_p, 1) == length(N)
                p = GH_p
            else
                println("Warning: Number of load vectors must match number of load nodes or be 1.")
                println("Using zero loads.")
                p = zeros(length(N), 3)
            end
            Pn = p      
        end

        # connectivity
        i = Int.(problem["I"]) .+ 1
        j = Int.(problem["J"]) .+ 1
        v = Int.(problem["V"])

        C = sparse(i, j, v, ne, nn)
        Cn = C[:, 1:length(N)]
        Cf = C[:, length(N)+1:end]

        if haskey(problem, "Parameters")
            p = Parameters(problem["Parameters"], ne, nn, N, F)

            #prevent the force densities from being outside the bounds
            q = clamp(q, p.LB, p.UB)
        else
            println("No optimization parameters provided.")
            println("Running FDM using given force densities only.")
            p = nothing
        end

        if haskey(problem, "VariableAnchors")
            varAnchors = problem["NodeIndex"]
            fixAnchors = problem["FixedAnchorIndices"]
            ap = AnchorParameters(problem["VariableAnchors"], varAnchors, fixAnchors)
        else
            println("No anchor parameters provided.")
            println("Using only given fixed node positions.")
            ap = nothing
        end

        

        return new(
            q, 
            N, 
            F,
            xyzf, 
            Pn,
            C,
            Cn,
            Cf,
            ne,
            nn,
            p,
            ap)        
    end
end