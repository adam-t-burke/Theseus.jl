function obj(q)   

    xyznew = solve_explicit(q, receiver.Cn, receiver.Cf, receiver.Pn, receiver.XYZf)
    
    xyzfull = vcat(xyznew, receiver.XYZf)  
    
    lengths = norm.(eachrow(receiver.C * xyzfull))
    forces = q .* lengths        

    loss = lossFunc(xyznew, lengths, forces, receiver, q)
    
    return loss
end

begin
    ###Define a simple problem
    
end

function optimize_with_mooncake()

    
            """
            Mooncake setup
            """
            backend = AutoMooncake(; config=nothing)
            prep = DI.prepare_gradient(obj, backend, parameters)
            grad = similar(parameters)
            DI.gradient!(obj, grad, prep, backend, parameters)

            println(grad)

end