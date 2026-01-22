module Theseus

using LinearAlgebra
using Statistics
using SparseArrays
using Optim
using Mooncake
using JSON3
using HTTP
using LineSearches
using ChainRulesCore
using CUDA
using LinearSolve
using LDLFactorizations
using MKL

include("types.jl")
include("kernels.jl")
include("FDM.jl")
include("optimization.jl")
include("analysis.jl")
include("communication.jl")
include("objectives.jl")
include("adjoint.jl")

export start!

function __init__()
    # Prevent thread contention during hybrid CPU/GPU execution
    MKL.set_num_threads(1)
end

end # module Theseus
