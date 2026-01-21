module Theseus

using LinearAlgebra
using Statistics
using SparseArrays
using Optim
using Mooncake
using LinearSolve
using LDLFactorizations
using SparseDiffTools
using MKL
using JSON3
using HTTP
using LineSearches
using ChainRulesCore

function __init__()
    if isdefined(Main, :MKL) || (isdefined(Base, :MKL) && Base.MKL isa Module)
        MKL.set_num_threads(1)
    end
end

include("types.jl")
include("FDM.jl")
include("optimization.jl")
include("analysis.jl")
include("communication.jl")
include("objectives.jl")
include("adjoint.jl")

export start!

end # module Theseus
