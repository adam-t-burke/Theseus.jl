module Theseus

using LinearAlgebra
using Statistics
using SparseArrays
using Optim
using JSON3
using HTTP
using DifferentiationInterface
using Mooncake

include("FDM.jl")
include("types.jl")
include("optimization.jl")
include("analysis.jl")
include("communication.jl")
include("objectives.jl")
include("anchors.jl")
include("utils.jl")

export start!, shutdown_server!

end # module FDMremote
