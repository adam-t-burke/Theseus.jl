module Theseus

using LinearAlgebra
using Statistics
using SparseArrays
using Optim
using DifferentiationInterface
using Mooncake: AutoMooncake
using JSON3
using HTTP
using LineSearches

include("types.jl")
include("FDM.jl")
include("optimization.jl")
include("analysis.jl")
include("communication.jl")
include("objectives.jl")
include("anchors.jl")

export start!

end # module FDMremote
