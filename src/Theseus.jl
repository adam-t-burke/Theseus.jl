module Theseus

using LinearAlgebra
using Statistics
using SparseArrays
using Optim
using Zygote
using Zygote: @adjoint
using JSON3
using HTTP
using LineSearches
using ChainRulesCore
using DifferentiationInterface
using Mooncake

const J3 = JSON3
const DI = DifferentiationInterface

include("FDM.jl")
include("types.jl")
include("optimization.jl")
include("analysis.jl")
include("communication.jl")
include("objectives.jl")
include("adjoint.jl")
include("anchors.jl")

export start!, kill!

end # module FDMremote
