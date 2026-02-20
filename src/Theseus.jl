"""
    Theseus

A Julia package for form-finding and optimization of tensile structures, compression structures,
and mixed tension/compression structures using the Force Density Method (FDM).

Theseus is designed as a server component that communicates with 
[Ariadne](https://github.com/fibrous-tendencies/Ariadne), a Grasshopper plugin for Rhino3D,
via WebSockets. It receives network topology and optimization parameters from Ariadne,
performs form-finding or optimization using automatic differentiation and L-BFGS,
and returns the resulting geometry.

# Quick Start

```julia
using Theseus
start!()  # Starts the WebSocket server on localhost:2000
```

# Features

- **Force Density Method**: Efficient linear solver for cable/tensile network equilibrium
- **Gradient-based Optimization**: L-BFGS optimization with automatic differentiation via Mooncake.jl
- **Multiple Objectives**: Support for 13+ objective types including target positions, length/force constraints
- **Real-time Communication**: WebSocket server for interactive design with Grasshopper/Rhino

# Exports

- [`start!`](@ref): Start the WebSocket server

See the [documentation](https://adam-t-burke.github.io/Theseus.jl/) for more details.
"""
module Theseus

using LinearAlgebra
using SparseArrays
using Optim
using Mooncake
using ADTypes
using DifferentiationInterface
using LDLFactorizations
using JSON3
using HTTP

include("types.jl")
include("objectives.jl")
include("FDM.jl")
include("adjoint.jl")
include("optimization.jl")
include("analysis.jl")
include("communication.jl")

export start!

end # module Theseus
