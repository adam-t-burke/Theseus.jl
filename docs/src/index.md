# Theseus.jl

Theseus is a [Julia](https://julialang.org/) package for **form-finding and optimization of tensile structures**, compression structures, and mixed tension/compression structures using the **Force Density Method (FDM)**.

It is designed as a computational backend for [Ariadne](https://github.com/fibrous-tendencies/Ariadne), a Grasshopper plugin for Rhino3D, enabling real-time structural optimization in an interactive design environment.

## Key Features

- **Force Density Method**: Efficient linear solver for computing equilibrium shapes of cable and strut networks
- **Gradient-Based Optimization**: L-BFGS optimization with automatic differentiation via [Mooncake.jl](https://github.com/compintell/Mooncake.jl)
- **Multiple Objective Types**: 13+ objective functions including target positions, length/force constraints, and reaction force objectives
- **Real-Time Communication**: WebSocket server for interactive design with Grasshopper/Rhino

## Architecture

Theseus operates as a server that communicates with the Ariadne Grasshopper plugin via WebSockets:

```
┌─────────────────────────────────────────────────────────────┐
│                  Grasshopper / Rhino3D                      │
│                    (Ariadne Plugin)                         │
└─────────────────────────────────────────────────────────────┘
                            │ WebSocket (JSON)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      Theseus.jl Server                      │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   Parser    │───▶│  Optimizer  │───▶│   Solver    │     │
│  │  (types.jl) │    │(optimization│    │  (FDM.jl)   │     │
│  └─────────────┘    │    .jl)     │    └─────────────┘     │
│                     └─────────────┘           │            │
│                           │                   ▼            │
│                     ┌─────────────┐    ┌─────────────┐     │
│                     │ Objectives  │    │  Autodiff   │     │
│                     │(objectives  │    │ (adjoint.jl)│     │
│                     │    .jl)     │    └─────────────┘     │
│                     └─────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

1. **Julia**: Download and install Julia from [julialang.org/downloads](https://julialang.org/downloads/). Theseus requires Julia 1.9 or later (1.11+ recommended).

2. **Ariadne** (optional but typical): Install the [Ariadne Grasshopper plugin](https://github.com/fibrous-tendencies/Ariadne) to use Theseus interactively.

### Installing Theseus

1. Open the Julia REPL

2. Enter package mode by pressing `]`

3. Create and activate a new environment (recommended):
   ```julia
   activate Theseus
   ```

4. Install the package:
   ```julia
   add https://github.com/adam-t-burke/Theseus.jl
   ```

5. Press `Backspace` to exit package mode

## Starting the Server

Each time you want to use Theseus:

1. Open the Julia REPL

2. Activate your environment:
   ```julia
   ] activate Theseus
   ```
   (Press `Backspace` to exit package mode)

3. Load and start Theseus:
   ```julia
   using Theseus
   start!()
   ```

You should see:
```
[ Info: Theseus server listening host = "127.0.0.1" port = 2000
```

The server is now ready to receive connections from Ariadne.

### Custom Port

To use a different port:
```julia
start!(port=3000)
```

## Next Steps

- **[Concepts](@ref)**: Learn about the Force Density Method, optimization approach, and available objectives
- **[API Reference](@ref)**: Detailed documentation of types and functions

## Citation

If you use Theseus in your research, please cite it appropriately. See the [CITATION.cff](https://github.com/adam-t-burke/Theseus.jl/blob/main/CITATION.cff) file for details.

## License

Theseus is open source software. See the [LICENSE](https://github.com/adam-t-burke/Theseus.jl/blob/main/LICENSE) file for details.
