# Optimization

Theseus uses gradient-based optimization to find force densities that minimize a combination of objective functions while satisfying constraints. This page explains the optimization approach.

## Overview

The optimization problem in Theseus is:

```
minimize    L(q) = Σᵢ wᵢ · fᵢ(geometry(q))
subject to  q_lower ≤ q ≤ q_upper
```

where:
- `q` is the vector of force densities (optimization variables)
- `geometry(q)` computes equilibrium node positions via FDM
- `fᵢ` are objective functions evaluated on the geometry
- `wᵢ` are weights for each objective
- `q_lower`, `q_upper` are bounds on force densities

## L-BFGS Optimizer

Theseus uses the **Limited-memory BFGS (L-BFGS)** algorithm from [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl). L-BFGS is a quasi-Newton method that:

- Approximates the inverse Hessian using gradient history
- Requires only gradient (not Hessian) evaluations
- Has low memory footprint for large problems
- Converges quickly for smooth, well-conditioned problems

## Automatic Differentiation

Computing gradients efficiently is critical for optimization. Theseus uses **reverse-mode automatic differentiation (AD)** via [Mooncake.jl](https://github.com/compintell/Mooncake.jl).

### How it Works

1. The forward pass computes the objective value
2. The backward pass propagates gradients through all operations
3. Custom adjoint rules in `adjoint.jl` handle the FDM linear solve efficiently

### Custom Adjoint Rules

The FDM involves solving a sparse linear system:
```
A · x = b
```

Rather than differentiating through the factorization algorithm, Theseus uses the **adjoint method**:

```
∂L/∂q = (∂A/∂q)ᵀ · λ · x
```

where `λ` solves the adjoint system:
```
Aᵀ · λ = ∂L/∂x
```

This is much more efficient than naive AD through the factorization.

## Barrier Functions for Constraints

Instead of using projected gradients or penalty methods with discontinuous derivatives, Theseus uses **smooth barrier functions** for bound constraints:

```julia
softplus(x, b, k) = log(1 + exp(k * (x - b))) / k
```

This provides:
- Smooth gradients everywhere (good for AD)
- Increasing penalty as variables approach bounds
- Configurable sharpness via the `k` parameter

## Solver Options

The optimization can be configured via [`SolverOptions`](@ref):

| Option | Description | Default |
|--------|-------------|---------|
| `max_iterations` | Maximum optimization iterations | 500 |
| `absolute_tolerance` | Convergence tolerance (absolute) | 1e-6 |
| `relative_tolerance` | Convergence tolerance (relative) | 1e-6 |
| `barrier_weight` | Weight for bound penalty terms | 1000.0 |
| `barrier_sharpness` | Sharpness of barrier functions | 10.0 |
| `use_auto_scaling` | Auto-scale the problem | true |

## Convergence

The optimizer stops when any of these conditions are met:
1. Gradient norm falls below tolerance
2. Objective change falls below tolerance
3. Maximum iterations reached

For complex problems with many objectives, you may need to:
- Increase `max_iterations`
- Adjust objective weights for better conditioning
- Tune barrier parameters if bounds are important
