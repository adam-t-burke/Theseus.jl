# Force Density Method

The **Force Density Method (FDM)** is a linear approach to form-finding for cable and strut networks. It was developed in the 1970s and remains one of the most efficient methods for computing equilibrium shapes of tensile structures.

## Basic Concept

In a cable network under load, each cable carries a certain tension force. The **force density** `q` of a cable is defined as:

```math
q = \frac{F}{L}
```

where `F` is the axial force and `L` is the length of the cable.

The key insight of FDM is that if we **prescribe the force densities** rather than the forces themselves, the equilibrium equations become linear and can be solved directly.

## Mathematical Formulation

### Network Representation

A cable network is represented as a graph with:
- **Nodes**: Points where cables meet (some free to move, some fixed as anchors)
- **Edges**: Cables connecting nodes

The network topology is encoded in an **incidence matrix** `C`, where each row represents an edge and each column represents a node. For edge `i` connecting nodes `j` and `k`:
- `C[i,j] = -1` (start node)
- `C[i,k] = +1` (end node)

### Equilibrium Equations

For a network with force densities `q`, fixed node positions `x_f`, and external loads `P`, the equilibrium positions `x` of free nodes satisfy:

```math
(C_n^T \cdot Q \cdot C_n) \cdot x = P - C_n^T \cdot Q \cdot C_f \cdot x_f
```

where:
- `C_n` is the incidence matrix columns for free nodes
- `C_f` is the incidence matrix columns for fixed nodes
- `Q = diag(q)` is a diagonal matrix of force densities

This is a **linear system** `A·x = b` that can be solved efficiently using sparse LDL factorization.

## Advantages of FDM

1. **Linear**: No iterative solver needed for form-finding
2. **Fast**: Sparse matrix factorization is highly efficient
3. **Stable**: Always converges to a valid equilibrium (if one exists)
4. **Differentiable**: Smooth gradients for optimization

## Theseus Implementation

Theseus uses FDM as the core solver within an optimization loop:

1. Given force densities `q`, solve the linear system to get node positions
2. Compute objective function value from the geometry
3. Use automatic differentiation to get gradients
4. Update force densities using L-BFGS
5. Repeat until convergence

The [`solve_FDM!`](@ref) function implements the linear solve, with custom adjoint rules in `adjoint.jl` for efficient gradient computation through the factorization.

## References

- Schek, H.J. (1974). "The force density method for form finding and computation of general networks." *Computer Methods in Applied Mechanics and Engineering*, 3(1), 115-134.
- Linkwitz, K., & Schek, H.J. (1971). "Einige Bemerkungen zur Berechnung von vorgespannten Seilnetzkonstruktionen." *Ingenieur-Archiv*, 40(3), 145-158.
