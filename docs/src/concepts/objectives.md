# Objectives

Theseus supports multiple objective types that can be combined to define complex optimization goals. Each objective has a **weight** that controls its relative importance in the total loss function.

## Objective Categories

Objectives fall into several categories:

| Category | Purpose |
|----------|---------|
| **Position targets** | Drive nodes toward desired locations |
| **Length objectives** | Control member lengths |
| **Force objectives** | Control member forces |
| **Uniformity** | Minimize variation across members |
| **Reaction objectives** | Control support reactions |

## Position Objectives

### TargetXYZObjective

Penalizes deviation of free nodes from target 3D positions.

```
Loss = Σᵢ ||xᵢ - targetᵢ||²
```

**Use case**: Primary form-finding objective when you have specific target positions for nodes.

**Fields**:
- `weight`: Relative importance
- `node_indices`: Which nodes to constrain
- `target`: N×3 matrix of target positions

### TargetXYObjective

Same as `TargetXYZObjective` but only considers X and Y coordinates (ignores Z).

**Use case**: Planar constraints where vertical position should be determined by equilibrium.

### RigidSetCompareObjective

Compares a set of node positions to a rigid reference shape, allowing rotation and translation.

**Use case**: Matching overall form shape while allowing rigid body motion.

## Length Objectives

### TargetLengthObjective

Penalizes deviation of member lengths from target values.

```
Loss = Σᵢ (Lᵢ - targetᵢ)²
```

**Use case**: When specific member lengths are required (e.g., for fabrication).

### MinLengthObjective

Soft constraint enforcing minimum member lengths using a smooth barrier function.

```
Loss = Σᵢ softplus(threshold - Lᵢ, 0, k)
```

**Use case**: Preventing members from becoming too short (e.g., minimum cable length for connections).

### MaxLengthObjective

Soft constraint enforcing maximum member lengths.

**Use case**: Material limitations, transportation constraints.

### LengthVariationObjective

Minimizes variance in member lengths.

```
Loss = Var(L) = Σᵢ (Lᵢ - L̄)²
```

**Use case**: Uniform member sizing for fabrication efficiency or aesthetics.

## Force Objectives

### MinForceObjective

Soft constraint enforcing minimum member forces.

**Use case**: Ensuring cables remain in tension (positive force).

### MaxForceObjective

Soft constraint enforcing maximum member forces.

**Use case**: Structural capacity limits, preventing overstress.

### ForceVariationObjective

Minimizes variance in member forces.

**Use case**: Uniform stress distribution for structural efficiency.

### SumForceLengthObjective

Minimizes the sum of force times length (related to structural weight/volume).

```
Loss = Σᵢ Fᵢ · Lᵢ
```

**Use case**: Material efficiency optimization.

## Reaction Objectives

### ReactionDirectionObjective

Penalizes deviation of anchor reaction forces from target directions.

**Use case**: Constraining reaction force directions for foundation design.

**Fields**:
- `anchor_indices`: Which anchors to constrain
- `target_directions`: Unit direction vectors (N×3)

### ReactionDirectionMagnitudeObjective

Penalizes deviation of anchor reactions from target directions AND magnitudes.

**Use case**: When both direction and magnitude of reactions are constrained (e.g., matching foundation capacity).

## Combining Objectives

Multiple objectives can be combined by adding them to the `objectives` list. The total loss is:

```
L_total = Σᵢ wᵢ · Lᵢ
```

### Weight Selection Guidelines

- Start with equal weights and adjust based on results
- Scale weights so objectives have similar magnitudes
- Higher weights = stronger enforcement
- Use barrier objectives (Min/Max) with moderate weights to avoid numerical issues

### Example Combinations

**Form-finding with target shape**:
- `TargetXYZObjective` (weight: 1.0) - primary shape goal
- `MinForceObjective` (weight: 0.1) - ensure tension

**Uniform cable network**:
- `TargetXYZObjective` (weight: 1.0) - shape goal
- `LengthVariationObjective` (weight: 0.5) - uniform lengths
- `ForceVariationObjective` (weight: 0.5) - uniform forces

**Constrained optimization**:
- `TargetXYZObjective` (weight: 1.0) - shape goal
- `MinLengthObjective` (weight: 10.0) - hard minimum length
- `MaxForceObjective` (weight: 10.0) - hard maximum force

## Barrier Parameters

Barrier-based objectives (`MinLengthObjective`, `MaxLengthObjective`, `MinForceObjective`, `MaxForceObjective`) have a `sharpness` parameter:

- **Low sharpness** (1-5): Gradual penalty, easier optimization
- **Default sharpness** (10): Good balance
- **High sharpness** (20+): Sharp cutoff, may cause convergence issues

The global `barrier_weight` in `SolverOptions` scales all barrier penalties together.
