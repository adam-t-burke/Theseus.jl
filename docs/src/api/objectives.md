# Objective Types

This page documents all objective types available in Theseus.

## Abstract Type

```@docs
AbstractObjective
```

## Position Objectives

### TargetXYZObjective

```@docs
TargetXYZObjective
```

### TargetXYObjective

```@docs
TargetXYObjective
```

### RigidSetCompareObjective

```@docs
RigidSetCompareObjective
```

## Length Objectives

### TargetLengthObjective

```@docs
TargetLengthObjective
```

### LengthVariationObjective

```@docs
LengthVariationObjective
```

### MinLengthObjective

```@docs
MinLengthObjective
```

### MaxLengthObjective

```@docs
MaxLengthObjective
```

## Force Objectives

### ForceVariationObjective

```@docs
ForceVariationObjective
```

### SumForceLengthObjective

```@docs
SumForceLengthObjective
```

### MinForceObjective

```@docs
MinForceObjective
```

### MaxForceObjective

```@docs
MaxForceObjective
```

## Reaction Objectives

### ReactionDirectionObjective

```@docs
ReactionDirectionObjective
```

### ReactionDirectionMagnitudeObjective

```@docs
ReactionDirectionMagnitudeObjective
```

## Objective Functions

The following functions compute objective losses from the `objectives.jl` file:

### softplus

```@docs
softplus
```

### minPenalty

```@docs
minPenalty
```

### maxPenalty

```@docs
maxPenalty
```

### target_xyz

```@docs
target_xyz
```

### target_xy

```@docs
target_xy
```

### lenVar

```@docs
lenVar
```

### forceVar

```@docs
forceVar
```

### lenTarget

```@docs
lenTarget
```
