//! Objective loss functions — pure ℝ → ℝ math, no AD.
//!
//! Mirrors `src/objectives.jl` from the Julia code.  Each function computes
//! a scalar loss from the current geometry snapshot.  The corresponding
//! hand-coded gradients live in `gradients.rs`.

use crate::types::{GeometrySnapshot, Objective};
use ndarray::Array2;

// ─────────────────────────────────────────────────────────────
//  Softplus barrier
// ─────────────────────────────────────────────────────────────

/// Numerically stable log(1 + exp(z)).
#[inline]
fn log1pexp(z: f64) -> f64 {
    if z > 0.0 {
        z + (-z).exp().ln_1p()
    } else {
        z.exp().ln_1p()
    }
}

/// Smooth one-sided barrier.
/// `k < 0` ⟹  penalise x < b  (min barrier).
/// `k > 0` ⟹  penalise x > b  (max barrier).
#[inline]
pub fn softplus(x: f64, b: f64, k: f64) -> f64 {
    let z = -k * (b - x) - 1.0;
    log1pexp(z)
}

/// Derivative of `softplus` w.r.t. `x`.
/// d/dx softplus = k · σ(z)  where z = −k(b−x)−1 and σ is the logistic fn.
#[inline]
pub fn softplus_grad(x: f64, b: f64, k: f64) -> f64 {
    let z = -k * (b - x) - 1.0;
    let sigma = 1.0 / (1.0 + (-z).exp());
    k * sigma
}

// ─────────────────────────────────────────────────────────────
//  Bound penalties  (barrier on the packed θ vector)
// ─────────────────────────────────────────────────────────────

/// Smooth barrier loss for lower and upper bounds on θ.
pub fn bounds_penalty(theta: &[f64], lb: &[f64], ub: &[f64], lb_idx: &[usize], ub_idx: &[usize], sharpness: f64) -> f64 {
    let mut loss = 0.0;
    for &i in lb_idx {
        if lb[i].is_finite() {
            loss += softplus(theta[i], lb[i], -sharpness);
        }
    }
    for &i in ub_idx {
        if ub[i].is_finite() {
            loss += softplus(theta[i], ub[i], sharpness);
        }
    }
    loss
}

/// Gradient of `bounds_penalty` w.r.t. θ.  Accumulates into `grad`.
pub fn bounds_penalty_grad(
    grad: &mut [f64],
    theta: &[f64],
    lb: &[f64],
    ub: &[f64],
    lb_idx: &[usize],
    ub_idx: &[usize],
    sharpness: f64,
    barrier_weight: f64,
) {
    for &i in lb_idx {
        if lb[i].is_finite() {
            grad[i] += barrier_weight * softplus_grad(theta[i], lb[i], -sharpness);
        }
    }
    for &i in ub_idx {
        if ub[i].is_finite() {
            grad[i] += barrier_weight * softplus_grad(theta[i], ub[i], sharpness);
        }
    }
}

// ─────────────────────────────────────────────────────────────
//  Individual objective losses
// ─────────────────────────────────────────────────────────────

/// TargetXYZ:  Σ_i ‖xyz[idx_i] − target_i‖²
fn target_xyz_loss(xyz: &Array2<f64>, node_indices: &[usize], target: &Array2<f64>) -> f64 {
    let mut loss = 0.0;
    for (i, &idx) in node_indices.iter().enumerate() {
        for d in 0..3 {
            let diff = xyz[[idx, d]] - target[[i, d]];
            loss += diff * diff;
        }
    }
    loss
}

/// TargetXY:  Σ_i (Δx² + Δy²)
fn target_xy_loss(xyz: &Array2<f64>, node_indices: &[usize], target: &Array2<f64>) -> f64 {
    let mut loss = 0.0;
    for (i, &idx) in node_indices.iter().enumerate() {
        for d in 0..2 {
            let diff = xyz[[idx, d]] - target[[i, d]];
            loss += diff * diff;
        }
    }
    loss
}

/// TargetLength:  Σ_i (ℓ[idx_i] − target_i)²
fn target_length_loss(lengths: &[f64], edge_indices: &[usize], target: &[f64]) -> f64 {
    let mut loss = 0.0;
    for (i, &idx) in edge_indices.iter().enumerate() {
        let diff = lengths[idx] - target[i];
        loss += diff * diff;
    }
    loss
}

/// LengthVariation:  max(ℓ) − min(ℓ)  over selected edges.
fn length_variation_loss(lengths: &[f64], edge_indices: &[usize]) -> f64 {
    if edge_indices.is_empty() {
        return 0.0;
    }
    let mut v_min = lengths[edge_indices[0]];
    let mut v_max = v_min;
    for &idx in &edge_indices[1..] {
        let v = lengths[idx];
        if v < v_min { v_min = v; }
        if v > v_max { v_max = v; }
    }
    v_max - v_min
}

/// ForceVariation:  max(f) − min(f)  over selected edges.
fn force_variation_loss(forces: &[f64], edge_indices: &[usize]) -> f64 {
    if edge_indices.is_empty() {
        return 0.0;
    }
    let mut v_min = forces[edge_indices[0]];
    let mut v_max = v_min;
    for &idx in &edge_indices[1..] {
        let v = forces[idx];
        if v < v_min { v_min = v; }
        if v > v_max { v_max = v; }
    }
    v_max - v_min
}

/// SumForceLength:  Σ_i ℓ_i · f_i  =  Σ_i q_i · ℓ_i²
fn sum_force_length_loss(lengths: &[f64], forces: &[f64], edge_indices: &[usize]) -> f64 {
    let mut loss = 0.0;
    for &idx in edge_indices {
        loss += lengths[idx] * forces[idx];
    }
    loss
}

/// MinLength / MinForce barrier:  Σ softplus(x_i, threshold_i, −k)
fn min_penalty(values: &[f64], edge_indices: &[usize], threshold: &[f64], k: f64) -> f64 {
    let mut loss = 0.0;
    for (i, &idx) in edge_indices.iter().enumerate() {
        if threshold[i].is_finite() {
            loss += softplus(values[idx], threshold[i], -k);
        }
    }
    loss
}

/// MaxLength / MaxForce barrier:  Σ softplus(x_i, threshold_i, +k)
fn max_penalty(values: &[f64], edge_indices: &[usize], threshold: &[f64], k: f64) -> f64 {
    let mut loss = 0.0;
    for (i, &idx) in edge_indices.iter().enumerate() {
        if threshold[i].is_finite() {
            loss += softplus(values[idx], threshold[i], k);
        }
    }
    loss
}

/// RigidSetCompare: Σ_{i<j} (d_target − d_network)²
fn rigid_set_compare_loss(xyz: &Array2<f64>, node_indices: &[usize], target: &Array2<f64>) -> f64 {
    let n = node_indices.len();
    let mut loss = 0.0;
    for i in 0..n {
        let idx_i = node_indices[i];
        for j in 0..i {
            let idx_j = node_indices[j];

            let dx = xyz[[idx_i, 0]] - xyz[[idx_j, 0]];
            let dy = xyz[[idx_i, 1]] - xyz[[idx_j, 1]];
            let dz = xyz[[idx_i, 2]] - xyz[[idx_j, 2]];
            let d_net = (dx * dx + dy * dy + dz * dz).sqrt();

            let tx = target[[i, 0]] - target[[j, 0]];
            let ty = target[[i, 1]] - target[[j, 1]];
            let tz = target[[i, 2]] - target[[j, 2]];
            let d_tgt = (tx * tx + ty * ty + tz * tz).sqrt();

            let diff = d_tgt - d_net;
            loss += diff * diff;
        }
    }
    loss
}

/// Cosine-based misalignment:  1 − (r̂ · d̂).
/// Returns 1.0 for zero reaction (full penalty).
#[inline]
fn direction_misalignment(reaction: [f64; 3], target_dir: [f64; 3]) -> f64 {
    let r_norm = (reaction[0].powi(2) + reaction[1].powi(2) + reaction[2].powi(2)).sqrt();
    if r_norm < f64::EPSILON {
        return 1.0;
    }
    let dot: f64 = (0..3).map(|d| reaction[d] * target_dir[d]).sum();
    let cos = (dot / r_norm).clamp(-1.0, 1.0);
    1.0 - cos
}

/// ReactionDirection:  Σ_i  (1 − r̂_i · d̂_i)
fn reaction_direction_loss(
    reactions: &Array2<f64>,
    anchor_indices: &[usize],
    target_directions: &Array2<f64>,
) -> f64 {
    let mut total = 0.0;
    for (row, &node) in anchor_indices.iter().enumerate() {
        let r = [reactions[[node, 0]], reactions[[node, 1]], reactions[[node, 2]]];
        let d = [target_directions[[row, 0]], target_directions[[row, 1]], target_directions[[row, 2]]];
        total += direction_misalignment(r, d);
    }
    total
}

/// ReactionDirectionMagnitude:  Σ_i [ (1 − r̂·d̂) + max(‖r‖ − m_i, 0) ]
fn reaction_direction_magnitude_loss(
    reactions: &Array2<f64>,
    anchor_indices: &[usize],
    target_directions: &Array2<f64>,
    target_magnitudes: &[f64],
) -> f64 {
    let mut total = 0.0;
    for (row, &node) in anchor_indices.iter().enumerate() {
        let r = [reactions[[node, 0]], reactions[[node, 1]], reactions[[node, 2]]];
        let d = [target_directions[[row, 0]], target_directions[[row, 1]], target_directions[[row, 2]]];
        let dir_loss = direction_misalignment(r, d);
        let r_norm = (r[0].powi(2) + r[1].powi(2) + r[2].powi(2)).sqrt();
        let mag_loss = (r_norm - target_magnitudes[row]).max(0.0);
        total += dir_loss + mag_loss;
    }
    total
}

// ─────────────────────────────────────────────────────────────
//  Dispatch:  Objective enum → scalar loss
// ─────────────────────────────────────────────────────────────

/// Evaluate a single objective's contribution to the total loss.
pub fn objective_loss(obj: &Objective, snap: &GeometrySnapshot) -> f64 {
    match obj {
        Objective::TargetXYZ { weight, node_indices, target } => {
            weight * target_xyz_loss(snap.xyz_full, node_indices, target)
        }
        Objective::TargetXY { weight, node_indices, target } => {
            weight * target_xy_loss(snap.xyz_full, node_indices, target)
        }
        Objective::TargetLength { weight, edge_indices, target } => {
            weight * target_length_loss(snap.member_lengths, edge_indices, target)
        }
        Objective::LengthVariation { weight, edge_indices } => {
            weight * length_variation_loss(snap.member_lengths, edge_indices)
        }
        Objective::ForceVariation { weight, edge_indices } => {
            weight * force_variation_loss(snap.member_forces, edge_indices)
        }
        Objective::SumForceLength { weight, edge_indices } => {
            weight * sum_force_length_loss(snap.member_lengths, snap.member_forces, edge_indices)
        }
        Objective::MinLength { weight, edge_indices, threshold, sharpness } => {
            weight * min_penalty(snap.member_lengths, edge_indices, threshold, *sharpness)
        }
        Objective::MaxLength { weight, edge_indices, threshold, sharpness } => {
            weight * max_penalty(snap.member_lengths, edge_indices, threshold, *sharpness)
        }
        Objective::MinForce { weight, edge_indices, threshold, sharpness } => {
            weight * min_penalty(snap.member_forces, edge_indices, threshold, *sharpness)
        }
        Objective::MaxForce { weight, edge_indices, threshold, sharpness } => {
            weight * max_penalty(snap.member_forces, edge_indices, threshold, *sharpness)
        }
        Objective::RigidSetCompare { weight, node_indices, target } => {
            weight * rigid_set_compare_loss(snap.xyz_full, node_indices, target)
        }
        Objective::ReactionDirection { weight, anchor_indices, target_directions } => {
            weight * reaction_direction_loss(snap.reactions, anchor_indices, target_directions)
        }
        Objective::ReactionDirectionMagnitude { weight, anchor_indices, target_directions, target_magnitudes } => {
            weight * reaction_direction_magnitude_loss(snap.reactions, anchor_indices, target_directions, target_magnitudes)
        }
    }
}

/// Evaluate total geometric loss (sum of all objectives).
pub fn total_loss(objectives: &[Objective], snap: &GeometrySnapshot) -> f64 {
    objectives.iter().map(|obj| objective_loss(obj, snap)).sum()
}
