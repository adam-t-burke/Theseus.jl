//! Forward FDM solver: assemble A(q), build RHS, factorise, triangular solve.
//!
//! Mirrors `src/FDM.jl` from the Julia code.

use crate::types::{FdmCache, Problem};
use ndarray::Array2;
use sprs::CsMat;

// ─────────────────────────────────────────────────────────────
//  Fixed-node position assembly
// ─────────────────────────────────────────────────────────────

/// Write current fixed-node positions into `cache.nf`, overlaying reference
/// positions and (optionally) variable anchor positions.
pub fn update_fixed_positions(cache: &mut FdmCache, problem: &Problem, anchor_positions: &Array2<f64>) {
    let fixed = &problem.topology.fixed_node_indices;
    let nn_fixed = fixed.len();

    // 1. Copy reference positions for fixed nodes
    let ref_pos = &problem.anchors.reference_positions;
    if ref_pos.nrows() == nn_fixed {
        for (i, &node) in fixed.iter().enumerate() {
            for d in 0..3 {
                cache.nf[[node, d]] = ref_pos[[i, d]];
            }
        }
    } else if ref_pos.nrows() == problem.topology.num_nodes {
        for &node in fixed {
            for d in 0..3 {
                cache.nf[[node, d]] = ref_pos[[node, d]];
            }
        }
    }

    // 2. Overlay variable anchors
    for (i, &node) in problem.anchors.variable_indices.iter().enumerate() {
        for d in 0..3 {
            cache.nf[[node, d]] = anchor_positions[[i, d]];
        }
    }
}

// ─────────────────────────────────────────────────────────────
//  System matrix assembly  A(q)
// ─────────────────────────────────────────────────────────────

/// Zero-allocation in-place update of `cache.a_data` from current q values.
/// A = Cn^T diag(q) Cn  via the precomputed `q_to_nz` mapping.
pub fn assemble_a(cache: &mut FdmCache) {
    cache.a_data.fill(0.0);
    for (k, entries) in cache.q_to_nz.entries.iter().enumerate() {
        let qk = cache.q[k];
        for &(nz_idx, coeff) in entries {
            cache.a_data[nz_idx] += qk * coeff;
        }
    }
}

// ─────────────────────────────────────────────────────────────
//  RHS assembly:  b = Pn − Cn^T diag(q) Cf Nf_fixed
// ─────────────────────────────────────────────────────────────

/// Build the right-hand side for A x = b.
///
/// Steps:
///   1. Copy fixed-node positions into dense buffer `nf_fixed`
///   2. cf_nf  = Cf * nf_fixed        (ne × 3)
///   3. q_cf_nf = diag(q) * cf_nf     (ne × 3)
///   4. rhs    = Pn − Cn^T * q_cf_nf  (nn_free × 3)
pub fn assemble_rhs(cache: &mut FdmCache, problem: &Problem) {
    let fixed = &problem.topology.fixed_node_indices;

    // 1. nf_fixed dense buffer
    for (i, &node) in fixed.iter().enumerate() {
        for d in 0..3 {
            cache.nf_fixed[[i, d]] = cache.nf[[node, d]];
        }
    }

    // 2. cf_nf = Cf * nf_fixed   (sparse × dense, column by column)
    let cf_csc = cache.cf.to_csc();
    spmm_into(&cf_csc, &cache.nf_fixed, &mut cache.cf_nf);

    // 3. q_cf_nf = diag(q) * cf_nf
    let ne = cache.q.len();
    for i in 0..ne {
        let qi = cache.q[i];
        for d in 0..3 {
            cache.q_cf_nf[[i, d]] = qi * cache.cf_nf[[i, d]];
        }
    }

    // 4. rhs = Pn − Cn^T * q_cf_nf
    cache.rhs.assign(&cache.pn);
    let cn_t = cache.cn.transpose_view();
    let cn_t_csc = cn_t.to_csc();
    // rhs -= Cn^T * q_cf_nf
    spmm_sub_into(&cn_t_csc, &cache.q_cf_nf, &mut cache.rhs);
}

// ─────────────────────────────────────────────────────────────
//  Linear solve  (dense fallback — will be replaced with LDL)
// ─────────────────────────────────────────────────────────────

/// Solve A x = rhs using a dense LDL/Cholesky factorisation.
///
/// This is a temporary implementation that converts A to dense.
/// Once `sprs-ldl` or `cholmod` bindings are integrated, this will
/// use the sparse factorisation directly.
pub fn solve_linear_system(cache: &mut FdmCache, perturbation: f64) {
    let n = cache.a_dim;

    // Reconstruct dense A from CSC
    let mut a_dense = Array2::<f64>::zeros((n, n));
    for col in 0..n {
        let start = cache.a_indptr[col];
        let end_ = cache.a_indptr[col + 1];
        for nz in start..end_ {
            let row = cache.a_indices[nz];
            a_dense[[row, col]] = cache.a_data[nz];
        }
    }

    // Add perturbation to diagonal if requested
    if perturbation > 0.0 {
        for i in 0..n {
            a_dense[[i, i]] += perturbation;
        }
    }

    // Solve via dense LDL (Bunch-Kaufman-like pivot)
    // For now, use a straightforward Gaussian elimination with partial pivoting.
    // TODO: Replace with sprs-ldl sparse factorisation
    for d in 0..3 {
        let mut b: Vec<f64> = (0..n).map(|i| cache.rhs[[i, d]]).collect();
        let x = dense_solve(&a_dense, &mut b);
        for i in 0..n {
            cache.x[[i, d]] = x[i];
        }
    }
}

/// Simple dense symmetric solve via LDLT factorisation (diagonal pivoting).
/// Operates on a copy to preserve the original.
pub fn dense_solve(a: &Array2<f64>, b: &mut [f64]) -> Vec<f64> {
    let n = b.len();
    // Copy A and b for Gaussian elimination
    let mut m = a.clone();
    let mut x = b.to_vec();

    // Gaussian elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = m[[col, col]].abs();
        for row in (col + 1)..n {
            let v = m[[row, col]].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }

        // Swap rows
        if max_row != col {
            for j in 0..n {
                let tmp = m[[col, j]];
                m[[col, j]] = m[[max_row, j]];
                m[[max_row, j]] = tmp;
            }
            x.swap(col, max_row);
        }

        let pivot = m[[col, col]];
        if pivot.abs() < 1e-30 {
            // Singular — return zeros
            return vec![0.0; n];
        }

        // Eliminate below
        for row in (col + 1)..n {
            let factor = m[[row, col]] / pivot;
            for j in col..n {
                m[[row, j]] -= factor * m[[col, j]];
            }
            x[row] -= factor * x[col];
        }
    }

    // Back substitution
    for col in (0..n).rev() {
        for row in (col + 1)..n {
            x[col] -= m[[col, row]] * x[row];
        }
        x[col] /= m[[col, col]];
    }

    x
}

// ─────────────────────────────────────────────────────────────
//  Top-level forward solve
// ─────────────────────────────────────────────────────────────

/// Full forward FDM solve.  Updates `cache.x`, `cache.nf`,
/// `cache.member_lengths`, `cache.member_forces`, `cache.reactions`.
pub fn solve_fdm(
    cache: &mut FdmCache,
    q: &[f64],
    problem: &Problem,
    anchor_positions: &Array2<f64>,
    perturbation: f64,
) {
    // 0. Sync q
    cache.q.copy_from_slice(q);

    // 1. Assemble A
    assemble_a(cache);

    // 2. Update fixed positions in Nf
    update_fixed_positions(cache, problem, anchor_positions);

    // 3. Assemble RHS
    assemble_rhs(cache, problem);

    // 4. Solve A x = rhs
    solve_linear_system(cache, perturbation);

    // 5. Write free-node positions back to Nf
    for (i, &node) in problem.topology.free_node_indices.iter().enumerate() {
        for d in 0..3 {
            cache.nf[[node, d]] = cache.x[[i, d]];
        }
    }

    // 6. Compute derived geometry
    compute_geometry(cache, problem);
}

/// Compute member lengths, forces, and reactions from current positions and q.
pub fn compute_geometry(cache: &mut FdmCache, problem: &Problem) {
    let ne = problem.topology.num_edges;

    for i in 0..ne {
        let s = cache.edge_starts[i];
        let e = cache.edge_ends[i];

        let dx = cache.nf[[e, 0]] - cache.nf[[s, 0]];
        let dy = cache.nf[[e, 1]] - cache.nf[[s, 1]];
        let dz = cache.nf[[e, 2]] - cache.nf[[s, 2]];

        let len = (dx * dx + dy * dy + dz * dz).sqrt();
        cache.member_lengths[i] = len;
        cache.member_forces[i] = cache.q[i] * len;
    }

    // Reactions: for each edge, accumulate axial force contributions
    cache.reactions.fill(0.0);
    for i in 0..ne {
        let s = cache.edge_starts[i];
        let e = cache.edge_ends[i];
        let qi = cache.q[i];

        let rx = (cache.nf[[e, 0]] - cache.nf[[s, 0]]) * qi;
        let ry = (cache.nf[[e, 1]] - cache.nf[[s, 1]]) * qi;
        let rz = (cache.nf[[e, 2]] - cache.nf[[s, 2]]) * qi;

        cache.reactions[[s, 0]] += rx;
        cache.reactions[[s, 1]] += ry;
        cache.reactions[[s, 2]] += rz;

        cache.reactions[[e, 0]] -= rx;
        cache.reactions[[e, 1]] -= ry;
        cache.reactions[[e, 2]] -= rz;
    }
}

// ─────────────────────────────────────────────────────────────
//  Sparse × dense helpers
// ─────────────────────────────────────────────────────────────

/// out = A * B   where A is CSC (m × k), B is dense (k × 3), out is dense (m × 3).
fn spmm_into(a: &CsMat<f64>, b: &Array2<f64>, out: &mut Array2<f64>) {
    out.fill(0.0);
    let ncols_a = a.cols();
    for col in 0..ncols_a {
        let start = a.indptr().raw_storage()[col];
        let end_ = a.indptr().raw_storage()[col + 1];
        for nz in start..end_ {
            let row = a.indices()[nz];
            let val = a.data()[nz];
            for d in 0..3 {
                out[[row, d]] += val * b[[col, d]];
            }
        }
    }
}

/// out -= A * B   (subtract sparse-dense product from existing out).
fn spmm_sub_into(a: &CsMat<f64>, b: &Array2<f64>, out: &mut Array2<f64>) {
    let ncols_a = a.cols();
    for col in 0..ncols_a {
        let start = a.indptr().raw_storage()[col];
        let end_ = a.indptr().raw_storage()[col + 1];
        for nz in start..end_ {
            let row = a.indices()[nz];
            let val = a.data()[nz];
            for d in 0..3 {
                out[[row, d]] -= val * b[[col, d]];
            }
        }
    }
}
