//! C-compatible FFI for Grasshopper (C# P/Invoke).
//!
//! All functions are `#[no_mangle] extern "C"` so they can be called from
//! C# via `[DllImport("theseus")]`.
//!
//! Memory convention:
//!   - Caller allocates flat arrays and passes pointers + lengths.
//!   - Opaque handles (`*mut Problem`, `*mut FdmCache`) are created by
//!     Rust and freed by Rust via explicit `_free` functions.
//!   - No JSON, no WebSocket — pure value types over the boundary.

use crate::types::*;
use crate::optimizer;
use ndarray::Array2;
use sprs::{CsMat, TriMat};
use std::slice;

// ─────────────────────────────────────────────────────────────
//  Opaque handle helpers
// ─────────────────────────────────────────────────────────────

/// Solver handle that owns the problem + cache + state.
pub struct TheseusHandle {
    pub problem: Problem,
    pub state: OptimizationState,
}

// ─────────────────────────────────────────────────────────────
//  Problem construction
// ─────────────────────────────────────────────────────────────

/// Create a new problem from raw arrays.
///
/// # Safety
/// All pointers must be valid for the given lengths.
#[no_mangle]
pub unsafe extern "C" fn theseus_create(
    // ── Incidence (COO triplets) ──
    num_edges: usize,
    num_nodes: usize,
    num_free: usize,
    coo_rows: *const usize,    // length = nnz
    coo_cols: *const usize,
    coo_vals: *const f64,
    coo_nnz: usize,
    free_node_indices: *const usize, // length = num_free
    fixed_node_indices: *const usize, // length = num_nodes - num_free
    num_fixed: usize,
    // ── Loads & geometry ──
    loads: *const f64,             // num_free × 3  row-major
    fixed_positions: *const f64,   // num_fixed × 3 row-major
    // ── Initial q ──
    q_init: *const f64,            // length = num_edges
    // ── Bounds ──
    lower_bounds: *const f64,      // length = num_edges
    upper_bounds: *const f64,
) -> *mut TheseusHandle {
    // Reconstruct slices
    let rows = slice::from_raw_parts(coo_rows, coo_nnz);
    let cols = slice::from_raw_parts(coo_cols, coo_nnz);
    let vals = slice::from_raw_parts(coo_vals, coo_nnz);
    let free_idx = slice::from_raw_parts(free_node_indices, num_free).to_vec();
    let fixed_idx = slice::from_raw_parts(fixed_node_indices, num_fixed).to_vec();
    let loads_slice = slice::from_raw_parts(loads, num_free * 3);
    let fixed_pos_slice = slice::from_raw_parts(fixed_positions, num_fixed * 3);
    let q_slice = slice::from_raw_parts(q_init, num_edges);
    let lb_slice = slice::from_raw_parts(lower_bounds, num_edges);
    let ub_slice = slice::from_raw_parts(upper_bounds, num_edges);

    // Build incidence matrix from COO
    let mut tri = TriMat::new((num_edges, num_nodes));
    for i in 0..coo_nnz {
        tri.add_triplet(rows[i], cols[i], vals[i]);
    }
    let incidence = tri.to_csc();

    // Free / fixed sub-matrices
    let free_inc = extract_columns(&incidence, &free_idx);
    let fixed_inc = extract_columns(&incidence, &fixed_idx);

    let topology = NetworkTopology {
        incidence,
        free_incidence: free_inc,
        fixed_incidence: fixed_inc,
        num_edges,
        num_nodes,
        free_node_indices: free_idx,
        fixed_node_indices: fixed_idx.clone(),
    };

    // Loads: row-major num_free × 3
    let free_node_loads = Array2::from_shape_vec(
        (num_free, 3),
        loads_slice.to_vec(),
    ).unwrap();

    // Fixed positions
    let fixed_node_positions = Array2::from_shape_vec(
        (num_fixed, 3),
        fixed_pos_slice.to_vec(),
    ).unwrap();

    let anchors = AnchorInfo::all_fixed(fixed_node_positions.clone());

    let bounds = Bounds {
        lower: lb_slice.to_vec(),
        upper: ub_slice.to_vec(),
    };

    let problem = Problem {
        topology,
        free_node_loads,
        fixed_node_positions,
        anchors,
        objectives: Vec::new(),
        bounds,
        solver: SolverOptions::default(),
    };

    let state = OptimizationState::new(
        q_slice.to_vec(),
        Array2::zeros((0, 3)),
    );

    Box::into_raw(Box::new(TheseusHandle { problem, state }))
}

/// Free a handle.
///
/// # Safety
/// `handle` must be a pointer returned by `theseus_create`.
#[no_mangle]
pub unsafe extern "C" fn theseus_free(handle: *mut TheseusHandle) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

// ─────────────────────────────────────────────────────────────
//  Objective registration
// ─────────────────────────────────────────────────────────────

/// Add a TargetXYZ objective.
///
/// # Safety
/// Valid handle and arrays.
#[no_mangle]
pub unsafe extern "C" fn theseus_add_target_xyz(
    handle: *mut TheseusHandle,
    weight: f64,
    node_indices: *const usize,
    num_nodes: usize,
    target_xyz: *const f64, // num_nodes × 3 row-major
) {
    let h = &mut *handle;
    let idx = slice::from_raw_parts(node_indices, num_nodes).to_vec();
    let target = Array2::from_shape_vec(
        (num_nodes, 3),
        slice::from_raw_parts(target_xyz, num_nodes * 3).to_vec(),
    ).unwrap();
    h.problem.objectives.push(Objective::TargetXYZ {
        weight,
        node_indices: idx,
        target,
    });
}

/// Add a TargetLength objective.
///
/// # Safety
/// Valid handle and arrays.
#[no_mangle]
pub unsafe extern "C" fn theseus_add_target_length(
    handle: *mut TheseusHandle,
    weight: f64,
    edge_indices: *const usize,
    num_edges: usize,
    targets: *const f64,
) {
    let h = &mut *handle;
    let idx = slice::from_raw_parts(edge_indices, num_edges).to_vec();
    let tgt = slice::from_raw_parts(targets, num_edges).to_vec();
    h.problem.objectives.push(Objective::TargetLength {
        weight,
        edge_indices: idx,
        target: tgt,
    });
}

/// Add a MinLength barrier objective.
///
/// # Safety
/// Valid handle and arrays.
#[no_mangle]
pub unsafe extern "C" fn theseus_add_min_length(
    handle: *mut TheseusHandle,
    weight: f64,
    edge_indices: *const usize,
    num_edges: usize,
    thresholds: *const f64,
    sharpness: f64,
) {
    let h = &mut *handle;
    let idx = slice::from_raw_parts(edge_indices, num_edges).to_vec();
    let thr = slice::from_raw_parts(thresholds, num_edges).to_vec();
    h.problem.objectives.push(Objective::MinLength {
        weight,
        edge_indices: idx,
        threshold: thr,
        sharpness,
    });
}

/// Configure solver options.
///
/// # Safety
/// Valid handle.
#[no_mangle]
pub unsafe extern "C" fn theseus_set_solver_options(
    handle: *mut TheseusHandle,
    max_iterations: usize,
    abs_tol: f64,
    rel_tol: f64,
    barrier_weight: f64,
    barrier_sharpness: f64,
) {
    let h = &mut *handle;
    h.problem.solver = SolverOptions {
        max_iterations,
        absolute_tolerance: abs_tol,
        relative_tolerance: rel_tol,
        report_frequency: 1,
        barrier_weight,
        barrier_sharpness,
    };
}

// ─────────────────────────────────────────────────────────────
//  Run optimisation
// ─────────────────────────────────────────────────────────────

/// Run L-BFGS optimisation.  Results are written into caller-provided buffers.
///
/// Returns 0 on success, non-zero on error.
///
/// # Safety
/// All output buffers must have the correct sizes.
#[no_mangle]
pub unsafe extern "C" fn theseus_optimize(
    handle: *mut TheseusHandle,
    // ── Outputs ──
    out_xyz: *mut f64,           // num_nodes × 3 row-major
    out_lengths: *mut f64,       // num_edges
    out_forces: *mut f64,        // num_edges
    out_q: *mut f64,             // num_edges
    out_reactions: *mut f64,     // num_nodes × 3 row-major
    out_iterations: *mut usize,
    out_converged: *mut bool,
) -> i32 {
    let h = &mut *handle;

    match optimizer::optimize(&h.problem, &mut h.state) {
        Ok(result) => {
            let nn = h.problem.topology.num_nodes;
            let ne = h.problem.topology.num_edges;

            // Copy xyz
            let xyz_out = slice::from_raw_parts_mut(out_xyz, nn * 3);
            for i in 0..nn {
                for d in 0..3 {
                    xyz_out[i * 3 + d] = result.xyz[[i, d]];
                }
            }

            // Copy lengths, forces, q
            slice::from_raw_parts_mut(out_lengths, ne).copy_from_slice(&result.member_lengths);
            slice::from_raw_parts_mut(out_forces, ne).copy_from_slice(&result.member_forces);
            slice::from_raw_parts_mut(out_q, ne).copy_from_slice(&result.q);

            // Copy reactions
            let r_out = slice::from_raw_parts_mut(out_reactions, nn * 3);
            for i in 0..nn {
                for d in 0..3 {
                    r_out[i * 3 + d] = result.reactions[[i, d]];
                }
            }

            *out_iterations = result.iterations;
            *out_converged = result.converged;

            0 // success
        }
        Err(_) => 1, // error
    }
}

// ─────────────────────────────────────────────────────────────
//  Forward solve only  (no optimisation)
// ─────────────────────────────────────────────────────────────

/// Single forward FDM solve — useful for previewing geometry without optimising.
///
/// # Safety
/// Valid handle and output buffers.
#[no_mangle]
pub unsafe extern "C" fn theseus_solve_forward(
    handle: *mut TheseusHandle,
    out_xyz: *mut f64,       // num_nodes × 3
    out_lengths: *mut f64,   // num_edges
    out_forces: *mut f64,    // num_edges
) -> i32 {
    let h = &mut *handle;
    let mut cache = FdmCache::new(&h.problem);
    let anchors = h.state.variable_anchor_positions.clone();

    crate::fdm::solve_fdm(&mut cache, &h.state.force_densities, &h.problem, &anchors, 1e-12);

    let nn = h.problem.topology.num_nodes;
    let ne = h.problem.topology.num_edges;

    let xyz_out = slice::from_raw_parts_mut(out_xyz, nn * 3);
    for i in 0..nn {
        for d in 0..3 {
            xyz_out[i * 3 + d] = cache.nf[[i, d]];
        }
    }
    slice::from_raw_parts_mut(out_lengths, ne).copy_from_slice(&cache.member_lengths);
    slice::from_raw_parts_mut(out_forces, ne).copy_from_slice(&cache.member_forces);

    0
}

// ─────────────────────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────────────────────

/// Extract columns from a CSC matrix by index.
fn extract_columns(mat: &CsMat<f64>, cols: &[usize]) -> CsMat<f64> {
    let nrows = mat.rows();
    let ncols = cols.len();
    let mut tri = TriMat::new((nrows, ncols));

    let mat_csc = mat.to_csc();
    for (new_col, &old_col) in cols.iter().enumerate() {
        let start = mat_csc.indptr().raw_storage()[old_col];
        let end_ = mat_csc.indptr().raw_storage()[old_col + 1];
        for nz in start..end_ {
            tri.add_triplet(mat_csc.indices()[nz], new_col, mat_csc.data()[nz]);
        }
    }

    tri.to_csc()
}
