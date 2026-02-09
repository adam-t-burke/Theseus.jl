//! L-BFGS optimisation driver via the `argmin` crate.
//!
//! Wraps the hand-coded `value_and_gradient` into argmin's `CostFunction`
//! + `Gradient` traits, then runs L-BFGS with the user's solver options.
//!
//! Uses `Vec<f64>` as the argmin parameter type to avoid ndarray version
//! conflicts between our ndarray 0.16 and argmin-math's bundled ndarray.

use crate::gradients::value_and_gradient;
use crate::types::{FdmCache, Problem, SolverResult, OptimizationState, TheseusError,
                   ALSettings, ALState};
use crate::objectives::{constraint_violations, max_violation};
use argmin::core::{CostFunction, Gradient, Executor, State, TerminationReason};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;
use ndarray::Array2;
use std::cell::RefCell;

// ─────────────────────────────────────────────────────────────
//  argmin problem wrapper
// ─────────────────────────────────────────────────────────────

/// Wraps the FDM problem + cache + barrier data so argmin can evaluate
/// cost and gradient.
///
/// `RefCell` is used for the cache because argmin's `CostFunction` /
/// `Gradient` traits take `&self`, but our forward solver mutates the cache.
/// The solver is single-threaded, so the borrow never actually conflicts,
/// but `RefCell` gives us debug-mode borrow checking for free.
///
/// **Evaluation cache**: argmin calls `cost(θ)` and `gradient(θ)` separately
/// at the same θ each iteration.  We cache the last `(θ, loss, grad)` so the
/// expensive forward + adjoint solve runs only once per unique θ.
struct FdmProblem<'a> {
    problem: &'a Problem,
    cache: RefCell<FdmCache>,
    lb: Vec<f64>,
    ub: Vec<f64>,
    lb_idx: Vec<usize>,
    ub_idx: Vec<usize>,
    /// Optional AL state for constrained inner solves.
    al: Option<ALState>,
    /// Cached (θ, loss, gradient) from the last evaluation.
    last_eval: RefCell<Option<(Vec<f64>, f64, Vec<f64>)>>,
}

impl<'a> FdmProblem<'a> {
    /// Ensure the cache contains results for `theta`.
    /// If θ matches the cached value, this is a no-op.
    /// Otherwise, runs the full forward + adjoint solve.
    fn ensure_evaluated(&self, theta: &[f64]) -> Result<(), argmin::core::Error> {
        {
            let cached = self.last_eval.borrow();
            if let Some((ref t, _, _)) = *cached {
                if t == theta {
                    return Ok(());
                }
            }
        }
        // Cache miss — run the full solve
        let mut fdm_cache = self.cache.borrow_mut();
        let mut grad = vec![0.0; theta.len()];
        let val = value_and_gradient(
            &mut fdm_cache,
            self.problem,
            theta,
            &mut grad,
            &self.lb,
            &self.ub,
            &self.lb_idx,
            &self.ub_idx,
            self.al.as_ref(),
        ).map_err(|e| argmin::core::Error::msg(e.to_string()))?;
        *self.last_eval.borrow_mut() = Some((theta.to_vec(), val, grad));
        Ok(())
    }
}

impl<'a> CostFunction for FdmProblem<'a> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, theta: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        self.ensure_evaluated(theta)?;
        let cached = self.last_eval.borrow();
        Ok(cached.as_ref().unwrap().1)
    }
}

impl<'a> Gradient for FdmProblem<'a> {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, theta: &Self::Param) -> Result<Self::Gradient, argmin::core::Error> {
        self.ensure_evaluated(theta)?;
        let cached = self.last_eval.borrow();
        Ok(cached.as_ref().unwrap().2.clone())
    }
}

// ─────────────────────────────────────────────────────────────
//  Parameter packing / unpacking
// ─────────────────────────────────────────────────────────────

/// Pack q and anchor positions into a single θ vector.
pub fn pack_parameters(problem: &Problem, state: &OptimizationState) -> Vec<f64> {
    let ne = problem.topology.num_edges;
    let nvar = problem.anchors.variable_indices.len();
    let mut theta = Vec::with_capacity(ne + nvar * 3);
    theta.extend_from_slice(&state.force_densities);
    if nvar > 0 {
        for i in 0..nvar {
            theta.push(state.variable_anchor_positions[[i, 0]]);
            theta.push(state.variable_anchor_positions[[i, 1]]);
            theta.push(state.variable_anchor_positions[[i, 2]]);
        }
    }
    theta
}

/// Unpack θ into q and anchor positions.
pub fn unpack_parameters(problem: &Problem, theta: &[f64]) -> (Vec<f64>, Array2<f64>) {
    let ne = problem.topology.num_edges;
    let q = theta[..ne].to_vec();
    let nvar = problem.anchors.variable_indices.len();
    let anchors = if nvar > 0 {
        let mut a = Array2::zeros((nvar, 3));
        for i in 0..nvar {
            a[[i, 0]] = theta[ne + i * 3];
            a[[i, 1]] = theta[ne + i * 3 + 1];
            a[[i, 2]] = theta[ne + i * 3 + 2];
        }
        a
    } else {
        Array2::zeros((0, 3))
    };
    (q, anchors)
}

// ─────────────────────────────────────────────────────────────
//  Bound index precomputation
// ─────────────────────────────────────────────────────────────

fn parameter_bounds(problem: &Problem) -> (Vec<f64>, Vec<f64>) {
    let nvar = problem.anchors.variable_indices.len();
    let mut lb = problem.bounds.lower.clone();
    let mut ub = problem.bounds.upper.clone();
    if nvar > 0 {
        lb.extend(vec![f64::NEG_INFINITY; nvar * 3]);
        ub.extend(vec![f64::INFINITY; nvar * 3]);
    }
    (lb, ub)
}

fn finite_indices(v: &[f64]) -> Vec<usize> {
    v.iter().enumerate().filter(|(_, &x)| x.is_finite()).map(|(i, _)| i).collect()
}

// ─────────────────────────────────────────────────────────────
//  Top-level optimisation entry point
// ─────────────────────────────────────────────────────────────

/// Run L-BFGS optimisation on the FDM problem.
///
/// Returns a `SolverResult` with the optimised geometry.
pub fn optimize(problem: &Problem, state: &mut OptimizationState) -> Result<SolverResult, TheseusError> {
    let cache = FdmCache::new(problem)?;

    let (lb, ub) = parameter_bounds(problem);
    let lb_idx = finite_indices(&lb);
    let ub_idx = finite_indices(&ub);

    let init_param = pack_parameters(problem, state);

    let fdm_problem = FdmProblem {
        problem,
        cache: RefCell::new(cache),
        lb,
        ub,
        lb_idx,
        ub_idx,
        al: None,
        last_eval: RefCell::new(None),
    };

    // Configure L-BFGS
    let linesearch = MoreThuenteLineSearch::new();
    let solver = LBFGS::new(linesearch, 10); // 10 correction pairs

    let executor = Executor::new(fdm_problem, solver)
        .configure(|config| {
            config
                .param(init_param)
                .max_iters(problem.solver.max_iterations as u64)
                .target_cost(f64::NEG_INFINITY)
        });

    let result = executor.run()?;

    // Extract solution
    let best_param = result.state().get_best_param()
        .ok_or_else(|| TheseusError::Solver("L-BFGS returned no best parameters".into()))?;
    let (q, anchors) = unpack_parameters(problem, best_param);

    // Final forward solve to get geometry
    let mut final_cache = FdmCache::new(problem)?;
    crate::fdm::solve_fdm(&mut final_cache, &q, problem, &anchors, 1e-12)?;
    crate::fdm::compute_geometry(&mut final_cache, problem);

    let converged = matches!(
        result.state().get_termination_reason(),
        Some(TerminationReason::SolverConverged)
    );

    state.force_densities = q.clone();
    state.variable_anchor_positions = anchors.clone();
    state.iterations = result.state().get_iter() as usize;

    Ok(SolverResult {
        q,
        anchor_positions: anchors,
        xyz: final_cache.nf,
        member_lengths: final_cache.member_lengths,
        member_forces: final_cache.member_forces,
        reactions: final_cache.reactions,
        loss_trace: Vec::new(), // TODO: collect via observer
        iterations: state.iterations,
        converged,
        constraint_max_violation: 0.0,
    })
}

// ─────────────────────────────────────────────────────────────
//  Inner L-BFGS solve (used by both optimize and optimize_constrained)
// ─────────────────────────────────────────────────────────────

/// Run one inner L-BFGS solve, optionally with AL penalty terms.
///
/// Returns the best parameter vector found.
fn inner_lbfgs(
    problem: &Problem,
    init_param: Vec<f64>,
    al: Option<ALState>,
) -> Result<Vec<f64>, TheseusError> {
    let cache = FdmCache::new(problem)?;

    let (lb, ub) = parameter_bounds(problem);
    let lb_idx = finite_indices(&lb);
    let ub_idx = finite_indices(&ub);

    let fdm_problem = FdmProblem {
        problem,
        cache: RefCell::new(cache),
        lb,
        ub,
        lb_idx,
        ub_idx,
        al,
        last_eval: RefCell::new(None),
    };

    let linesearch = MoreThuenteLineSearch::new();
    let solver = LBFGS::new(linesearch, 10);

    let executor = Executor::new(fdm_problem, solver)
        .configure(|config| {
            config
                .param(init_param)
                .max_iters(problem.solver.max_iterations as u64)
                .target_cost(f64::NEG_INFINITY)
        });

    let result = executor.run()?;

    result.state().get_best_param()
        .cloned()
        .ok_or_else(|| TheseusError::Solver("L-BFGS returned no best parameters".into()))
}

// ─────────────────────────────────────────────────────────────
//  Constrained optimisation via Augmented Lagrangian
// ─────────────────────────────────────────────────────────────

/// Run augmented Lagrangian constrained optimisation.
///
/// Solves a sequence of unconstrained inner problems (L-BFGS), each
/// incorporating the AL penalty:
///
///   min  f(θ)  +  Σ_k (μ/2) [max(0, λ_k/μ + g_k(θ))]²
///
/// After each inner solve, the multipliers and penalty are updated:
///
///   λ_k ← max(0, λ_k + μ · g_k)
///   μ   ← min(μ_max, α · μ)
///
/// Terminates when max(g⁺_k) < constraint_tol or max outer iterations
/// reached.
pub fn optimize_constrained(
    problem: &Problem,
    state: &mut OptimizationState,
    al_settings: &ALSettings,
) -> Result<SolverResult, TheseusError> {
    if problem.constraints.is_empty() {
        // No constraints — fall back to unconstrained optimizer
        return optimize(problem, state);
    }

    let mut al = ALState::new(&problem.constraints, al_settings);
    let mut best_param = pack_parameters(problem, state);
    let mut total_iters = 0usize;

    for outer in 0..al_settings.max_outer_iters {
        // Inner solve with current AL state
        best_param = inner_lbfgs(problem, best_param.clone(), Some(al.clone()))?;

        // Evaluate constraint violations at solution
        let (q, anchors) = unpack_parameters(problem, &best_param);
        let mut eval_cache = FdmCache::new(problem)?;
        crate::fdm::solve_fdm(&mut eval_cache, &q, problem, &anchors, 1e-12)?;
        crate::fdm::compute_geometry(&mut eval_cache, problem);

        let g = constraint_violations(&problem.constraints, &eval_cache.member_lengths);
        let viol = max_violation(&g);
        total_iters += problem.solver.max_iterations; // approximate

        eprintln!(
            "AL outer {}: μ={:.2e}, max_violation={:.4e}, |λ|_max={:.4e}",
            outer + 1, al.mu, viol,
            al.lambdas.iter().fold(0.0_f64, |m, &v| m.max(v.abs())),
        );

        if viol < al_settings.constraint_tol {
            eprintln!("AL converged: constraints satisfied to {:.2e}", viol);
            break;
        }

        // Multiplier update: λ_k ← max(0, λ_k + μ g_k)
        for (k, &gk) in g.iter().enumerate() {
            al.lambdas[k] = (al.lambdas[k] + al.mu * gk).max(0.0);
        }

        // Penalty growth
        al.mu = (al.mu * al_settings.mu_factor).min(al_settings.mu_max);
    }

    // Final forward solve for output geometry
    let (q, anchors) = unpack_parameters(problem, &best_param);
    let mut final_cache = FdmCache::new(problem)?;
    crate::fdm::solve_fdm(&mut final_cache, &q, problem, &anchors, 1e-12)?;
    crate::fdm::compute_geometry(&mut final_cache, problem);

    let g = constraint_violations(&problem.constraints, &final_cache.member_lengths);
    let final_viol = max_violation(&g);

    state.force_densities = q.clone();
    state.variable_anchor_positions = anchors.clone();
    state.iterations = total_iters;

    Ok(SolverResult {
        q,
        anchor_positions: anchors,
        xyz: final_cache.nf,
        member_lengths: final_cache.member_lengths,
        member_forces: final_cache.member_forces,
        reactions: final_cache.reactions,
        loss_trace: Vec::new(),
        iterations: total_iters,
        converged: final_viol < al_settings.constraint_tol,
        constraint_max_violation: final_viol,
    })
}
