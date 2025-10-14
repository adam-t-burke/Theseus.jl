# Automatic Differentiation Alignment Workplan

This document captures the current state of Theseus' automatic-differentiation (AD) hookups and lays out a concrete plan to align the implementation with the guidance from the Mooncake and DifferentiationInterface packages.

---

## Quick Reference · Mooncake & DifferentiationInterface APIs

- **Two gradient pathways:** Mooncake supports both the standardized DifferentiationInterface (DI) façade and its native API. We stick with DI for flexibility, while the native reverse-mode calls (`Mooncake.value_and_gradient!!`, `Mooncake.value_and_pullback!!`) remain available when we need bespoke control.
- **Backend selection:** Construct a reusable backend with `DI.AutoMooncake(; config = nothing)` (re-exported by ADTypes). Supplying an explicit `Mooncake.Config` is optional but gives access to debug/stability toggles.
- **Preparation pays off:** Calling `DI.gradient` directly is slow because Mooncake must synthesize rules each time. Instead, `DI.prepare_gradient(f, backend, typical_x)` captures the rule once; pair it with `DI.gradient!` or `DI.value_and_gradient!` and buffers you allocate ahead of time for high-throughput solves.
- **Multiple arguments:** DI differentiates with respect to the first argument. Wrap immutable parameters in `DI.Constant` and scratch buffers in `DI.Cache` so Mooncake can update them without retracing.
- **Tuple trick:** When gradients for all arguments are needed, bundle them into a single tuple input (`g_tup((x, a, b))`) and reuse the same prep object.
- **Beyond gradients:** DI mirrors Mooncake's broader capabilities with `prepare_jacobian` / `DI.jacobian`. Native Mooncake terminology maps Fréchet derivatives (forward pushforwards) and their adjoints (reverse pullbacks) to the same operations.

---

## Phase 1 · Baseline Assessment

### Goals
- Audit how Theseus prepares, executes, and reuses gradients.
- Compare the current code paths to Mooncake's tutorial guidance and DifferentiationInterface contracts.
- Decide when to stay on the DI façade versus dropping to Mooncake's native API.

### Findings

#### DifferentiationInterface usage
- **Backend reuse** ✅  
  `AutoMooncake()` is instantiated once (as `MOONCAKE_BACKEND`) and shared across solves, matching DI's recommendation. We can optionally surface `Mooncake.Config` knobs by constructing the backend via `DI.AutoMooncake(; config = ...)` when we need debug toggles.

- **Context arguments & multi-parameter handling** ⚠️  
  The gradient closure (`objective_plain`) still captures `problem`, `state`, and a mutable `GeometryWorkspace`. The Mooncake tutorial confirms DI expects non-differentiated data to be passed explicitly using `DI.Constant` (for immutable metadata) and `DI.Cache` (for scratch buffers). Until we refactor to an explicit context, Mooncake must guess how to treat these captured values.

- **Gradient API choice** ⚠️  
  We call `DI.value_and_gradient!`, discard the value, and let `Optim.jl` evaluate the objective separately. Tutorial guidance emphasizes choosing the minimal primitive—`DI.gradient!` with a buffer we allocate up front—which would eliminate redundant objective evaluations and align with our new context types.

- **Preparation & storage strategy** ✅  
  `prepare_gradient` runs once per solve, and we reuse the resulting prep object exactly as Mooncake recommends. Allocating gradient storage ahead of time (once the cache rewrite lands) will let us pair the prep with `DI.gradient!` for "very fast" evaluations.

#### Mooncake-specific considerations
- **Custom rule coverage** ⚠️  
  Mooncake ignores ChainRules-style `rrule`s, so AutoMooncake still retraces the CHOLMOD factorization. We must finish the Mooncake-native `rrule!!`/`value_and_pullback!!` path built around `solve_explicit_pullback!` to expose our analytic derivatives.

- **Opaque cache contract** ⚠️  
  Capturing the sparse workspace inside the closure leaves Mooncake to differentiate through CHOLMOD pointers. The tutorial's emphasis on `DI.Cache` and `Mooncake.NoTangent` reinforces the need for an `FDMContext`/`FDMCache` layer that marks the sparse internals as opaque.

- **Derivative hygiene** ⚠️  
  `ensure_parameter_bounds!` still clamps θ outside the differentiable region, effectively zeroing gradients. We should either document the intentional projection or replace it with a smooth penalty compatible with Mooncake's debug tooling.

#### Testing & diagnostics
- No automated test yet exercises Mooncake's gradient against the optimizer loop; only the ChainRules path is covered by finite differences. We need explicit Mooncake regression tests once the new context lands.
- Switching backends or enabling Mooncake debug still requires code edits. Documenting how to wire `Mooncake.Config` through DI will reduce friction during diagnostics.

#### DifferentiationInterface vs. direct Mooncake
- **Keep DI** when we want to swap AD engines quickly (e.g., ForwardDiff for debugging) or lean on DI's preparation caches.
- **Go native** when bespoke Mooncake rules or cache plumbing demand full control (e.g., scalar pullbacks, direct `value_and_gradient!!` usage).

Given Theseus already hand-derives pullbacks and needs Mooncake-friendly mutation patterns, we continue to pursue a **hybrid** strategy: retain the DI façade for flexibility while exposing Mooncake-native rules and explicit cache/context arguments so AutoMooncake can use our analytic derivatives efficiently.

---

## Phase 2 · Immediate Refactor Plan

1. **Isolate dense interface from sparse workspace internals**
  - Split immutable metadata into an `FDMConstantData` (topology, templates, dimensions) and mutable buffers into an `FDMCache` that wraps the existing `FDMSolverWorkspace` plus dense scratch arrays.
  - Introduce an `FDMContext` that holds `{cache, constants}`; this is the single object passed through DifferentiationInterface as `DI.Cache(ctx)` / `DI.Constant(problem)`.
  - Ensure `FDMCache`/`FDMContext` expose only dense views (q, loads, anchors, free-node coordinates) and keep all sparse state private.

2. **Refactor `solve_explicit` entry points**
  - Define a forward helper `solve_geometry(ctx, q, loads, anchors)` that writes into cache buffers and returns the dense free-node positions.
  - Update `solve_explicit_pullback!` to accept the new context/cache and keep writing gradients into the pre-allocated dense buffers (grad_q, grad_loads, grad_fixed).

3. **Teach AD stacks to treat the cache as opaque**
  - Provide `Mooncake.zero_tangent(::FDMCache/FDMContext) = Mooncake.NoTangent()` (and analogous ChainRules `NoTangent`) so Mooncake never recurses into CHOLMOD pointers or sparse buffers.
  - Adjust `objective_core` and the gradient prep to receive `(θ, DI.Cache(ctx), DI.Constant(problem))`, delegating to the new `solve_geometry` helper.

4. **Re-implement custom rules on the clean interface**
  - Rewrite the ChainRules `rrule(::typeof(solve_geometry), …)` and Mooncake `rrule!!` to call the shared `solve_explicit_pullback!`, copying out dense gradient views for q, loads, and anchors.
  - Remove any leftover duplication between ChainRules and Mooncake paths; they should share the helper and return the same dense tangents.

5. **Validation & ergonomics**
  - Update tests to exercise `optimize_problem!` through the new context (including the Mooncake-native rule).
  - Document the new architecture (constants vs cache, opaque context) so future objectives can reuse the pattern.

---

## Phase 3 · Stretch Goals

- Allow `Optim.jl` (or future solvers) to consume Mooncake caches directly, minimizing repeated factorizations when θ changes incrementally.
- Explore Mooncake's roadmap for sparse pullbacks or Jacobian/Hessian computations (e.g., `prepare_jacobian`, `prepare_pullback_cache`).
- Establish benchmarking scripts to compare the ChainRules vs. Mooncake-native gradient paths once the new rules land.

---

## Quality Gates & Follow-up

- **Build / Lint / Tests:** not run yet (analysis document only). When implementing the plan, ensure CI runs `Optim.jl` tests with Mooncake gradients in both normal and debug configurations.
- **Next action:** Implement Phase 2, starting with the objective refactor and Mooncake rule translation, then iterate toward the stretch items as needed.
