# PALM-Mean source code

This folder contains the PALM-Mean solver. The [main README](../README.md) describes how to use it and the method; this page describes how the code is organized, for readers who want to modify or debug it.

| Module | Contents |
|---|---|
| [`ProblemSetup.py`](ProblemSetup.py) | `setup_problem()`: loads the GP, runs the presolve, sets up the scaled search box and the solver settings |
| [`GPModel.py`](GPModel.py) | `GPModel`: the trained GP (data, hyperparameters, $\alpha = (K+\sigma_\varepsilon^2 I)^{-1}y$, posterior mean) |
| [`Kernels.py`](Kernels.py) | RBF and Matérn 1/2, 3/2, 5/2 kernels: $k(r)$, $dk/dr$, inflection point |
| [`BranchBound1.py`](BranchBound1.py) | B&B driver: preprocessing, nodes, branching, pruning, main loop |
| [`LowerBound.py`](LowerBound.py) | `get_LB()`: bounds one node (analytical part + PWL MIQCP part + local upper bound) |
| [`Analytical_LB.py`](Analytical_LB.py) | Distance bounds, importance split, closed-form bounds |
| [`PWL_Custom_LB.py`](PWL_Custom_LB.py) | Gurobi MIQCP with sign-aware PWL objective (`setPWLObj`) |
| [`UpperBound.py`](UpperBound.py) | Multistart L-BFGS-B on the true posterior mean |
| [`Report.py`](Report.py) | Banner, progress table and results summary |

## How the code works

```
main.py
 └─ setup_problem(...)                     src/ProblemSetup.py
     ├─ GPModel(param_path, X_path, Y_path)  src/GPModel.py  -> x, y, ℓ, σ_f, kernel, α = K⁻¹y
     ├─ get_UB(..., n_multi=5) on [0,1]^d    presolve: initial incumbent UB*
     └─ convert to scaled space z = x / ℓ    (all B&B boxes live in z-space)
 └─ BranchBound(time_limit, GP_obj)         src/BranchBound1.py
     ├─ PrePross: split root into ≤ 500 boxes, evaluate each one as a Node
     │   (if the time limit is hit, remaining boxes get an analytical-only LB)
     └─ loop until no active node or time limit:
         ├─ pick node with min LB; update UB* from node upper bounds
         ├─ BranchOrPrune: prune (LB > UB*, node converged, or within tolerance of UB*)
         └─ or SpatialBranching: bisect longest edge -> 2 child Nodes
     └─ store run statistics in GP_obj.result
 └─ print_results(GP_obj)                   src/Report.py

Node(xL, xU) -> get_LB(bounds)              src/LowerBound.py
 ├─ get_analytic(bounds)                    src/Analytical_LB.py
 │    r_i^L (projection), r_i^U (farthest vertex), β_i, split into
 │    unimportant (Σ analytical bound = Sigma1) / important (index2)
 ├─ get_UB(bounds, n_multi)                 src/UpperBound.py   -> node UB and its argmin
 └─ PWL_LB(important terms)                 src/PWL_Custom_LB.py
      Gurobi model: x ∈ box, one distance variable per important term
        RBF:    d_i = ||(x - x_i)/ℓ||²,  PWL in d_i
                α_i < 0 -> chords; α_i > 0 -> max-of-tangents envelope
        Matérn: r_i² = ||(x - x_i)/ℓ||², PWL in r_i
                [r_i^L, r_i^U] split at the inflection point; on each piece
                max-of-tangents where α_i k is convex, chords where it is concave
      (each link is written as two quadratic inequalities)
      returns m.ObjBound = Sigma2
 LB = Sigma1 + Sigma2
```

## Implementation details

- **Adaptive MIQCP tolerance.** Nodes whose local upper bound is far above the incumbent don't need a tight lower bound.
  - If $\mathrm{UB}_h-\mathrm{UB}^*>0.51$, Gurobi's `MIPGapAbs` is set to $\mathrm{UB}_h-\mathrm{UB}^*-0.5$.
  - Otherwise it is set to $0.01$.
  - The relative gap is set to $10^{-3}$.
  - Each node MIQCP has a 10 s time limit. If Gurobi hits it, `ObjBound` is still a valid bound.
- **Preprocessing.** The number of partitions per dimension is proportional to the box side length, so the sub-boxes have similar diagonals. The count is reduced until there are at most 500 boxes.
- **Coordinates.** All bounding works in lengthscale-scaled coordinates $z=x/\ell$, where the distance is the plain Euclidean norm. Results are mapped back to $x$ before printing.
- **Kernel selection.** `GPModel` reads the optional `.kernel.type` key of `params.json` (default `rbf`). The kernel object is used for the covariance matrix, the posterior mean, the analytical bounds, the importance score and the PWL relaxation.
- **Near-zero-width distance intervals** ($r_i^U-r_i^L<10^{-5}$). Gurobi cannot use such an interval as a PWL domain. The term is bounded by the exact constant $\min(f_i(r_i^L),f_i(r_i^U))$ instead, which is valid because every kernel is monotone in the distance.
- **Tangent intersection outside its segment.** This can happen only through rounding, e.g. when two slopes are nearly equal. The midpoint is used instead, with the lower of the two tangents there, so the envelope stays below the term.
