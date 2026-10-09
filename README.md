# PALM-Mean: An Efficient Spatial Branch-and-Bound Algorithm for Global Optimization of Gaussian Process Posterior Mean Functions

This repository contains the reference implementation of **PALM-Mean** (**P**iecewise-**A**na**L**ytic **M**ean bounding), proposed in

> W.-T. Tang, A. Kudva, C. Tsay, J. A. Paulson. *An Efficient Spatial Branch-and-Bound Algorithm for Global Optimization of Gaussian Process Posterior Mean Functions.* arXiv:[2605.00855](https://arxiv.org/abs/2605.00855), 2026.

PALM-Mean finds the **certified global minimum** of a trained Gaussian process (GP) posterior mean over a box. It is a reduced-space spatial branch-and-bound (B&B) method. On each node, kernel terms that strongly affect the posterior mean are bounded with a sign-aware piecewise-linear (PWL) relaxation, and all other terms are bounded in closed form.

![Demo](figure/palm_mean_1d(3).gif)

*1-D illustration of PALM-Mean: posterior mean (black), node lower-bounding functions, incumbent, and the active boxes as the search proceeds. The animation was made with [`figure/animate_palm_mean_1d.py`](figure/animate_palm_mean_1d.py).*

---

## Contents

1. [Installation](#installation)
2. [Quick start](#quick-start)
3. [Using your own GP](#using-your-own-gp)
4. [Solver settings](#solver-settings)
5. [Repository structure](#repository-structure)
6. [Method](#method)
7. [How the code works](#how-the-code-works)
8. [Benchmarks](#benchmarks)
9. [Limitations](#limitations)
10. [Citation](#citation)

---

## Installation

Requirements: Python 3 (tested with 3.14) and a **Gurobi license**. Gurobi solves the node lower-bounding problem, a mixed-integer quadratically constrained program (MIQCP). Free academic licenses are available from [gurobi.com](https://www.gurobi.com/academia/academic-program-and-licenses/).

```sh
git clone https://github.com/PaulsonLab/PALM-Mean.git
cd PALM-Mean

# create and activate a virtual environment (recommended)
python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1
# Linux / macOS:         source .venv/bin/activate

pip install -r requirements.txt
# torch is only used for tensor arithmetic; the CPU build is sufficient:
# pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Point Gurobi to your license file if it is not in a default location:

```sh
# Windows (PowerShell)
$env:GRB_LICENSE_FILE = "C:\path\to\gurobi.lic"
# Linux / macOS
export GRB_LICENSE_FILE=/path/to/gurobi.lic
```

To check that the license works, run `python -c "import gurobipy as gp; gp.Model()"`. It should print your license information without an error.

## Quick start

```sh
python main.py
```

This minimizes the posterior mean of the example GP in [`data/`](data) (3 inputs, 20 training points). It finishes in a few seconds and prints something like:

```
x* =  tensor([[0.4457, 0.4568, 0.6762]], dtype=torch.float64)
Minimum posterior mean =  -1.6607193097371928
Absolute Gap =  0.025950985561651763
```

| Output | Meaning |
|---|---|
| `x*` | Best point found (the incumbent), in the original input space. |
| `Minimum posterior mean` | $\mu(x^*)$, the best upper bound $\mathrm{UB}^*$. |
| `Absolute Gap` | $\mathrm{UB}^* - \mathrm{LB}^*$. The global minimum is guaranteed to lie in $[\mathrm{LB}^*, \mathrm{UB}^*]$. |

If the time limit is reached, the run prints `B&B process exceeds specified time limit!` and the reported gap is still a valid (but looser) certificate.

## Using your own GP

PALM-Mean takes a **trained** GP. Hyperparameters are not re-fitted. Provide three files and set their paths in [`main.py`](main.py):

| File | Content |
|---|---|
| `X.npy` | Training inputs, `float64` array of shape `(N, d)`. **The search domain is fixed to $[0,1]^d$**, so scale your inputs to the unit box first. |
| `Y.npy` | Training outputs, array of shape `(N, 1)`. A zero prior mean is assumed, so **center (or standardize) the outputs** the same way they were centered when the GP was trained. |
| `params.json` | Hyperparameters of the ARD squared-exponential (RBF) kernel, using GPflow-style keys (see below). |

```json
{".kernel.variance": 1.0, ".kernel.lengthscales": [0.1, 0.1, 0.1], ".likelihood.variance": 0.0001}
```

- `.kernel.variance`: signal variance $\sigma_f^2$
- `.kernel.lengthscales`: one lengthscale $\ell_j$ per input dimension
- `.likelihood.variance`: observation-noise variance $\sigma_\varepsilon^2$

From these values the code builds $\mu(x)=\sum_i \alpha_i\,\sigma_f^2\exp\!\big(-\tfrac12\lVert L^{-1}(x-x_i)\rVert^2\big)$ with $\alpha=(K+\sigma_\varepsilon^2 I)^{-1}y$. If you trained the GP in BoTorch/GPyTorch or scikit-learn, copy the fitted outputscale, lengthscales and noise into these three keys.

## Solver settings

All settings are at the top of [`main.py`](main.py):

| Parameter | Default | Description |
|---|---|---|
| `time_limit` | `600` | Wall-clock limit (s) for preprocessing and B&B combined. |
| `abs_gap` | `0.1` | Absolute optimality tolerance $\varepsilon_{\rm abs}$. |
| `rel_gap` | `0.01` | Relative optimality tolerance $\varepsilon_{\rm rel}$. A node is fathomed once **either** tolerance is met. |
| `hyperparameter` | `0.01` | Importance threshold $\rho$. Terms with $\beta_{i,h}\ge\rho$ get the PWL relaxation, the rest are bounded analytically. $\rho=0$ relaxes every term with PWL (tightest bounds, slowest solves). |
| `PWL_segment` | `6` | Number of PWL segments per important kernel term. |
| `Preprocess` | `True` | Pre-partition the root box into sub-boxes with comparable diagonals, capped at 500 boxes. |
| `n_prepart` | `10` | Starting number of partitions along the diagonal for preprocessing. Reduced automatically until there are at most 500 boxes. |
| `n_multi` | `2` | Number of random L-BFGS-B starts per node for the upper bound. |
| `Thread_gurobi` | `1` | Threads used by Gurobi for each node MIQCP (and by torch). |

The paper's experiments use $\rho=0.01$, 6 PWL segments, $\varepsilon_{\rm abs}=0.1$ and $\varepsilon_{\rm rel}=0.01$.

## Repository structure

```
PALM-Mean/
├── main.py                     # Entry point: settings, data paths, runs B&B, prints result
├── requirements.txt            # torch, numpy, scipy, gurobipy
├── data/
│   ├── X.npy                   # Example training inputs  (20 x 3, in [0,1]^3)
│   ├── Y.npy                   # Example training outputs (20 x 1, centered)
│   └── params.json             # Example RBF hyperparameters
├── src/
│   ├── GPModel.py              # GPModel: trained GP - loads data, builds K, computes alpha = K^-1 y
│   ├── ProblemSetup.py         # setup_problem(): GPModel + presolve + scaled box + solver settings
│   ├── BranchBound1.py         # B&B driver: preprocessing, Node, branching, pruning, main loop
│   ├── LowerBound.py           # get_LB(): node bounding = analytical part + PWL MIQCP part + local UB
│   ├── Analytical_LB.py        # Distance bounds, importance split, closed-form bounds
│   ├── PWL_Custom_LB.py        # Gurobi MIQCP with sign-aware PWL objective (setPWLObj)
│   └── UpperBound.py           # Multistart L-BFGS-B on the true posterior mean
├── benchmarks/
│   ├── common.py               # Test problems, GP fitting (BoTorch), instance cache, result logging
│   ├── run_palm.py             # Run PALM-Mean (src/, main.py settings) on a benchmark
│   ├── run_scip.py             # Same instances with SCIP
│   ├── run_maingo.py           # Same instances with MAiNGO
│   ├── run_baron.py            # Same instances with BARON (via Pyomo)
│   ├── requirements.txt        # Extra packages for the benchmarks
│   └── data/                   # AutoAM and N-benzylation data sets
└── figure/
    ├── animate_palm_mean_1d.py # Stand-alone 1-D re-implementation used to make the demo GIF
    └── palm_mean_1d(3).gif     # Demo animation
```

## Method

### Problem

For a GP with an ARD squared-exponential kernel, the posterior mean is

$$
\mu(x)=\sum_{i=1}^{N}\alpha_i\,k_i(x),\qquad
k_i(x)=\sigma_f^2\exp\!\left(-\tfrac12 d_i(x)\right),\qquad
d_i(x)=\lVert L^{-1}(x-x_i)\rVert_2^2,
$$

where $L=\mathrm{diag}(\ell_1,\dots,\ell_d)$ and $\alpha=(K_{XX}+\sigma_\varepsilon^2I)^{-1}y$. PALM-Mean solves $\min_{x\in[x^L,x^U]}\mu(x)$ to global optimality. This is hard because $\mu$ is a sum of $N$ bumps whose coefficients $\alpha_i$ can have either sign, so it is highly multimodal.

### Key idea: bound each kernel term through a single scalar

Each $k_i$ depends on $x$ only through the scalar $d_i(x)$, and $\sigma_f^2 e^{-d/2}$ is decreasing and convex in $d$. On a box $X_h$:

1. **Distance range (closed form).**
   - $d_i^L$ is the squared distance from $x_i$ to its projection onto the box.
   - $d_i^U$ is the squared distance from $x_i$ to the farthest vertex, computed one coordinate at a time.

   This gives $k_i^U=\sigma_f^2e^{-d_i^L/2}$ and $k_i^L=\sigma_f^2e^{-d_i^U/2}$.
2. **Sign-aware bound.** A positive term is smallest when the kernel is smallest, and a negative term when the kernel is largest. So $\alpha_i k_i(x)\ge \alpha_i^+k_i^L-\alpha_i^-k_i^U$.
3. **Importance split.** The score $\beta_{i,h}=|\alpha_i|\,k_i^U$ bounds how much term $i$ can contribute on $X_h$.
   - Terms with $\beta_{i,h}<\rho$ are **unimportant**. They get the cheap analytical bound from step 2, whose error is less than $\rho$.
   - Terms with $\beta_{i,h}\ge\rho$ are **important**. They get the tighter piecewise-linear relaxation below.
4. **PWL relaxation in $d$ (important terms).** Split $[d_i^L,d_i^U]$ into $S$ segments.
   - $\alpha_i>0$: the term is convex in $d$. Under-estimate it with the **maximum of the two end-point tangents** on each segment.
   - $\alpha_i<0$: the term is concave in $d$. Under-estimate it with **secants (chords)**.

   Both envelopes are exact at the breakpoints and converge to the true term as the box shrinks.

The node lower bound is

$$
\mathrm{LB}(X_h)=\min_{x\in X_h,\;d}\;\sum_{i\in\mathcal I(h)}\mathrm{PWL}_i(d_i)\;+\sum_{i\in\mathcal U(h)}\big(\alpha_i^+k_i^L-\alpha_i^-k_i^U\big)
\quad\text{s.t.}\quad d_i=\lVert L^{-1}(x-x_i)\rVert^2,\; d_i\in[d_i^L,d_i^U].
$$

This is a nonconvex MIQCP: the PWL pieces add binary variables, and the distance equalities are nonconvex quadratics. Gurobi solves it, and its **best bound** (`ObjBound`) is used. That value is a valid lower bound even if Gurobi stops early. Only the important terms enter the MIQCP, so its size depends on how many terms matter locally, not on $N$.

### Branch-and-bound

- **Upper bound:** multistart L-BFGS-B on the true $\mu$ inside each node. Any point found is feasible, so its value is a valid upper bound.
- **Node selection:** best-first, i.e. the node with the smallest lower bound.
- **Branching:** bisect the longest edge, measured in lengthscale-scaled coordinates.
- **Fathoming:** a node is removed if any of the following holds:
  - its LB is above the incumbent $\mathrm{UB}^*$;
  - its own gap $\mathrm{UB}_h-\mathrm{LB}_h$ meets $\varepsilon_{\rm abs}$ or $\varepsilon_{\rm rel}$;
  - $\mathrm{UB}^*-\mathrm{LB}_h$ meets $\varepsilon_{\rm abs}$ or $\varepsilon_{\rm rel}$.
- **Convergence:** the PWL envelope gap vanishes as node diameters go to zero, so the algorithm returns an $\varepsilon$-global minimizer (see the paper for the proof).

## How the code works

```
main.py
 └─ setup_problem(...)                     src/ProblemSetup.py
     ├─ GPModel(param_path, X_path, Y_path)  src/GPModel.py  -> x, y, ℓ, σ_f, α = K⁻¹y
     ├─ get_UB(..., n_multi=5) on [0,1]^d    presolve: initial incumbent UB*
     └─ convert to scaled space z = x / ℓ    (all B&B boxes live in z-space)
 └─ BranchBound(time_limit, GP_obj)         src/BranchBound1.py
     ├─ PrePross: split root into ≤ 500 boxes, evaluate each one as a Node
     │   (if the time limit is hit, remaining boxes get an analytical-only LB)
     └─ loop until no active node or time limit:
         ├─ pick node with min LB; update UB* from node upper bounds
         ├─ BranchOrPrune: prune (LB > UB*, node converged, or within tolerance of UB*)
         └─ or SpatialBranching: bisect longest edge -> 2 child Nodes

Node(xL, xU) -> get_LB(bounds)              src/LowerBound.py
 ├─ get_analytic(bounds)                    src/Analytical_LB.py
 │    d_i^L (projection), d_i^U (farthest vertex), β_i, split into
 │    unimportant (Σ analytical bound = Sigma1) / important (index2)
 ├─ get_UB(bounds, n_multi)                 src/UpperBound.py   -> node UB and its argmin
 └─ PWL_LB(important terms)                 src/PWL_Custom_LB.py
      Gurobi model: x ∈ box, one d_i variable per important term,
      d_i = ||(x - x_i)/ℓ||² (two quadratic inequalities),
      setPWLObj(d_i, breakpoints, values):
        α_i < 0 -> chords through S+1 points on the curve
        α_i > 0 -> points + tangent intersections (max-of-tangents envelope)
      returns m.ObjBound = Sigma2
 LB = Sigma1 + Sigma2
```

Implementation details:

- **Adaptive MIQCP tolerance.** Nodes whose local upper bound is far above the incumbent don't need a tight lower bound.
  - If $\mathrm{UB}_h-\mathrm{UB}^*>0.51$, Gurobi's `MIPGapAbs` is set to $\mathrm{UB}_h-\mathrm{UB}^*-0.5$.
  - Otherwise it is set to $0.01$.
  - The relative gap is set to $10^{-3}$.
  - Each node MIQCP has a 10 s time limit. If Gurobi hits it, `ObjBound` is still a valid bound.
- **Preprocessing.** The number of partitions per dimension is proportional to the box side length, so the sub-boxes have similar diagonals. The count is reduced until there are at most 500 boxes.
- **Coordinates.** All bounding works in lengthscale-scaled coordinates $z=x/\ell$, where the distance is the plain Euclidean norm. Results are mapped back to $x$ before printing.

## Benchmarks

### Running the benchmarks

[`benchmarks/`](benchmarks) reproduces the paper's test problems and compares PALM-Mean with SCIP, MAiNGO and BARON. PALM-Mean is the unmodified solver in `src/` with the settings in `main.py`. Install the extra packages first:

```sh
pip install -r benchmarks/requirements.txt   # botorch, gpytorch, pandas, pyscipopt, maingopy, pyomo
```

| `--problem` | Description | Default replicates | Time limit |
|---|---|---|---|
| `eggholder` | $d$-dimensional EggHolder on $[-512,512]^d$ | 10 | 600 s |
| `schwefel` | Schwefel on $[-500,500]^d$ | 40 | 600 s |
| `gp_prior` | Functions drawn from a GP prior; each seed draws $d\in\{4,\dots,9\}$ and $N\in\{300,\dots,599\}$ | 100 | 600 s |
| `autoam` | Autonomous additive manufacturing data ($d=4$, $N=100$), maximize Score | 1 | 3600 s |
| `benzylation` | N-benzylation flow reaction ($d=4$, $N=78$), minimize −STY + 100·max(impurity − 5 %, 0) | 1 | 600 s |

**Data sources** (copies are in [`benchmarks/data/`](benchmarks/data)):
- **AutoAM:** [`AutoAM_dataset.csv`](https://github.com/PV-Lab/Benchmarking/blob/main/datasets/AutoAM_dataset.csv) from the [PV-Lab/Benchmarking](https://github.com/PV-Lab/Benchmarking) repository.
- **N-benzylation:** A. M. Schweidtmann, A. D. Clayton, N. Holmes, E. Bradford, R. A. Bourne, A. A. Lapkin. *Machine learning meets continuous flow chemistry: Automated optimization towards the Pareto front of multiple objectives.* Chemical Engineering Journal 352 (2018) 277–282. [doi:10.1016/j.cej.2018.07.031](https://www.sciencedirect.com/science/article/pii/S1385894718312634)

```sh
# EggHolder, d = 2, four data sizes, 10 seeds each
python benchmarks/run_palm.py   --problem eggholder --dim 2 --n 100 500 1000 1500
python benchmarks/run_scip.py   --problem eggholder --dim 2 --n 100 500 1000 1500
python benchmarks/run_maingo.py --problem eggholder --dim 2 --n 100 500 1000 1500
python benchmarks/run_baron.py  --problem eggholder --dim 2 --n 100 500 1000 1500

# Schwefel: d = 6 with increasing N, and N = 400 with increasing d (16 Gurobi threads)
python benchmarks/run_palm.py --problem schwefel --dim 6 --n 100 200 300 400 500 600 --threads 16
python benchmarks/run_palm.py --problem schwefel --dim 3 4 5 6 7 8 9 10 --n 400 --threads 16

# GP prior and case studies
python benchmarks/run_palm.py --problem gp_prior
python benchmarks/run_palm.py --problem autoam --threads 16
python benchmarks/run_palm.py --problem benzylation --threads 16
```

BARON is commercial: `run_baron.py` needs the `baron` executable on the PATH and a valid [BARON license](https://minlp.com/baron-licenses). It exits with a message if BARON is not found.

Other options are `--replicates`, `--seed-start`, `--time-limit`, `--threads`, `--instances` and `--out`; see `--help`.

How the benchmark scripts work:

- **Instances.** For each instance, the training data is generated from Latin hypercube samples seeded by the replicate number. It is identical to `pyDOE2.lhs(d, N, random_state=seed)` as used in the paper. Inputs are scaled to $[0,1]^d$ and outputs standardized. A zero-mean ARD-RBF GP is then fitted with BoTorch, and the result is written to `benchmarks/instances/<name>/` in the same format as `data/`. Instances are cached, so all solvers optimize exactly the same posterior mean. Any instance can also be solved with `main.py` by pointing it to that folder.
- **Results.** Each run appends a row to `benchmarks/results/<solver>_<problem>.csv` with these columns:
  - dimension, $N$, seed
  - time (s) and number of nodes
  - UB, LB, gap
  - `solved`: gap ≤ 0.1 or ≤ 1 % of |UB|

  UB and LB are in standardized output units.
- **Timing.** PALM-Mean reports wall-clock time of the B&B, SCIP reports `getSolvingTime()`, MAiNGO reports `get_cpu_solution_time()`, and BARON reports the solver time returned through Pyomo.
- **Windows only.** On Windows, SCIP runs in a separate process: pyscipopt and torch each bundle an Intel OpenMP runtime, and they crash if both are loaded in one process.

### Results reported in the paper

**Setup used in the paper**
- Comparison: PALM-Mean against SCIP, MAiNGO and BARON, all deterministic global solvers.
- PALM-Mean was run on 1 core and on 16 cores (multithreaded Gurobi node solves).
- Settings: $\rho=0.01$, 6 PWL segments per term, $\varepsilon_{\rm abs}=0.1$, $\varepsilon_{\rm rel}=0.01$.

**Synthetic problems**

- **EggHolder (2-D)**, $N\in\{100,500,1000,1500\}$, 10 replicates, 600 s limit.
  - PALM-Mean solved all instances on a single core.
  - SCIP was the strongest baseline up to $N=1000$, but its solved fraction dropped sharply at $N=1500$.
- **Schwefel**, 40 replicates, 600 s limit, in two studies.
  - *$d=6$ with increasing $N$:* PALM-Mean (16 cores) had the lowest average CPU time and solved nearly all instances up to $N=500$.
  - *$N=400$ with increasing $d$:* PALM-Mean was the most reliable across the range of dimensions.
- **Functions drawn from a GP prior**: 100 instances with $d\in\{6,\dots,10\}$ and $N\in\{200,\dots,600\}$.
  - PALM-Mean (16 cores) beat SCIP in both solved fraction and solving rate.
  - Single-core PALM-Mean was competitive with SCIP.

**Real-world case studies**

| Case | Method | B&B nodes | CPU time (s) | Final abs. gap |
|---|---|---:|---:|---:|
| N-benzylation ($d=4$, $N=78$, limit 600 s) | **PALM-Mean (16 cores)** | 110 | **15.4** | 0.10 |
| | PALM-Mean (1 core) | 110 | 33.5 | 0.10 |
| | SCIP | 66,222 | 106.9 | 0.10 |
| | MAiNGO | 609,292 | 600 (limit) | 3.52 |
| | BARON | 45,861 | 600 (limit) | 21.81 |
| AutoAM ($d=4$, $N=100$, limit 3600 s) | **PALM-Mean (16 cores)** | 4,423 | **1,074.1** | 0.10 |
| | PALM-Mean (1 core) | 3,163 | 3,600 (limit) | 0.19 |
| | SCIP | 523,427 | 1,734.8 | 0.10 |
| | MAiNGO | 945,300 | 3,600 (limit) | 6.64 |
| | BARON | 69,958 | 3,600 (limit) | 23.15 |

PALM-Mean needs 2–3 orders of magnitude fewer B&B nodes than general-purpose solvers, because each node bound is much tighter.

## Limitations

- Only the **squared-exponential (RBF) kernel** is implemented. The paper also covers Matérn-1/2, 3/2 and 5/2 kernels; support for these will be released later.
- The search domain is hard-coded to the **unit box** $[0,1]^d$, and a **zero prior mean** is assumed.
- Each locally important term adds a distance variable, a quadratic constraint and a PWL objective to the node MIQCP. Node cost therefore grows with the number of important terms; increase `hyperparameter` ($\rho$) to trade bound tightness for speed.

## Citation

```bibtex
@article{tang2026palmmean,
  title   = {An Efficient Spatial Branch-and-Bound Algorithm for Global Optimization of Gaussian Process Posterior Mean Functions},
  author  = {Tang, Wei-Ting and Kudva, Akshay and Tsay, Calvin and Paulson, Joel A.},
  journal = {arXiv preprint arXiv:2605.00855},
  year    = {2026}
}
```

## License

[MIT](LICENSE)
