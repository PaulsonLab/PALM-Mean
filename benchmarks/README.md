# PALM-Mean benchmarks

This folder reproduces the test problems of the [paper](https://arxiv.org/abs/2605.00855) and compares PALM-Mean with SCIP, MAiNGO and BARON. PALM-Mean is the unmodified solver in [`src/`](../src) with the default settings of [`main.py`](../main.py); see the [main README](../README.md) for installation and the method, and the [paper](https://arxiv.org/abs/2605.00855) for the results.

## Running the benchmarks

Run all commands from the repository root. Install the extra packages first:

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

**Data sources** (copies are in [`data/`](data)):
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

To run specific seeds, list them with `--seeds`. It overrides `--replicates`, and the instance for a given seed is the same for every solver:

```sh
# Schwefel, d = 6, N = 500, seeds 7 and 9 only
python benchmarks/run_palm.py   --problem schwefel --dim 6 --n 500 --seeds 7 9 --threads 16
python benchmarks/run_scip.py   --problem schwefel --dim 6 --n 500 --seeds 7 9
python benchmarks/run_maingo.py --problem schwefel --dim 6 --n 500 --seeds 7 9
```

Use `--kernel` to fit a Matern GP instead of the default RBF, e.g. `python benchmarks/run_palm.py --problem schwefel --dim 6 --n 500 --seeds 7 --kernel matern32`. Matern instances and results are stored separately, with the kernel name appended to their names.

Other options are `--kernel`, `--replicates` (run seeds `0..R-1`), `--seed-start`, `--time-limit`, `--threads`, `--instances` and `--out`; see `--help`. With `--report`, `run_palm.py` also prints progress every 100 nodes and the full results summary of every run.

## How the benchmark scripts work

- **Instances.** For each instance, the training data is generated from Latin hypercube samples seeded by the replicate number. It is identical to `pyDOE2.lhs(d, N, random_state=seed)` as used in the paper. Inputs are scaled to $[0,1]^d$ and outputs standardized. A zero-mean ARD GP is then fitted with BoTorch, with an RBF kernel or, with `--kernel matern12|matern32|matern52`, a Matern kernel, and the result is written to `benchmarks/instances/<name>/` in the same format as the example GP in [`../data/`](../data). Instances are cached, so all solvers optimize exactly the same posterior mean. Any instance can also be solved with `main.py`, e.g. `python main.py --params benchmarks/instances/<name>/params.json --X benchmarks/instances/<name>/X.npy --Y benchmarks/instances/<name>/Y.npy`.
- **Results.** Each run appends a row to `benchmarks/results/<solver>_<problem>.csv` with these columns:
  - dimension, $N$, seed
  - time (s) and number of nodes
  - UB, LB, gap
  - `solved`: gap ≤ 0.1 or ≤ 1 % of |UB|

  UB and LB are in standardized output units.
- **Timing.** PALM-Mean reports wall-clock time of the B&B, SCIP reports `getSolvingTime()`, MAiNGO reports `get_cpu_solution_time()`, and BARON reports the solver time returned through Pyomo.
- **Windows only.** On Windows, SCIP runs in a separate process: pyscipopt and torch each bundle an Intel OpenMP runtime, and they crash if both are loaded in one process.

## Example: EggHolder with 1000 training points

One replicate (seed 0) of the 2-D EggHolder benchmark with $N = 1000$, solved with 16 Gurobi threads:

```sh
python benchmarks/run_palm.py --problem eggholder --dim 2 --n 1000 --replicates 1 --threads 16 --report
```

The output below omits Gurobi's license message.

- **Progress table.** A row is printed every 100 generated nodes. Branching adds nodes in pairs after the 99 preprocessing boxes, so the rows appear at 101, 201, and so on.
- **Preprocessing rows.** Runs with at least 100 preprocessing boxes, such as `python main.py` on `data/`, also print rows while preprocessing. These show the boxes bounded so far, e.g. `100/343`. Their lower-bound and gap columns show `-`, since the boxes not yet bounded could still contain lower values.
- **Summary.** It is printed when the search ends.

```
    ____  ___    __    __  ___      __  ___
   / __ \/   |  / /   /  |/  /     /  |/  /__  ____ _____
  / /_/ / /| | / /   / /|_/ /_____/ /|_/ / _ \/ __ `/ __ \
 / ____/ ___ |/ /___/ /  / /_____/ /  / /  __/ /_/ / / / /
/_/   /_/  |_/_____/_/  /_/     /_/  /_/\___/\__,_/_/ /_/

 Global optimization of Gaussian process posterior means
 Developed by Wei-Ting (Jonathan) Tang, Akshay Kudva, Calvin Tsay, and Joel Paulson

 d = 2, N = 1000 training points | time limit 600 s | 16 Gurobi thread(s)

+---------+--------+-------------+-------------+------------+----------+----------+---------------+
|   Nodes |   Open |     Best UB | Lower bound |    Abs gap |  Rel gap | Time (s) | Stage         |
+---------+--------+-------------+-------------+------------+----------+----------+---------------+
|     101 |    100 |   -4.535390 | -135.445192 | 130.909802 | 2886.41% |     19.4 | branching     |
|     201 |    150 |   -4.535390 |  -10.294490 |   5.759100 |  126.98% |     30.6 | branching     |
|     301 |    200 |   -4.535390 |   -6.346114 |   1.810724 |   39.92% |     38.3 | branching     |
|     401 |    250 |   -4.535390 |   -5.238202 |   0.702811 |   15.50% |     45.5 | branching     |
|     501 |    300 |   -4.535390 |   -4.911225 |   0.375835 |    8.29% |     51.6 | branching     |
+---------+--------+-------------+-------------+------------+----------+----------+---------------+

===================================================================================================
 PALM-Mean result
===================================================================================================
 Status               : Converged: global minimum certified within tolerance
 Minimum value        : -4.535390   (posterior mean at x*, best upper bound)
 Minimum (original)   : -1394.99   (in the units of the original outputs)
 Lower bound          : -4.629221   (certified: mean >= this everywhere in the box)
 Absolute gap         : 0.093831   (tolerance 0.1)
 Relative gap         : 2.07 %     (tolerance 1 %)
 x* (original)        : [512, 392.523]
---------------------------------------------------------------------------------------------------
 Problem              : d = 2, N = 1000 training points
 Settings             : rbf kernel, rho = 0.01, 6 PWL segments, 16 Gurobi thread(s)
 Nodes bounded        : 529 (99 preprocessing boxes + 430 from branching)
 Time                 : 53.5 s (preprocessing 19.2 s, B&B 34.3 s)
===================================================================================================
```

- **Same optimum as the paper.** The minimum value matches the paper's run of this instance to six decimals (−4.535391; 655 nodes, 42.6 s on 16 cores of a Xeon Gold 6444Y).
- **Timing depends on the machine.** Run times differ with hardware.
- **Values are for the GP surrogate, not the true function.** x\* lies close to EggHolder's true global minimizer (512, 404.2). The minimum in original units, however, is that of the GP posterior mean: the true EggHolder value is −820 at x\* and −960 at its global minimum. With a finite data set the surrogate overshoots near the boundary, where training points are sparse.
