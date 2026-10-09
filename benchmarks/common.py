#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared code for the benchmark scripts: test problems, GP fitting, instance
caching, interval bounds and result logging.

Every benchmark instance is written once to <instances>/<name>/ in the same
format main.py reads (X.npy, Y.npy, params.json), so PALM-Mean, SCIP, MAiNGO
and BARON all solve exactly the same GP posterior mean.
"""
import os
import sys
import csv
import json
import argparse
import numpy as np
import pandas as pd
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from src.GPModel import GPModel
from src.Analytical_LB import distance_bounds, analytic_bound

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))

# PALM-Mean settings, identical to main.py
PALM_SETTINGS = {"Preprocess": True, "PWL_segment": 6, "n_prepart": 10, "hyperparameter": 0.01,
                 "abs_gap": 0.1, "rel_gap": 0.01, "n_multi": 2}


# ---------------------------------------------------------------- test functions
class Schwefel():
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, x):
        return 418.9829*self.dim - torch.sum(x * torch.sin(torch.sqrt(torch.abs(x))), dim=1)


class N_EggHolder():
    # d-dimensional EggHolder (sum of the 2-D EggHolder over consecutive coordinate pairs)
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, x):
        x_i = x[..., :-1]
        x_j = x[..., 1:]
        term1 = -(x_j + 47) * torch.sin(torch.sqrt(torch.abs(x_i / 2 + x_j + 47)))
        term2 = -x_i * torch.sin(torch.sqrt(torch.abs(x_i - (x_j + 47))))
        return torch.sum(term1 + term2, dim=-1)


def lhs(n, samples, random_state):
    """
    Classic Latin hypercube design, identical to pyDOE2.lhs(n, samples, random_state=seed)
    (pyDOE2, BSD-3-Clause) which no longer imports on Python >= 3.12.
    """
    rs = np.random.RandomState(random_state)
    cut = np.linspace(0, 1, samples + 1)
    u = rs.rand(samples, n)
    a = cut[:samples]
    b = cut[1:samples + 1]
    rdpoints = u*(b - a)[:, None] + a[:, None]
    H = np.zeros_like(rdpoints)
    for j in range(n):
        H[:, j] = rdpoints[rs.permutation(range(samples)), j]
    return H


def synthetic_GP_prior(x, dim, lscale):
    from gpytorch.kernels import RBFKernel
    x = x.to(torch.float32)
    covar = RBFKernel(ard_num_dims=dim)
    covar.lengthscale = lscale
    with torch.no_grad():
        K = covar(x, x).to_dense()
    L = torch.linalg.cholesky(K)
    obj = torch.matmul(L, torch.randn((len(x), 1)))
    return obj.to(torch.float64)


# ---------------------------------------------------------------- problems
# bounds: box of every input (None: problem-specific), noise: upper bound of the GP noise variance,
# time_limit: default time limit per instance (s)
PROBLEMS = {
    "eggholder": {"bounds": (-512., 512.), "noise": 1e-4, "time_limit": 600},
    "schwefel":  {"bounds": (-500., 500.), "noise": 1e-4, "time_limit": 600},
    "gp_prior":  {"bounds": (0., 1.),      "noise": 1e-4, "time_limit": 600},
    "autoam":    {"bounds": None,          "noise": 1e-3, "time_limit": 3600},
    "benzylation": {"bounds": None,        "noise": 1e-4, "time_limit": 600},
}


def generate_data(problem, dim, n, seed):
    """Return unscaled training data (X, Y) and box bounds (2 x dim) for one instance."""
    torch.manual_seed(seed)
    if problem in ("eggholder", "schwefel"):
        test_func = N_EggHolder(dim) if problem == "eggholder" else Schwefel(dim)
        lo, hi = PROBLEMS[problem]["bounds"]
        x_min, x_max = lo*torch.ones(dim), hi*torch.ones(dim)
        X = x_min + torch.tensor(lhs(dim, n, seed), dtype=torch.double)*(x_max - x_min)
        Y = test_func(X).unsqueeze(1)

    elif problem == "gp_prior":
        # dimension and data size are drawn per seed, as in the paper's experiments
        dim = int(torch.randint(4, 10, (1,)))
        n = int(torch.randint(300, 600, (1,)))
        x_min, x_max = torch.zeros(dim), torch.ones(dim)
        X = x_min + torch.tensor(lhs(dim, n, seed), dtype=torch.double)*(x_max - x_min)
        lscale = torch.distributions.Gamma(0.5, 10.0).sample((1, dim))
        Y = synthetic_GP_prior(X, dim, lscale)

    elif problem == "autoam":
        # Autonomous additive manufacturing data set; maximize Score -> minimize -Score
        ds = pd.read_csv(os.path.join(BENCH_DIR, "data", "AutoAM_dataset.csv"))
        objective_name = list(ds.columns)[-1]
        feature_name = list(ds.columns)[:-1]
        ds[objective_name] = -ds[objective_name].values
        ds = ds.groupby(feature_name)[objective_name].agg(lambda x: x.unique().mean()).to_frame().reset_index()
        X = torch.tensor(ds[feature_name].values).to(torch.float64)
        Y = torch.tensor(ds[objective_name].values).to(torch.float64).unsqueeze(1)
        x_min = torch.tensor([0.0, 0.0, -1.0, -1.0]).to(torch.float64)
        x_max = torch.tensor([5.0, 10.0, 1.0, 1.0]).to(torch.float64)
        dim, n = 4, len(X)

    elif problem == "benzylation":
        # N-benzylation reaction data (Schweidtmann et al., 2018); minimize -STY with a penalty on impurity > 5 %
        data = pd.read_csv(os.path.join(BENCH_DIR, "data", "benzylation_dataset.csv"), encoding="utf-8-sig")
        X = torch.tensor(data[['6/mL min-1', '07:06', 'Solvent:6', 'Temp/°C']].values).to(torch.float64)
        rho = 100.0 # penalty constant
        impurity = torch.tensor(data['% Impurity'].values).to(torch.float64) - 5.0
        penalty = torch.where(impurity < 0, torch.tensor(0.0, dtype=torch.float64), impurity)*rho
        Y = (-torch.tensor(data['STY/kg m-3 h-1'].values).to(torch.float64) + penalty).unsqueeze(1)
        x_min = torch.tensor([0.2, 1.0, 0.5, 110]).to(torch.float64)
        x_max = torch.tensor([0.4, 5.0, 1.0, 150]).to(torch.float64)
        dim, n = 4, len(X)

    else:
        raise ValueError("Unknown problem: " + problem)

    return X, Y, torch.stack((x_min, x_max)).to(torch.float64), dim, n


def fit_gp(x, y, noise):
    """
    Fit a zero-mean GP with an ARD RBF kernel by maximum likelihood (BoTorch).
    x must already be scaled to [0,1]^d and y standardized.
    """
    import gpytorch
    from gpytorch.kernels import RBFKernel, ScaleKernel
    from gpytorch.constraints import Positive, GreaterThan, Interval
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP

    kernel = ScaleKernel(RBFKernel(ard_num_dims=x.shape[1], lengthscale_constraint=GreaterThan(1e-3)),
                         outputscale_constraint=Positive())
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=Interval(0, noise))
    model = SingleTaskGP(x, y, likelihood=likelihood, covar_module=kernel,
                         mean_module=gpytorch.means.ZeroMean(), outcome_transform=None)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(likelihood, model))

    params = {".kernel.variance": float(model.covar_module.outputscale),
              ".kernel.lengthscales": model.covar_module.base_kernel.lengthscale.detach().flatten().tolist(),
              ".likelihood.variance": float(model.likelihood.noise)}
    return params


def instance_name(problem, dim, n, seed):
    if problem == "gp_prior":
        return "gp_prior_seed%d" % seed
    if problem in ("autoam", "benzylation"):
        return problem
    return "%s_%dD_%dN_seed%d" % (problem, dim, n, seed)


def make_instance(problem, dim, n, seed, root):
    """Generate data, fit the GP and write it in main.py's input format (cached)."""
    path = os.path.join(root, instance_name(problem, dim, n, seed))
    if os.path.exists(os.path.join(path, "meta.json")):
        add_normalization(path) # instances cached before params.json stored the normalization
        return path

    X, Y, bounds, dim, n = generate_data(problem, dim, n, seed)
    x = (X - bounds[0])/(bounds[1] - bounds[0]) # scale to [0,1]^d
    y_mean, y_std = torch.mean(Y), torch.std(Y)
    y = (Y - y_mean)/y_std # standardize
    params = fit_gp(x, y, PROBLEMS[problem]["noise"])

    os.makedirs(path, exist_ok=True)
    np.save(os.path.join(path, "X.npy"), x.numpy())
    np.save(os.path.join(path, "Y.npy"), y.numpy())
    with open(os.path.join(path, "params.json"), "w") as f:
        json.dump(params, f)
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump({"problem": problem, "dim": dim, "n": n, "seed": seed, "bounds": bounds.tolist(),
                   "y_mean": float(y_mean), "y_std": float(y_std)}, f)
    add_normalization(path)
    return path


def add_normalization(path):
    """Store the data normalization in params.json, so results can be reported in original units."""
    meta = load_meta(path)
    with open(os.path.join(path, "params.json")) as f:
        params = json.load(f)
    if "x_lower" in params:
        return
    params.update({"x_lower": meta["bounds"][0], "x_upper": meta["bounds"][1], "y_mean": meta["y_mean"], "y_std": meta["y_std"]})
    with open(os.path.join(path, "params.json"), "w") as f:
        json.dump(params, f)


def instance_paths(path):
    """Keyword arguments locating an instance's GP files, as taken by GPModel and setup_problem."""
    return {"param_path": os.path.join(path, "params.json"), "X_path": os.path.join(path, "X.npy"),
            "Y_path": os.path.join(path, "Y.npy")}


def load_meta(path):
    """Instance metadata: problem, dim, n, seed, original bounds, output mean and std."""
    with open(os.path.join(path, "meta.json")) as f:
        return json.load(f)


def load_instance(path):
    """Return the repo's GPModel object (alpha, lengthscales, sigma_f) and the instance metadata."""
    return GPModel(**instance_paths(path)), load_meta(path)


def scaled_problem(gp):
    """
    The posterior mean in lengthscale-scaled coordinates z = x/l, as plain Python floats for the
    algebraic modeling interfaces of SCIP, MAiNGO and BARON:
        mu(z) = sum_k alpha[k] * sigma_f^2 * exp(-0.5 * ||z - Z[k]||^2),   z in [0, zU].
    """
    Z = (gp.x/gp.l).tolist()
    zU = (1/gp.l).flatten().tolist()
    return Z, zU, gp.alpha.flatten().tolist(), float(gp.sigma_f)


def interval_bounds(gp):
    """Analytical lower/upper bounds of the posterior mean over the whole box (bounds SCIP's objective variable)."""
    alpha = gp.alpha.flatten()
    box = torch.stack((torch.zeros(gp.dim, dtype=torch.float64), (1/gp.l).flatten()))
    dist_near, dist_far = distance_bounds(box, gp.x/gp.l)
    mu_LB = analytic_bound(alpha, gp.sigma_f, dist_near, dist_far)
    mu_UB = -analytic_bound(-alpha, gp.sigma_f, dist_near, dist_far) # upper bound of sum = -(lower bound of -sum)
    return float(mu_LB), float(mu_UB)


# ---------------------------------------------------------------- driver helpers
def parse_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--problem", required=True, choices=list(PROBLEMS))
    parser.add_argument("--dim", type=int, nargs="+", default=[2], help="input dimension(s) (eggholder/schwefel)")
    parser.add_argument("--n", type=int, nargs="+", default=[100], help="number(s) of training points (eggholder/schwefel)")
    parser.add_argument("--replicates", type=int, default=None,
                        help="seeds 0..R-1 (default: 10 eggholder, 40 schwefel, 100 gp_prior, 1 autoam/benzylation)")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--time-limit", type=float, default=None, help="seconds (default: 600, 3600 for autoam)")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--instances", default=os.path.join(BENCH_DIR, "instances"), help="cache directory for GP instances")
    parser.add_argument("--out", default=os.path.join(BENCH_DIR, "results"), help="directory for result CSV files")
    args = parser.parse_args()
    if args.replicates is None:
        args.replicates = {"eggholder": 10, "schwefel": 40, "gp_prior": 100, "autoam": 1, "benzylation": 1}[args.problem]
    if args.time_limit is None:
        args.time_limit = PROBLEMS[args.problem]["time_limit"]
    return args


def iterate_instances(args):
    """Yield (instance path, seed) for every requested configuration."""
    seeds = range(args.seed_start, args.seed_start + args.replicates)
    if args.problem in ("gp_prior", "autoam", "benzylation"):
        configs = [(None, None)]
    else:
        configs = [(d, n) for d in args.dim for n in args.n]
    for dim, n in configs:
        for seed in seeds:
            yield make_instance(args.problem, dim, n, seed, args.instances), seed


def write_result(args, solver, row):
    ub, lb = row["ub"], row["lb"]
    row["gap"] = ub - lb
    row["solved"] = bool(ub - lb <= PALM_SETTINGS["abs_gap"] or ub - lb <= PALM_SETTINGS["rel_gap"]*abs(ub))
    row = {"solver": solver, "problem": args.problem, "threads": args.threads, "time_limit": args.time_limit, **row}

    os.makedirs(args.out, exist_ok=True)
    file = os.path.join(args.out, "%s_%s.csv" % (solver, args.problem))
    new = not os.path.exists(file)
    with open(file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)
    print("[%s] %s dim=%s n=%s seed=%s  time=%.1fs  UB=%.4f  LB=%.4f  gap=%.4f  solved=%s"
          % (solver, args.problem, row["dim"], row["n"], row["seed"], row["time_s"], ub, lb, row["gap"], row["solved"]))
