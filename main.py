#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Find the global minimum of a trained GP posterior mean over [0,1]^d with PALM-Mean.

Examples:
    python main.py
    python main.py --time-limit 1200 --threads 16
    python main.py --params my_gp/params.json --X my_gp/X.npy --Y my_gp/Y.npy --no-preprocess
"""
import argparse
import torch
from src.BranchBound1 import BranchBound
from src.ProblemSetup import setup_problem
from src.Report import print_results


def parse_args():
    parser = argparse.ArgumentParser(description="Find the global minimum of a trained GP posterior mean with PALM-Mean.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    data = parser.add_argument_group("trained GP")
    data.add_argument("--params", default="data/params.json", help="GP hyperparameters (JSON)")
    data.add_argument("--X", default="data/X.npy", help="training inputs, scaled to [0,1]^d")
    data.add_argument("--Y", default="data/Y.npy", help="training outputs, centered")

    solver = parser.add_argument_group("solver")
    solver.add_argument("--time-limit", type=float, default=600, help="wall-clock limit for preprocessing + B&B (s)")
    solver.add_argument("--threads", type=int, default=1, help="threads per Gurobi MIQCP solve (also used by torch)")
    solver.add_argument("--abs-gap", type=float, default=0.1, help="absolute optimality tolerance")
    solver.add_argument("--rel-gap", type=float, default=0.01, help="relative optimality tolerance (a node is fathomed when either is met)")
    solver.add_argument("--rho", type=float, default=0.01,
                        help="importance threshold: terms with max |alpha_i k_i| below rho are bounded analytically")
    solver.add_argument("--pwl-segments", type=int, default=6, help="piecewise-linear segments per important kernel term")
    solver.add_argument("--preprocess", action=argparse.BooleanOptionalAction, default=True,
                        help="pre-partition the root box into at most 500 sub-boxes")
    solver.add_argument("--n-prepart", type=int, default=10, help="starting number of partitions along the diagonal for preprocessing")
    solver.add_argument("--n-multi", type=int, default=2, help="L-BFGS-B starts per node for the upper bound")

    output = parser.add_argument_group("output")
    output.add_argument("--log-every", type=int, default=100, help="print a progress line every N generated nodes (0: off)")
    output.add_argument("--seed", type=int, default=None, help="random seed for the local-search starts; omit for an unseeded run")
    return parser.parse_args()


if __name__=='__main__':

    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)

    GP_obj = setup_problem(Preprocess=args.preprocess, Thread_gurobi=args.threads, PWL_segment=args.pwl_segments,
                           n_prepart=args.n_prepart, hyperparameter=args.rho, abs_gap=args.abs_gap, rel_gap=args.rel_gap,
                           n_multi=args.n_multi, log_every=args.log_every,
                           param_path=args.params, X_path=args.X, Y_path=args.Y)

    BranchBound(args.time_limit, GP_obj)
    print_results(GP_obj)
