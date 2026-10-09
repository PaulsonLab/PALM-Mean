#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Find the global minimum of a trained GP posterior mean over [0,1]^d with PALM-Mean.
"""
import torch
from src.BranchBound1 import BranchBound
from src.ProblemSetup import setup_problem
from src.Report import print_results

if __name__=='__main__':

    time_limit = 600 # wall-clock limit for preprocessing + B&B (s)
    Thread_gurobi = 1 # threads per Gurobi MIQCP solve (also used by torch)

    settings = {
        "Preprocess": True,      # pre-partition the root box into <= 500 sub-boxes
        "Thread_gurobi": Thread_gurobi,
        "PWL_segment": 6,        # piecewise-linear segments per important kernel term
        "n_prepart": 10,         # starting number of partitions along the diagonal for preprocessing
        "hyperparameter": 0.01,  # importance threshold rho: terms with max |alpha_i k_i| < rho are bounded analytically
        "abs_gap": 0.1,          # absolute optimality tolerance
        "rel_gap": 0.01,         # relative optimality tolerance (a node is fathomed when either is met)
        "n_multi": 2,            # L-BFGS-B starts per node for the upper bound
    }

    # Trained GP: hyperparameters, training inputs (scaled to [0,1]^d) and centered outputs
    paths = {"param_path": "data/params.json", "X_path": "data/X.npy", "Y_path": "data/Y.npy"}

    torch.set_num_threads(Thread_gurobi)
    GP_obj = setup_problem(**settings, **paths)

    BranchBound(time_limit, GP_obj)
    print_results(GP_obj)
