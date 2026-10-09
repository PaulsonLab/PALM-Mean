#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run PALM-Mean (the solver in ../src, with the settings of main.py) on a benchmark.

Examples:
    python benchmarks/run_palm.py --problem eggholder --dim 2 --n 100 500 1000 1500
    python benchmarks/run_palm.py --problem schwefel --dim 6 --n 100 200 300 400 500 600 --threads 16
    python benchmarks/run_palm.py --problem gp_prior
    python benchmarks/run_palm.py --problem autoam --threads 16
    python benchmarks/run_palm.py --problem benzylation --threads 16
"""
import os
import time
import json
import torch
from common import parse_args, iterate_instances, write_result, PALM_SETTINGS
from src.BranchBound1 import BranchBound
from src.Build_GP import Build_GP

if __name__ == '__main__':

    args = parse_args("Run PALM-Mean on a benchmark problem")
    torch.set_num_threads(args.threads)

    for path, seed in iterate_instances(args):
        with open(os.path.join(path, "meta.json")) as f:
            meta = json.load(f)

        torch.manual_seed(seed) # random multistarts of the local upper-bounding solver
        GP_obj = Build_GP(**PALM_SETTINGS, Thread_gurobi=args.threads, param_path=os.path.join(path, "params.json"),
                          X_path=os.path.join(path, "X.npy"), Y_path=os.path.join(path, "Y.npy"))

        start_time = time.monotonic()
        BestUpperBound, BestLowerBound, node_count, BestUpperBound_arg = BranchBound(args.time_limit, GP_obj)
        BB_time = time.monotonic() - start_time

        write_result(args, "palm", {"dim": meta["dim"], "n": meta["n"], "seed": seed, "time_s": BB_time,
                                    "nodes": node_count, "ub": float(BestUpperBound), "lb": float(BestLowerBound),
                                    "x_best": [round(v, 6) for v in torch.as_tensor(BestUpperBound_arg).flatten().tolist()]})
