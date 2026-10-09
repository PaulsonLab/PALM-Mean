#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run PALM-Mean (the solver in ../src, with the settings of main.py) on a benchmark.

Examples:
    python benchmarks/run_palm.py --problem eggholder --dim 2 --n 100 500 1000 1500
    python benchmarks/run_palm.py --problem eggholder --dim 2 --n 1000 --replicates 1 --threads 16 --report
    python benchmarks/run_palm.py --problem schwefel --dim 6 --n 100 200 300 400 500 600 --threads 16
    python benchmarks/run_palm.py --problem gp_prior
    python benchmarks/run_palm.py --problem autoam --threads 16
    python benchmarks/run_palm.py --problem benzylation --threads 16
"""
import time
import torch
from common import parse_args, iterate_instances, instance_paths, load_meta, write_result, PALM_SETTINGS
from src.BranchBound1 import BranchBound
from src.ProblemSetup import setup_problem
from src.Report import print_results

if __name__ == '__main__':

    args = parse_args("Run PALM-Mean on a benchmark problem", palm=True)
    torch.set_num_threads(args.threads)

    for path, seed in iterate_instances(args):
        meta = load_meta(path)

        torch.manual_seed(seed) # random multistarts of the local upper-bounding solver
        GP_obj = setup_problem(**PALM_SETTINGS, Thread_gurobi=args.threads, **instance_paths(path))

        start_time = time.monotonic()
        BestUpperBound, BestLowerBound, node_count, BestUpperBound_arg = BranchBound(args.time_limit, GP_obj)
        BB_time = time.monotonic() - start_time
        if args.report:
            print_results(GP_obj)

        write_result(args, "palm", {"dim": meta["dim"], "n": meta["n"], "seed": seed, "time_s": BB_time,
                                    "nodes": node_count, "ub": float(BestUpperBound), "lb": float(BestLowerBound),
                                    "x_best": [round(v, 6) for v in torch.as_tensor(BestUpperBound_arg).flatten().tolist()]})
