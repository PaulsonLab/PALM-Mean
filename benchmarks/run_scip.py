#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solve the same benchmark instances with SCIP (full-space formulation of the posterior mean).

The posterior mean is written in lengthscale-scaled coordinates z = x/l as
    t = sum_k alpha_k sigma_f^2 k(||z - z_k||),   z in [0, 1/l],
and SCIP minimizes t, whose bounds are given by interval arithmetic.

Examples:
    python benchmarks/run_scip.py --problem eggholder --dim 2 --n 100 500 1000 1500
    python benchmarks/run_scip.py --problem gp_prior
"""
# Only numpy and pyscipopt are imported at module level: SCIP runs in a child process that must
# not load torch, because pyscipopt and torch each bundle an Intel OpenMP runtime that clash on Windows.
from pyscipopt import Model, quicksum, exp, sqrt
from kernel_expression import kernel_expression


def solve_SCIP(Z, zU, alpha, s_f, kernel_name, mu_LB, mu_UB, abs_gap, rel_gap, time_limit):
    dim = len(zU)

    model = Model("GP posterior mean")
    model.hideOutput()
    z = [model.addVar(vtype="C", lb=0.0, ub=zU[j]) for j in range(dim)]

    mean = 0
    for k in range(len(Z)):
        distance_square = quicksum((z[d] - Z[k][d])**2 for d in range(dim))
        mean += alpha[k]*s_f**2*kernel_expression(kernel_name, distance_square, sqrt, exp)

    t = model.addVar(vtype="C", lb=mu_LB, ub=mu_UB)
    model.addCons(t == mean)

    model.setRealParam('limits/absgap', abs_gap)
    model.setRealParam('limits/gap', rel_gap)
    model.setRealParam('limits/time', time_limit)
    model.setObjective(t, 'minimize')
    model.optimize()

    return model.getDualbound(), model.getPrimalbound(), model.getSolvingTime(), model.getNTotalNodes()


if __name__ == '__main__':
    import multiprocessing as mp
    from common import parse_args, iterate_instances, load_instance, scaled_problem, interval_bounds, write_result, PALM_SETTINGS

    args = parse_args("Solve benchmark instances with SCIP")

    with mp.get_context("spawn").Pool(1) as pool:
        for path, seed in iterate_instances(args):
            gp, meta = load_instance(path)
            LB, UB, solve_time, n_node = pool.apply(solve_SCIP, (*scaled_problem(gp), *interval_bounds(gp),
                                                                 PALM_SETTINGS["abs_gap"], PALM_SETTINGS["rel_gap"], args.time_limit))
            write_result(args, "scip", {"dim": meta["dim"], "n": meta["n"], "seed": seed, "time_s": solve_time,
                                        "nodes": n_node, "ub": UB, "lb": LB})
