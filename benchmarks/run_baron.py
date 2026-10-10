#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solve the same benchmark instances with BARON (through Pyomo).

Requires the BARON executable on the PATH and a BARON license (https://minlp.com/baron-licenses).

Examples:
    python benchmarks/run_baron.py --problem eggholder --dim 2 --n 100 500 1000 1500
    python benchmarks/run_baron.py --problem schwefel --dim 6 --n 100
"""
import sys
import pyomo.environ as pyo
from kernel_expression import kernel_expression
from common import parse_args, iterate_instances, load_instance, scaled_problem, write_result, PALM_SETTINGS


def build_model(gp):
    """Posterior mean in lengthscale-scaled coordinates z = x/l, z in [0, 1/l]."""
    Z, zU, alpha, s_f, kernel_name = scaled_problem(gp)
    dim = len(zU)

    model = pyo.ConcreteModel()
    model.z = pyo.Var(range(dim), bounds={d: (0.0, zU[d]) for d in range(dim)})

    def posterior_mean(model):
        mean = 0
        for k in range(len(Z)):
            distance_square = sum((model.z[d] - Z[k][d])**2 for d in range(dim))
            mean += alpha[k]*s_f**2*kernel_expression(kernel_name, distance_square, pyo.sqrt, pyo.exp)
        return mean

    model.OBJ = pyo.Objective(rule=posterior_mean, sense=pyo.minimize)
    return model


if __name__ == '__main__':

    args = parse_args("Solve benchmark instances with BARON")

    solver = pyo.SolverFactory('baron')
    if not solver.available(exception_flag=False):
        sys.exit("BARON executable not found: install BARON and add it to the PATH.")

    for path, seed in iterate_instances(args):
        gp, meta = load_instance(path)
        model = build_model(gp)

        solver.options['MaxTime'] = args.time_limit
        solver.options['EpsA'] = PALM_SETTINGS["abs_gap"]
        solver.options['EpsR'] = PALM_SETTINGS["rel_gap"]
        results = solver.solve(model, tee=False, load_solutions=False)

        write_result(args, "baron", {"dim": meta["dim"], "n": meta["n"], "seed": seed,
                                     "time_s": float(results['Solver'][0]['Time']),
                                     "nodes": float(results['Problem'][0]['Iterations']),
                                     "ub": float(results['Problem'][0]['Upper bound']),
                                     "lb": float(results['Problem'][0]['Lower bound'])})
