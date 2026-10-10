#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solve the same benchmark instances with MAiNGO (McCormick-based reduced-space B&B).

Examples:
    python benchmarks/run_maingo.py --problem eggholder --dim 2 --n 100 500 1000 1500
    python benchmarks/run_maingo.py --problem schwefel --dim 6 --n 100
"""
import maingopy as mg
from kernel_expression import kernel_expression
from common import parse_args, iterate_instances, load_instance, scaled_problem, write_result, PALM_SETTINGS


class GPMeanModel(mg.MAiNGOmodel):
    def __init__(self, Z, zU, alpha, s_f, kernel_name):
        mg.MAiNGOmodel.__init__(self) # required by MAiNGO
        self.kernel_name = kernel_name
        self.Z = Z
        self.zU = zU
        self.alpha = alpha
        self.s_f = s_f

    def get_variables(self):
        return [mg.OptimizationVariable(mg.Bounds(0.0, self.zU[j]), mg.VT_CONTINUOUS, "z%d" % (j+1)) for j in range(len(self.zU))]

    def evaluate(self, z):
        mean = 0
        for i in range(len(self.alpha)):
            distance_square = 0
            for j in range(len(self.zU)):
                distance_square += (z[j] - self.Z[i][j])**2
            mean += self.alpha[i]*self.s_f**2*kernel_expression(self.kernel_name, distance_square, mg.sqrt, mg.exp)

        result = mg.EvaluationContainer()
        result.objective = mean
        return result


if __name__ == '__main__':

    args = parse_args("Solve benchmark instances with MAiNGO")

    for path, seed in iterate_instances(args):
        gp, meta = load_instance(path)
        model = GPMeanModel(*scaled_problem(gp))

        solver = mg.MAiNGO(model)
        solver.set_option("maxTime", args.time_limit)
        solver.set_option("epsilonA", PALM_SETTINGS["abs_gap"])
        solver.set_option("epsilonR", PALM_SETTINGS["rel_gap"])
        solver.set_option("loggingDestination", mg.LOGGING_OUTSTREAM)
        solver.set_option("outstreamVerbosity", 0)
        solver.set_option("writeResultFile", 0)
        solver.solve()

        write_result(args, "maingo", {"dim": meta["dim"], "n": meta["n"], "seed": seed,
                                      "time_s": solver.get_cpu_solution_time(), "nodes": solver.get_iterations(),
                                      "ub": solver.get_objective_value(), "lb": solver.get_final_LBD()})
