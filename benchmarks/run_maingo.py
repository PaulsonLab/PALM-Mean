#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solve the same benchmark instances with MAiNGO (McCormick-based reduced-space B&B).

Examples:
    python benchmarks/run_maingo.py --problem eggholder --dim 2 --n 100 500 1000 1500
    python benchmarks/run_maingo.py --problem schwefel --dim 6 --n 100
"""
import maingopy as mg
from common import parse_args, iterate_instances, load_instance, write_result, PALM_SETTINGS


class GPMeanModel(mg.MAiNGOmodel):
    def __init__(self, alpha, x_center, zU, f_scale):
        mg.MAiNGOmodel.__init__(self) # Should be there for technical reasons
        self.alpha = alpha
        self.x_center = x_center
        self.zU = zU
        self.f_scale = f_scale

    def get_variables(self):
        return [mg.OptimizationVariable(mg.Bounds(0.0, self.zU[j]), mg.VT_CONTINUOUS, "z%d" % (j+1)) for j in range(len(self.zU))]

    def evaluate(self, z):
        mean = 0
        for i in range(len(self.alpha)):
            distance_square = 0
            for j in range(len(self.zU)):
                distance_square += (z[j] - self.x_center[i][j])**2
            mean += self.alpha[i]*self.f_scale**2*mg.exp(-0.5*distance_square)

        result = mg.EvaluationContainer()
        result.objective = mean
        return result


if __name__ == '__main__':

    args = parse_args("Solve benchmark instances with MAiNGO")

    for path, seed in iterate_instances(args):
        gp, meta = load_instance(path)
        Z = (gp.x/gp.l).tolist() # convert x to z space
        zU = (1/gp.l).flatten().tolist()
        model = GPMeanModel(gp.alpha.flatten().tolist(), Z, zU, float(gp.sigma_f))

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
