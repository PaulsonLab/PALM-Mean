#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Human-readable summary of a PALM-Mean run, from the statistics BranchBound stores in BO_obj.result.
"""
import torch


def print_results(BO_obj):
    """Print status, optimum, bounds, gaps, problem size, settings, node counts and timing."""
    r = BO_obj.result
    UB, LB = r["UB"], r["LB"]
    abs_gap = UB - LB
    rel_gap = abs_gap/abs(UB) if UB != 0 else float('inf')
    converged = abs_gap <= BO_obj.abs_gap or rel_gap <= BO_obj.rel_gap
    x = torch.as_tensor(r["x"]).flatten().tolist()

    if converged:
        status = "Converged: global minimum certified within tolerance"
    elif r["timed_out"]:
        status = "Time limit reached (%g s): gap not closed" % r["time_limit"]
    else:
        status = "Finished with gap above tolerance"
    if r["unexplored_boxes"] > 0:
        status += "\n%22s  (preprocessing stopped early: %d of %d boxes bounded analytically only)" % (
            "", r["unexplored_boxes"], r["preprocess_boxes"])

    line = "=" * 68
    print(line)
    print(" PALM-Mean result")
    print(line)
    print(" %-20s : %s" % ("Status", status))
    print(" %-20s : %.6f" % ("Minimum value", UB) + "   (posterior mean at x*, best upper bound)")
    print(" %-20s : %.6f" % ("Lower bound", LB) + "   (certified: mean >= this everywhere in the box)")
    print(" %-20s : %.6f" % ("Absolute gap", abs_gap) + "   (tolerance %g)" % BO_obj.abs_gap)
    print(" %-20s : %.2f %%" % ("Relative gap", 100*rel_gap) + "     (tolerance %g %%)" % (100*BO_obj.rel_gap))
    print(" %-20s : [%s]" % ("x*", ", ".join("%.6f" % v for v in x)))
    print("-" * 68)
    print(" %-20s : d = %d, N = %d training points" % ("Problem", BO_obj.dim, len(BO_obj.x)))
    print(" %-20s : rho = %g, %d PWL segments, %d Gurobi thread(s)" % (
        "Settings", BO_obj.hyperparameter, BO_obj.PWL_segment, BO_obj.Thread_gurobi))
    nodes = " %-20s : %d (%d preprocessing boxes + %d from branching)" % (
        "Nodes bounded", r["nodes"], r["preprocess_boxes"] - r["unexplored_boxes"], r["nodes"] - r["preprocess_boxes"] + r["unexplored_boxes"])
    print(nodes)
    if not converged:
        print(" %-20s : %d" % ("Open nodes left", r["active_nodes"]))
    print(" %-20s : %.1f s (preprocessing %.1f s, B&B %.1f s)" % (
        "Time", r["time_total"], r["time_preprocess"], r["time_total"] - r["time_preprocess"]))
    print(line)
