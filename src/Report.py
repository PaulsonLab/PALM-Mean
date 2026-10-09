#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Human-readable output of a PALM-Mean run: progress lines printed during the search, and the final
summary built from the statistics BranchBound stores in BO_obj.result.
"""
import torch


class ProgressLog():
    """
    Progress lines printed during the search, under a header of column titles. The header is printed
    with the first line, so it is not separated from the table by Gurobi's license banner, which
    appears when the first node MIQCP is built.
    """

    def __init__(self):
        self.header_printed = False

    def line(self, nodes, open_nodes, UB, LB, elapsed, note=""):
        """One progress line. open_nodes and LB are None during preprocessing, when the lower bound
        of the boxes not yet bounded is unknown."""
        if not self.header_printed:
            print("%8s %7s %12s %12s %10s %9s %9s" % ("Nodes", "Open", "Best UB", "Lower bound", "Abs gap", "Rel gap", "Time (s)"), flush=True)
            self.header_printed = True
        if LB is None:
            line = "%8d %7s %12.6f %12s %10s %9s %9.1f   %s" % (nodes, "-", UB, "-", "-", "-", elapsed, note)
        else:
            gap = UB - LB
            rel_gap = "%8.2f%%" % (100*gap/abs(UB)) if UB != 0 else "%9s" % "-"
            line = "%8d %7d %12.6f %12.6f %10.6f %s %9.1f   %s" % (nodes, open_nodes, UB, LB, gap, rel_gap, elapsed, note)
        print(line.rstrip(), flush=True)


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
    if BO_obj.y_std is not None or BO_obj.y_mean is not None:
        print(" %-20s : %.6g" % ("Minimum (original)", BO_obj.to_original_y(UB)) + "   (in the units of the original outputs)")
    else:
        print(" %-20s : %s" % ("Minimum (original)", "n/a (add y_mean, y_std to params.json)"))
    print(" %-20s : %.6f" % ("Lower bound", LB) + "   (certified: mean >= this everywhere in the box)")
    print(" %-20s : %.6f" % ("Absolute gap", abs_gap) + "   (tolerance %g)" % BO_obj.abs_gap)
    print(" %-20s : %.2f %%" % ("Relative gap", 100*rel_gap) + "     (tolerance %g %%)" % (100*BO_obj.rel_gap))
    if BO_obj.x_lower is not None and BO_obj.x_upper is not None:
        print(" %-20s : [%s]" % ("x* (original)", ", ".join("%.6g" % v for v in BO_obj.to_original_x(x))))
    else:
        print(" %-20s : [%s]" % ("x* (normalized)", ", ".join("%.6f" % v for v in x))
              + "\n%22s  (add x_lower, x_upper to params.json to report x* in original units)" % "")
    print("-" * 68)
    print(" %-20s : d = %d, N = %d training points" % ("Problem", BO_obj.dim, len(BO_obj.x)))
    print(" %-20s : rho = %g, %d PWL segments, %d Gurobi thread(s)" % (
        "Settings", BO_obj.hyperparameter, BO_obj.PWL_segment, BO_obj.Thread_gurobi))
    n_initial = r["preprocess_boxes"] - r["unexplored_boxes"] # boxes bounded before branching
    if BO_obj.PreProcess:
        initial, initial_time = "%d preprocessing boxes" % n_initial, "preprocessing"
    else:
        initial, initial_time = "root box", "root box"
    print(" %-20s : %d (%s + %d from branching)" % ("Nodes bounded", r["nodes"], initial, r["nodes"] - n_initial))
    if not converged:
        print(" %-20s : %d" % ("Open nodes left", r["active_nodes"]))
    print(" %-20s : %.1f s (%s %.1f s, B&B %.1f s)" % (
        "Time", r["time_total"], initial_time, r["time_preprocess"], r["time_total"] - r["time_preprocess"]))
    print(line)
