#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Human-readable output of a PALM-Mean run: a banner, a progress table printed during the search,
and the final summary built from the statistics BranchBound stores in BO_obj.result.
Plain ASCII only, so the output renders in any terminal encoding.
"""
import torch

WIDTH = 99 # width of the progress table and of the summary rules

LOGO = r"""
    ____  ___    __    __  ___      __  ___
   / __ \/   |  / /   /  |/  /     /  |/  /__  ____ _____
  / /_/ / /| | / /   / /|_/ /_____/ /|_/ / _ \/ __ `/ __ \
 / ____/ ___ |/ /___/ /  / /_____/ /  / /  __/ /_/ / / / /
/_/   /_/  |_/_____/_/  /_/     /_/  /_/\___/\__,_/_/ /_/
"""


def print_banner(BO_obj, time_limit):
    """Logo and a one-line description of the run, printed when the search starts."""
    print(LOGO.rstrip("\n"))
    print(" Global optimization of Gaussian process posterior means")
    print(" Developed by Wei-Ting (Jonathan) Tang, Akshay Kudva, Calvin Tsay, and Joel Paulson")
    print(" d = %d, N = %d training points | time limit %g s | %d Gurobi thread(s)" % (
        BO_obj.dim, len(BO_obj.x), time_limit, BO_obj.Thread_gurobi), flush=True)


class ProgressLog():
    """
    Progress table printed during the search. The header is printed with the first row, so it is not
    separated from the table by Gurobi's license banner, which appears when the first node MIQCP is
    built; close() draws the bottom border.
    """
    # (title, width) of each column
    COLUMNS = [("Nodes", 7), ("Open", 6), ("Best UB", 11), ("Lower bound", 11), ("Abs gap", 10),
               ("Rel gap", 8), ("Time (s)", 8), ("Stage", 13)]

    def __init__(self):
        self.header_printed = False
        self.rule = "+" + "+".join("-"*(w + 2) for _, w in self.COLUMNS) + "+"
        assert len(self.rule) == WIDTH

    def _row(self, cells):
        return "| " + " | ".join(("%-*s" if title == "Stage" else "%*s") % (w, c)
                                 for (title, w), c in zip(self.COLUMNS, cells)) + " |"

    def line(self, nodes, open_nodes, UB, LB, elapsed, n_boxes=None):
        """
        One table row. During preprocessing open_nodes and LB are None (the lower bound of the boxes
        not yet bounded is unknown) and n_boxes is the total number of preprocessing boxes.
        """
        if not self.header_printed:
            print("\n" + self.rule + "\n" + self._row([t for t, _ in self.COLUMNS]) + "\n" + self.rule, flush=True)
            self.header_printed = True
        if LB is None:
            cells = ["%d/%d" % (nodes, n_boxes), "-", "%.6f" % UB, "-", "-", "-", "%.1f" % elapsed, "preprocessing"]
        else:
            gap = UB - LB
            rel_gap = "%.2f%%" % (100*gap/abs(UB)) if UB != 0 else "-"
            cells = [str(nodes), str(open_nodes), "%.6f" % UB, "%.6f" % LB, "%.6f" % gap, rel_gap, "%.1f" % elapsed, "branching"]
        print(self._row(cells), flush=True)

    def close(self):
        """Bottom border of the table (only if a row was printed)."""
        if self.header_printed:
            print(self.rule, flush=True)


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

    line = "=" * WIDTH
    print()
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
    print("-" * WIDTH)
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
