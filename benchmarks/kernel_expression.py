#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kernel value as an algebraic expression for the modeling interfaces of SCIP, MAiNGO and Pyomo/BARON.
No torch import: run_scip.py solves in a child process that must not load torch (see run_scip.py).
"""


def kernel_expression(kernel_name, dsq, sqrt, exp):
    """
    k(r) of the given kernel as an expression of the squared scaled distance dsq = r^2, built with the
    interface's own sqrt and exp functions (same kernels as src/Kernels.py, without sigma_f^2).
    """
    if kernel_name == "rbf":
        return exp(-0.5*dsq)
    r = sqrt(dsq)
    if kernel_name == "matern12":
        return exp(-r)
    if kernel_name == "matern32":
        return (1 + 3**0.5*r)*exp(-3**0.5*r)
    if kernel_name == "matern52":
        return (1 + 5**0.5*r + 5*dsq/3)*exp(-5**0.5*r)
    raise ValueError("Unknown kernel " + kernel_name)
