#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 19:25:42 2025

Bounding of one B&B node: node lower bound = analytical bound of the unimportant terms
+ PWL MIQCP bound of the important terms; node upper bound from local optimization.

@author: tang.1856
"""
import torch
from src.Analytical_LB import get_analytic
from src.UpperBound import get_UB
from src.PWL_Custom_LB import PWL_LB


def get_LB(bounds, BO_obj, Thread=1):
    """
    bounds : node box in z-space -- tensor (2 x dim)
    Thread : Gurobi threads for the MIQCP

    Returns (x_arg, LB, UB): best local point (original x-space), node lower bound, node upper bound.
    """
    zL, zU = bounds[0], bounds[1]

    Sigma1, index2, dist_near, dist_far = get_analytic(bounds, BO_obj)

    z_arg, UB = get_UB(BO_obj.sigma_f, BO_obj.alpha, BO_obj.X, zL, zU, BO_obj.n_multi)

    if len(index2) > 0:
        # Adaptive MIQCP tolerance: a node whose local UB is far above the incumbent does not need a tight bound
        if UB - BO_obj.Best_UB > 0.51:
            gap = (UB - BO_obj.Best_UB) - 0.5
        else:
            gap = 0.01
        Sigma2 = PWL_LB(zL, zU, dist_near, dist_far, BO_obj, index2, n_segment=int(BO_obj.PWL_segment), Thread=Thread, gap=gap)
    else:
        Sigma2 = torch.tensor(0.0)

    return z_arg*BO_obj.l, Sigma1 + Sigma2, UB
