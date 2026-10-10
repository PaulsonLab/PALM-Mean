#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 10:06:46 2023

Lower bound of the important kernel terms on a box, from a nonconvex MIQCP solved by Gurobi.

Each term f_i = alpha_i * sigma_f^2 * k(r_i) depends on x only through the scaled distance
r_i = ||(x - x_i)/l||, whose range [rL_i, rU_i] on the box is known. Each term is replaced by a
piecewise-linear (PWL) under-estimator (Gurobi setPWLObj, see
https://www.gurobi.com/documentation/9.5/examples/piecewise_py.html), built segment by segment:

    where f_i is convex  -> on each segment, the maximum of the tangents at its two ends
    where f_i is concave -> chords through equally spaced points

RBF kernel: the term is written in d_i = r_i^2, in which exp(-d/2) is convex, so f_i is convex for
alpha_i > 0 and concave for alpha_i < 0.

Matern kernels: the term is written in r_i. Matern 3/2 and 5/2 are concave in r for r < r_infl and convex
for r > r_infl (Matern 1/2: convex everywhere). The interval [rL_i, rU_i] is split at r_infl when it
contains it; on each side the curvature of f_i = alpha_i * k is that of k for alpha_i > 0 and the
opposite for alpha_i < 0. Both pieces of the envelope pass through (r_infl, f_i(r_infl)), so the PWL
function is continuous and below f_i on the whole interval.

@author: kudva.7
"""
import gurobipy as gp
import torch


def check_nan(interval_x, dsq_intercept):
    """
    Indices where the tangent intersection is not strictly inside its segment (nan/inf when the two
    slopes are numerically equal), and the segment midpoints that replace them.
    """
    dsq_intercept_median = 0.5*(interval_x[0,:]+interval_x[1,:])
    nan_indices = torch.isnan(dsq_intercept) | torch.isinf(dsq_intercept) | (dsq_intercept <= interval_x[0,:]) | (dsq_intercept >= interval_x[1,:])
    nan_indices = torch.where(nan_indices)[0]
    return nan_indices, dsq_intercept_median[nan_indices]


def chord_breakpoints(lo, hi, f, n_segment):
    """
    Breakpoints (x, f(x)) of the chord interpolation of f on n_segment equal segments of [lo, hi]
    (valid under-estimator where f is concave). lo, hi: one entry per term; f maps a tensor of points to
    the term values, column by column.
    """
    dd = (hi - lo)/n_segment
    points = [lo]
    temp_val = lo + dd
    for _ in range(n_segment-1):
        points.append(temp_val)
        temp_val = temp_val + dd
    points.append(hi)
    x = torch.vstack(points)
    return x, f(x)


def tangent_breakpoints(lo, hi, f, df, n_segment):
    """
    Breakpoints of the max-of-tangents under-estimator of f on n_segment equal segments of [lo, hi]
    (valid where f is convex). On each segment [a, b] the envelope goes from (a, f(a)) along the tangent at a
    to the intersection with the tangent at b, then along the tangent at b to (b, f(b)).
    """
    step = (hi-lo)/n_segment
    x = [lo]
    val = [f(lo)]
    for i in range(n_segment):
        right = lo+step*(i+1) if i < n_segment-1 else hi
        interval_x = torch.vstack((lo+step*i, right))
        interval_y = f(interval_x)
        slope = df(interval_x)
        c = slope*(-interval_x) + interval_y # tangent intercepts

        x_intercept = 1.0*(c[1] - c[0])/(slope[0] - slope[1]) # intersection of the two tangents
        y_intercept = slope[0]*x_intercept + c[0]
        # Rounding can put the intersection outside the segment (e.g. nearly equal slopes). Use the midpoint
        # instead, with the lower of the two tangents there: the envelope then stays below both tangents.
        nan_indices, replacement_value = check_nan(interval_x, x_intercept)
        if len(nan_indices) > 0:
            mid = replacement_value.double()
            x_intercept[nan_indices] = mid
            y_intercept[nan_indices] = torch.minimum(slope[0, nan_indices]*mid + c[0, nan_indices],
                                                     slope[1, nan_indices]*mid + c[1, nan_indices])

        x += [x_intercept, right]
        val += [y_intercept, interval_y[1]]
    return torch.vstack(x), torch.vstack(val)


def envelope(lo, hi, f, df, convex, n_segment):
    """PWL under-estimator of f on [lo, hi]: tangents where f is convex, chords where it is concave."""
    return tangent_breakpoints(lo, hi, f, df, n_segment) if convex else chord_breakpoints(lo, hi, f, n_segment)


def matern_breakpoints(rL, rU, coef, kernel, n_segment):
    """
    PWL under-estimator of coef_i * k(r) on [rL_i, rU_i] for a Matern kernel, split at the inflection point.
    Returns one (x, y) pair of breakpoint tensors per term. Each convex or concave piece gets n_segment
    segments, so an interval containing the inflection point gets 2*n_segment segments.
    """
    r_infl = kernel.inflection
    breakpoints = [None]*len(coef)
    left = rU <= r_infl                  # k concave on the whole interval
    right = rL >= r_infl                 # k convex on the whole interval
    straddle = ~(left | right)           # rL < r_infl < rU: split at the inflection point

    for positive in (True, False):
        sign = (coef >= 0) if positive else (coef < 0)
        # f = coef * k: same curvature as k for coef >= 0, opposite for coef < 0
        convex_left, convex_right = (False, True) if positive else (True, False)
        for region in (left, right, straddle):
            idx = torch.where(sign & region)[0]
            if len(idx) == 0:
                continue
            c, lo, hi = coef[idx], rL[idx], rU[idx]
            f = lambda r, c=c: c*kernel.k(r)
            df = lambda r, c=c: c*kernel.dk(r)
            if region is left:
                x, y = envelope(lo, hi, f, df, convex_left, n_segment)
            elif region is right:
                x, y = envelope(lo, hi, f, df, convex_right, n_segment)
            else:
                infl = torch.full_like(lo, r_infl)
                x1, y1 = envelope(lo, infl, f, df, convex_left, n_segment)
                x2, y2 = envelope(infl, hi, f, df, convex_right, n_segment)
                x, y = torch.vstack((x1, x2[1:])), torch.vstack((y1, y2[1:])) # both pieces start/end at (r_infl, f(r_infl))
            for j, i in enumerate(idx.tolist()):
                breakpoints[i] = (x[:, j], y[:, j])
    return breakpoints


def PWL_LB(zL, zU, dist_near, dist_far, BO_obj, index2, n_segment=6, Thread=1, gap=0.01):
    """
    zL, zU             : node box in z-space
    dist_near/dist_far : smallest/largest distance between every center and the box (z-space)
    index2             : indices of the important terms
    n_segment          : number of PWL segments per term (per convex/concave piece for Matern kernels)
    gap                : absolute MIP gap for Gurobi (the bound returned is valid regardless)

    Returns Gurobi's best objective bound, a valid lower bound of the important terms on the box.
    """
    l_xspace = BO_obj.l[0]
    sigma = BO_obj.sigma_f
    alpha = BO_obj.alpha[index2]
    kernel = BO_obj.kernel

    # The model is written in the original x-scale (x = z*l)
    xL = zL*l_xspace
    xU = zU*l_xspace
    X = BO_obj.X[index2]*l_xspace
    dim = X.size(1)

    # Terms whose distance range is (almost) a single point cannot get a PWL objective (Gurobi needs breakpoints
    # at least 1e-6 apart). Every kernel is monotone in the distance, so min(f(rL), f(rU)) bounds such a term exactly.
    rL_all, rU_all = dist_near[index2], dist_far[index2]
    coef_all = alpha.reshape(-1)*(sigma**2)
    tiny = (rU_all - rL_all) < 1e-5
    if kernel.name == "rbf":
        tiny = tiny | ((rU_all**2 - rL_all**2) < 1e-5)
    constant = torch.sum(torch.minimum(coef_all[tiny]*kernel.k(rL_all[tiny]), coef_all[tiny]*kernel.k(rU_all[tiny])))
    if tiny.any():
        keep = ~tiny
        index2, alpha, X = index2[keep], alpha[keep], X[keep]

    m = gp.Model()
    m.setParam('OutputFlag', 0)
    m.Params.LogToConsole = 0

    x = m.addVars(dim, lb=xL, ub=xU, vtype=gp.GRB.CONTINUOUS, name="x")

    def squared_distance(center):
        return gp.quicksum((x[k]/l_xspace[k] - center[k]/l_xspace[k])**2 for k in range(dim))

    if kernel.name == "rbf":
        dsqL = dist_near[index2]**2 # range of the squared distance d_i on the box
        dsqU = dist_far[index2]**2
        pos_alpha_index = (alpha > 0).squeeze(-1)
        neg_alpha_index = torch.logical_not(pos_alpha_index)

        def add_terms(index, convex, name):
            # one variable d_i per term, linked to x by d_i = ||(x - x_i)/l||^2, with a PWL objective in d_i
            coef = alpha[index].squeeze(-1)*(sigma**2)
            centers = X[index]
            if len(coef) == 0:
                return
            f = lambda d: coef*torch.exp(-d/(2))
            df = lambda d: f(d)*(-0.5)
            dsq, val = envelope(dsqL[index], dsqU[index], f, df, convex, n_segment)
            d = m.addVars(len(coef), lb=dsqL[index], ub=dsqU[index], vtype=gp.GRB.CONTINUOUS, name=name)
            for i in range(len(coef)):
                m.setPWLObj(d[i], dsq[:,i], val[:,i])
                distance_sq = squared_distance(centers[i])
                m.addConstr(distance_sq >= d[i])
                m.addConstr(distance_sq <= d[i])

        add_terms(neg_alpha_index, False, "dneg") # alpha < 0: concave in d -> chords
        add_terms(pos_alpha_index, True, "dpos")  # alpha > 0: convex in d  -> tangents

    else:
        rL, rU = dist_near[index2], dist_far[index2]
        coef = alpha.reshape(-1)*(sigma**2)
        breakpoints = matern_breakpoints(rL, rU, coef, kernel, n_segment)
        # one variable r_i per term, linked to x by r_i^2 = ||(x - x_i)/l||^2 (r_i >= rL_i >= 0), with a PWL objective in r_i
        r = m.addVars(len(coef), lb=rL, ub=rU, vtype=gp.GRB.CONTINUOUS, name="r")
        for i, (r_points, values) in enumerate(breakpoints):
            m.setPWLObj(r[i], r_points, values)
            distance_sq = squared_distance(X[i])
            m.addConstr(distance_sq >= r[i]*r[i])
            m.addConstr(distance_sq <= r[i]*r[i])

    m.setParam('Threads', Thread)
    m.setParam('MIPGap', 0.001)
    m.setParam('MIPGapAbs', gap)
    m.setParam('TimeLimit', 10)
    m.optimize()

    # best bound: a valid lower bound even if Gurobi stops early
    return torch.tensor(m.ObjBound) + constant if tiny.any() else torch.tensor(m.ObjBound)
