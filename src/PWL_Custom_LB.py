#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 10:06:46 2023

Lower bound of the important kernel terms on a box, from a nonconvex MIQCP solved by Gurobi.

Each term alpha_i * sigma_f^2 * exp(-d_i/2) depends on x only through the squared scaled distance
d_i = ||(x - x_i)/l||^2, which lies in [dL_i, dU_i] on the box. Each term is replaced by a
piecewise-linear (PWL) under-estimator in d_i (Gurobi setPWLObj, see
https://www.gurobi.com/documentation/9.5/examples/piecewise_py.html):

    alpha_i < 0: the term is concave in d -> chords through equally spaced points
    alpha_i > 0: the term is convex in d  -> on each segment, the maximum of the tangents at its two ends

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


def chord_breakpoints(dsqL, dsqU, coef, n_segment):
    """Breakpoints (d, value) of the chord interpolation of coef*exp(-d/2) on n_segment equal segments (concave case)."""
    dd = (dsqU - dsqL)/n_segment
    points = [dsqL]
    temp_val = dsqL + dd
    for _ in range(n_segment-1):
        points.append(temp_val)
        temp_val = temp_val + dd
    points.append(dsqU)
    dsq = torch.vstack(points)
    return dsq, coef*torch.exp(-dsq/(2))


def tangent_breakpoints(dsqL, dsqU, coef, n_segment):
    """
    Breakpoints (d, value) of the max-of-tangents under-estimator of coef*exp(-d/2) (convex case).
    On each segment [a, b] the envelope goes from (a, f(a)) along the tangent at a to the intersection
    with the tangent at b, then along the tangent at b to (b, f(b)).
    """
    step = (dsqU-dsqL)/n_segment
    dsq = [dsqL]
    val = [coef*torch.exp(-dsqL/(2))]
    for i in range(n_segment):
        right = dsqL+step*(i+1) if i < n_segment-1 else dsqU
        interval_x = torch.vstack((dsqL+step*i, right))
        interval_y = coef*torch.exp(-interval_x/(2))
        slope = interval_y*(-0.5) # derivative of coef*exp(-d/2)
        c = slope*(-interval_x) + interval_y # tangent intercepts

        dsq_intercept = 1.0*(c[1] - c[0])/(slope[0] - slope[1]) # intersection of the two tangents
        nan_indices, replacement_value = check_nan(interval_x, dsq_intercept)
        dsq_intercept[nan_indices] = replacement_value.double()

        dsq += [dsq_intercept, right]
        val += [slope[0]*dsq_intercept + c[0], interval_y[1]]
    return torch.vstack(dsq), torch.vstack(val)


def PWL_LB(zL, zU, dist_near, dist_far, BO_obj, index2, n_segment=6, Thread=1, gap=0.01):
    """
    zL, zU             : node box in z-space
    dist_near/dist_far : smallest/largest distance between every center and the box (z-space)
    index2             : indices of the important terms
    n_segment          : number of PWL segments per term
    gap                : absolute MIP gap for Gurobi (the bound returned is valid regardless)

    Returns Gurobi's best objective bound, a valid lower bound of the important terms on the box.
    """
    l_xspace = BO_obj.l[0]
    sigma = BO_obj.sigma_f
    alpha = BO_obj.alpha[index2]

    # The model is written in the original x-scale (x = z*l)
    xL = zL*l_xspace
    xU = zU*l_xspace
    X = BO_obj.X[index2]*l_xspace
    dim = X.size(1)

    dsqL = dist_near[index2]**2 # range of the squared distance d_i on the box
    dsqU = dist_far[index2]**2

    pos_alpha_index = (alpha > 0).squeeze(-1)
    neg_alpha_index = torch.logical_not(pos_alpha_index)

    m = gp.Model()
    m.setParam('OutputFlag', 0)
    m.Params.LogToConsole = 0

    x = m.addVars(dim, lb=xL, ub=xU, vtype=gp.GRB.CONTINUOUS, name="x")

    def add_terms(index, breakpoints, name):
        # one distance variable per term, linked to x by d_i = ||(x - x_i)/l||^2, with a PWL objective in d_i
        coef = alpha[index].squeeze(-1)*(sigma**2)
        centers = X[index]
        if len(coef) == 0:
            return
        dsq, val = breakpoints(dsqL[index], dsqU[index], coef, n_segment)
        d = m.addVars(len(coef), lb=dsqL[index], ub=dsqU[index], vtype=gp.GRB.CONTINUOUS, name=name)
        for i in range(len(coef)):
            m.setPWLObj(d[i], dsq[:,i], val[:,i])
            distance_sq = gp.quicksum((x[k]/l_xspace[k] - centers[i,k]/l_xspace[k])**2 for k in range(dim))
            m.addConstr(distance_sq >= d[i])
            m.addConstr(distance_sq <= d[i])

    add_terms(neg_alpha_index, chord_breakpoints, "dneg")
    add_terms(pos_alpha_index, tangent_breakpoints, "dpos")

    m.setParam('Threads', Thread)
    m.setParam('MIPGap', 0.001)
    m.setParam('MIPGapAbs', gap)
    m.setParam('TimeLimit', 10)
    m.optimize()

    return torch.tensor(m.ObjBound) # best bound: a valid lower bound even if Gurobi stops early
