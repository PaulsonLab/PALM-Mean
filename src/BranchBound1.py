#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 13:38:50 2023

Spatial branch-and-bound driver: optional pre-partitioning of the root box, best-first node
selection, bisection of the longest edge, and pruning. All boxes are in z-space (z = x/l).

@author: tang.1856
"""
import math
import time
import itertools
import numpy as np
import torch
from src.LowerBound import get_LB
from src.Analytical_LB import get_analytic_all

# Outcomes of BranchOrPrune
PRUNE_BY_BOUND = 1     # node LB above the incumbent
NODE_CONVERGED = 2     # node's own UB - LB within tolerance
BRANCH = 3             # bisect the node
WITHIN_TOLERANCE = 4   # incumbent - node LB within tolerance


class Node():
    """A box [xL, xU] of the B&B tree, bounded on creation."""

    def __init__(self, xL, xU, BO_obj):
        self.xL = xL
        self.xU = xU
        self.bounds = torch.stack((xL, xU))
        with torch.no_grad():
            self.x_arg, self.LB, self.UB = get_LB(self.bounds, BO_obj, Thread=BO_obj.Thread_gurobi)


def partition_bounds(xL, xU, partitions):
    """Split the box [xL, xU] into a grid with partitions[i] equal intervals along dimension i."""
    intervals = [torch.linspace(xL[i], xU[i], partitions[i] + 1) for i in range(len(xL))]

    lower_bounds, upper_bounds = [], []
    for index_tuple in itertools.product(*(range(p) for p in partitions)):
        lower_bounds.append(torch.tensor([intervals[d][index_tuple[d]] for d in range(len(xL))]))
        upper_bounds.append(torch.tensor([intervals[d][index_tuple[d] + 1] for d in range(len(xL))]))
    return torch.stack(lower_bounds), torch.stack(upper_bounds)


def PrePross(BO_obj, time_limit):
    """
    Pre-partition the root box into at most 500 sub-boxes with comparable diagonals (if BO_obj.PreProcess)
    and bound each of them.

    Returns the bounded nodes, their upper and lower bounds, and an analytical lower bound for the
    boxes that could not be bounded before the time limit (inf if all were bounded).
    """
    xL, xU = BO_obj.xL, BO_obj.xU
    dim = len(xL)

    if BO_obj.PreProcess:
        # Partitions per dimension proportional to the side length; reduce until there are at most 500 boxes
        length = xU-xL
        partitions = [1000]*dim
        Part = BO_obj.n_prepart
        while math.prod(partitions) > 500:
            p1 = (dim*length[0]**2/torch.sum(length**2))**0.5*Part
            partitions = [int(torch.maximum(torch.tensor(1), torch.round(length[D]/length[0]*p1))) for D in range(dim)]
            Part -= 0.5
        x_LB, x_UB = partition_bounds(xL, xU, partitions)
    else:
        x_LB, x_UB = xL.unsqueeze(0), xU.unsqueeze(0)

    DecisionNodes, UpperBoundList, LowerBoundList = [], [], []
    start_time_pre = time.time()
    for i in range(len(x_LB)):
        node = Node(x_LB[i], x_UB[i], BO_obj)
        DecisionNodes.append(node)
        UpperBoundList.append(node.UB)
        LowerBoundList.append(node.LB)
        if time.time() - start_time_pre > time_limit:
            break

    # Boxes skipped because of the time limit still belong to the domain: bound them analytically
    unexplored_LB = float('inf')
    for i in range(len(DecisionNodes), len(x_LB)):
        unexplored_LB = min(unexplored_LB, float(get_analytic_all(torch.stack((x_LB[i], x_UB[i])), BO_obj)))
    if len(DecisionNodes) < len(x_LB):
        print('Pre-process exceeds specified time limit! %d of %d boxes bounded analytically only' % (len(x_LB)-len(DecisionNodes), len(x_LB)))

    return DecisionNodes, UpperBoundList, LowerBoundList, unexplored_LB


def SpatialBranching(node, BO_obj):
    """Bisect the node along its longest edge and return the two bounded children."""
    max_dim = int(torch.argmax(node.xU - node.xL))
    mid_point = float((node.xU[max_dim] + node.xL[max_dim])/2)

    x_max1 = node.xU.clone().detach() # left child: [xL, x_max1]
    x_min2 = node.xL.clone().detach() # right child: [x_min2, xU]
    x_max1[max_dim] = mid_point
    x_min2[max_dim] = mid_point

    return [Node(node.xL.clone().detach(), x_max1, BO_obj), Node(x_min2, node.xU.clone().detach(), BO_obj)]


def BranchOrPrune(node, BestUpperBound, BO_obj):
    """
    Decide what to do with the selected node. Returns (condition, children):
        PRUNE_BY_BOUND   -- node LB > incumbent: the node cannot contain a better point
        NODE_CONVERGED   -- node UB - node LB within abs_gap or rel_gap
        WITHIN_TOLERANCE -- incumbent - node LB within abs_gap or rel_gap
        BRANCH           -- otherwise: children holds the two child nodes
    """
    UB, LB = node.UB, node.LB

    abs_gap = torch.abs(UB - LB)
    rel_gap = torch.abs(UB - LB)/torch.abs(UB)
    abs_gap_incumbent = torch.abs(BestUpperBound - LB)
    rel_gap_incumbent = torch.abs(BestUpperBound - LB)/abs(BestUpperBound)

    if LB > BestUpperBound:
        return PRUNE_BY_BOUND, None
    if abs_gap <= BO_obj.abs_gap or rel_gap <= BO_obj.rel_gap:
        return NODE_CONVERGED, None
    if abs_gap_incumbent <= BO_obj.abs_gap or rel_gap_incumbent <= BO_obj.rel_gap:
        return WITHIN_TOLERANCE, None
    return BRANCH, SpatialBranching(node, BO_obj)


def BranchBound(time_limit, BO_obj):
    """
    Run the B&B on the box [BO_obj.xL, BO_obj.xU] (see setup_problem).

    Returns (best upper bound, best lower bound, number of nodes, best point in x-space).
    """
    start_time_BB = time.time()

    DecisionNodes, UpperBoundList, LowerBoundList, unexplored_LB = PrePross(BO_obj, time_limit)
    node_count = len(DecisionNodes)

    DecisionNodes = np.array(DecisionNodes)
    LowerBoundList = np.array(LowerBoundList)
    BestLowerBound = min(float(np.min(LowerBoundList)), unexplored_LB)
    condition = BRANCH

    while len(DecisionNodes) > 0:

        k = int(np.argmin(LowerBoundList)) # best-first: process the node with the smallest LB

        # Update the incumbent from the node upper bounds
        if BO_obj.Best_UB > np.min(UpperBoundList):
            BO_obj.Best_UB = float(torch.tensor(np.min(UpperBoundList)))
            BO_obj.Best_UB_arg = DecisionNodes[np.argmin(UpperBoundList)].x_arg

        # Global lower bound: smallest LB among the active nodes (kept from before a pruning step, so that the
        # reported bound also covers nodes fathomed within tolerance)
        if np.min(LowerBoundList) <= BO_obj.Best_UB and condition == BRANCH:
            BestLowerBound = min(float(np.min(LowerBoundList)), unexplored_LB)

        if time.time() - start_time_BB >= time_limit:
            print('B&B process exceeds specified time limit!')
            break

        condition, children = BranchOrPrune(DecisionNodes[k], BO_obj.Best_UB, BO_obj)

        if condition == BRANCH:
            # replace the node by its first child and append the second
            node_count += 2
            DecisionNodes[k] = children[0]
            DecisionNodes = np.append(DecisionNodes, children[1])
            LowerBoundList[k] = children[0].LB
            LowerBoundList = np.append(LowerBoundList, children[1].LB)
            UpperBoundList[k] = children[0].UB
            UpperBoundList = np.append(UpperBoundList, children[1].UB)
        else:
            if condition == NODE_CONVERGED and float(DecisionNodes[k].UB) <= BO_obj.Best_UB:
                BO_obj.Best_UB_arg = DecisionNodes[k].x_arg
                BO_obj.Best_UB = float(DecisionNodes[k].UB)
            DecisionNodes = np.delete(DecisionNodes, [k])
            LowerBoundList = np.delete(LowerBoundList, [k])
            UpperBoundList = np.delete(UpperBoundList, [k])

    return BO_obj.Best_UB, BestLowerBound, node_count, BO_obj.Best_UB_arg
