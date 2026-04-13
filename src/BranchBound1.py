#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 13:38:50 2023

@author: tang.1856
"""

import numpy as np
import torch
import time
import math
import itertools
from src.LowerBound import get_LB

def partition_bounds(xL, xU, partitions):
    dimensions = len(xL)
    intervals = []

    for i in range(dimensions):
        intervals.append(torch.linspace(xL[i], xU[i], partitions[i] + 1))

    # Generate all combinations of intervals
    lower_bounds = []
    upper_bounds = []
    for index_tuple in itertools.product(*(range(p) for p in partitions)):
        lower_bound = torch.tensor([intervals[dim][index_tuple[dim]] for dim in range(dimensions)])
        upper_bound = torch.tensor([intervals[dim][index_tuple[dim] + 1] for dim in range(dimensions)])
        lower_bounds.append(lower_bound)
        upper_bounds.append(upper_bound)

    # Convert the lists of lower and upper bounds to tensors
    x_LB = torch.stack(lower_bounds)
    x_UB = torch.stack(upper_bounds)
    return x_LB, x_UB

def PrePross(RootNode, x_arg = None, time_limit=None, Part = 3, BO_obj=None):

    xL = RootNode.xL
    xU = RootNode.xU
    
    dim = xL.size(dim = 0) # Only single dimension case does not work here
    
    if BO_obj.PreProcess:
        length = xU-xL # length of each dimension
      
        partitions = [1000]*dim
        Part = BO_obj.n_prepart
        while math.prod(partitions)>500:
            p1 = (dim*length[0]**2/torch.sum(length**2))**0.5*Part
           
            partitions = []
            for D in range(dim):
                partitions.append(int(torch.maximum(torch.tensor(1),torch.round(length[D]/length[0]*p1))))
                
            Part-=0.5
            
        x_LB, x_UB = partition_bounds(xL, xU, partitions)
    else:
        x_LB = xL.unsqueeze(0)
        x_UB = xU.unsqueeze(0)

    # Create a new list for evaluation
    DecisionNodes = []
    UpperBoundList = []
    LowerBoundList =[]
    
    start_time_pre = time.time()
    for i in range(len(x_LB)):
        
        DecisionNode = Node(x_LB[i], x_UB[i], PreProcess = False, BO_obj=BO_obj)
        UpperBoundList.append(DecisionNode.UB)
        LowerBoundList.append(DecisionNode.LB)
        DecisionNodes.append(DecisionNode)    
        end_time_pre = time.time()
        if end_time_pre - start_time_pre>time_limit:
            break        
    
    return DecisionNodes,UpperBoundList,LowerBoundList, len(DecisionNodes)


# Find the lower bounding procedure
class Node():
    """
    Set-up Node class for B&B procedure
    """

    def __init__(self, xL, xU, RootNode=False, PreProcess = False, BO_obj = None):

        self.xL = xL # Lower bound - spatial
        self.xU = xU # Upper bound for BB
              
        self.hyperparam = BO_obj.hyperparameter
        self.bounds = torch.stack((xL,xU))
        self.PreProcess = PreProcess
        self.BO_obj = BO_obj
        if RootNode is False:
            self.x_arg, self.LB, self.UB = self.LB()
           
    def LB(self): 
        with torch.no_grad():         
            x_arg, LB, UB = get_LB(self.bounds, self.BO_obj, Thread=self.BO_obj.Thread_gurobi)        
        
        return x_arg, LB, UB


def SpatialBranching(DecisionNode,hyperparam,BO_obj):    
    
    """
    Use spatial branching over a given Decision Node
    
    """       
       
    # Because both of the above criteria was not matched, we branch the Decision Nodes 
    # into a list of 2 Child Nodes - Initialization
    x_min = DecisionNode.xL
    x_max = DecisionNode.xU

    # Child node 1: Left side spatial Branch
    x_min1 = x_min.clone().detach()
    x_max1 = x_max.clone().detach()                                      

    # Child node 2: Right side spatial branch
    x_min2 = x_min.clone().detach()
    x_max2 = x_max.clone().detach()  

   
    # Find the mid-point of the bounds across each dimension
    mid_point = (x_max + x_min)/2
    max_dim = torch.argmax((x_max - x_min)) # Dimension that has the largest diameter
    
    x_max1[int(max_dim)] = float(mid_point[int(max_dim)])
    x_min2[int(max_dim)] = float(mid_point[int(max_dim)])

    Child1 = Node(x_min1, x_max1, BO_obj=BO_obj)
    Child2 = Node(x_min2, x_max2, BO_obj=BO_obj)

    ChildNodes = [Child1,Child2]
     
    return ChildNodes


def BranchOrPrune(DecisionNode, BestUpperBound, hyperparam, BO_obj):
    """This function asks the following questions:
    1. Is this Decision Node's Lower bound lower than the best upper-bound among all decision nodes?
    2. Did the Decision Node Converge according to our convergeance criteria?

    If 1. or 2. -> Terminate and return criteria number 
    Else -> Return list of 2 child nodes as QuadraticFit Class and return criteria number
    
    Arguements:
    Decision -- Decision Node   (Node class)
    BestUpperBound -- Best Upperbound found so far

    Output -- 
    1. QuadraticFit class (If condition 1 or 2 or 4) or a list containg 2 objects QudraticFit class (condition 3)
    2. Integer representing condition 1, 2 , 3 or 4
    """
    UB = DecisionNode.UB
    LB = DecisionNode.LB

    abs_gap = torch.abs(UB - LB) # Absolution gap
    rel_gap = torch.abs(UB - LB)/torch.abs(UB) # Relative gap  
    abs_gap1 = torch.abs(BestUpperBound-LB)
    rel_gap1 = torch.abs(BestUpperBound-LB)/abs(BestUpperBound)
    
    if LB > BestUpperBound:
        ChildNodes =  DecisionNode
        condition = 1  # "1" represents the first criteria

    elif abs_gap<= BO_obj.abs_gap or rel_gap<= BO_obj.rel_gap:  # 0.05, 0.001
         ChildNodes =  DecisionNode
         condition = 2  # "2" represents the second criteria 
         
    elif abs_gap1<= BO_obj.abs_gap or rel_gap1<=BO_obj.rel_gap:
         ChildNodes =  DecisionNode
         condition = 4 
    else :   
        # Create a seperate function to keep everything clean
        ChildNodes = SpatialBranching(DecisionNode,hyperparam,BO_obj)
        condition = 3 # "3" represents the third criteria

    return ChildNodes, condition
        
def BranchBound(time_limit, BO_obj):
    
    """
    Used to carryout the branch and bound procedure
    Use another function to define the branching procedure

    Arg:
        alpha: 
        kern_fun
        X_samp
        xL
        xU

    """

    xL = BO_obj.xL
    xU = BO_obj.xU
    start_time_BB = time.time() # starting time for the BranchBrand function
    
    hyperparam = BO_obj.hyperparameter # hyperparam for grouping analytic bound
    
    # Root node
    RootNode = Node(xL, xU, RootNode=True, BO_obj=BO_obj)
 
    DecisionNodes, UpperBoundList, LowerBoundList, n_pre_nodes = PrePross(RootNode, time_limit=time_limit, BO_obj=BO_obj)
    
    iter_count, node_count = n_pre_nodes, n_pre_nodes
    
    DecisionNodes = np.array(DecisionNodes)
    LowerBoundList = np.array(LowerBoundList)
    Ncondition = 3
    
  
    while np.any(DecisionNodes): # If there is any child node remaining in the DecisionNodes list
        # print('Number of node:', node_count)
        end_time_BB = time.time()
        
        
        min_LCB_index = int(np.argmin(LowerBoundList)) # The node that has the minimum LB, which is the next treating node
        DecisionNodes_bracn_or_prune = [DecisionNodes[min_LCB_index]] # node selection (select the lowest lower bound node)
       

        if BO_obj.Best_UB>np.min(UpperBoundList):
            BO_obj.Best_UB = float(torch.tensor(np.min(UpperBoundList)))
            BO_obj.Best_UB_arg = DecisionNodes[np.argmin(UpperBoundList)].x_arg 
           
            
        if np.min(LowerBoundList)<=BO_obj.Best_UB and Ncondition==3:
            BestLowerBound = np.min(LowerBoundList)
        
        if end_time_BB-start_time_BB>=time_limit:
            print('B&B process exceeds specified time limit!')
            break
        
        
        # Create a list for conditions
        ChildNodes = []
        BBcondition = []
       
           
        for DecisionNode in DecisionNodes_bracn_or_prune: # check the condition of each decision node
            # Check what to do with the decision nodes
            BoP_output , BoP_condition = BranchOrPrune(DecisionNode, BO_obj.Best_UB, hyperparam, BO_obj)
            ChildNodes.append(BoP_output)
            BBcondition.append(BoP_condition)
           

        for n, Ncondition in enumerate(BBcondition):

            if Ncondition == 2: # node converge
                    
                # Terminates the decision node due to convergance
                if float(ChildNodes[n].UB) <= BO_obj.Best_UB:
                    BO_obj.Best_UB_arg = ChildNodes[n].x_arg
                    BO_obj.Best_UB = float(ChildNodes[n].UB)
                    
            elif Ncondition == 3: # branching and create two childnodes
                
                iter_count+=2 # create two childnodes
                node_count+=2 
                                   
                # Continues branch and Bounds
                DecisionNodes[min_LCB_index] = ChildNodes[n][0] # replace the current operating node with the new childnode
                DecisionNodes = np.append(DecisionNodes, ChildNodes[n][1])  # add another new childnode
                LowerBoundList[min_LCB_index] = ChildNodes[n][0].LB 
                LowerBoundList = np.append(LowerBoundList, ChildNodes[n][1].LB)
                UpperBoundList[min_LCB_index] = ChildNodes[n][0].UB 
                UpperBoundList = np.append(UpperBoundList, ChildNodes[n][1].UB)
            
            
            if Ncondition!=3: # prune the node
                DecisionNodes = np.delete(DecisionNodes,[min_LCB_index]) 
                LowerBoundList = np.delete(LowerBoundList ,[min_LCB_index]) 
                UpperBoundList = np.delete(UpperBoundList ,[min_LCB_index]) 
       
    return BO_obj.Best_UB, BestLowerBound, node_count, BO_obj.Best_UB_arg 



