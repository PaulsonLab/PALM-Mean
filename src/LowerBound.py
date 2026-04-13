#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 19:25:42 2025

@author: tang.1856
"""
from src.Analytical_LB import get_analytic
from src.UpperBound import get_UB
from src.PWL_Custom_LB import PWL_LB
import torch

# Find the lower bounding procedure
def get_LB(bounds, BO_obj=None, Thread=3, Best_UB = 1000):
    
    """
    This procedure returns the lower and upper bounds of the spatial node
       
    """
    # Calculate the LB for unimportant group (analytical lower bounding group)
    Sigma1, index2, dist_center_proj, dist_center_corner = get_analytic(bounds, BO_obj)

    part_neg = int(BO_obj.PWL_segment)
    part_pos = int(BO_obj.PWL_segment)

    # ------------------------------------------------------------Below calculate the LB for important group (Group 2)----------------------------------------------------------------------------
    xL_BB = bounds[0]
    xU_BB = bounds[1]   
     
    # Use local solver to solve the upper bound
    x_arg1, UB = get_UB(BO_obj.sigma_f,  BO_obj.alpha, BO_obj.X, xL_BB, xU_BB, BO_obj.n_multi)
    
    # print('number of IMPORTANT group=',len(index2))
    if len(index2)>=0:
    
        if UB - BO_obj.Best_UB>0.51:
            gap = (UB - BO_obj.Best_UB) - 0.5
        else:
            gap = 0.01
       

        Sigma2 = PWL_LB(xL_BB, xU_BB, dist_center_proj, dist_center_corner, BO_obj, index2, part_neg = part_neg, part_pos = part_pos, Thread=Thread, gap=gap)
        
    else:

        Sigma2 = torch.tensor(0.0)

   
    LB = Sigma1 + Sigma2        
    x_arg = x_arg1*BO_obj.l # return x in original space 
    
    return x_arg, LB, UB
