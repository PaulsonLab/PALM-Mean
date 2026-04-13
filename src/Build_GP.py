#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 21:38:08 2025

@author: tang.1856
"""

import torch
from src.BuildGP import GPBuild
from src.UpperBound import get_UB

def Build_GP(Preprocess, Thread_gurobi,PWL_segment, n_prepart, hyperparameter, abs_gap, rel_gap, n_multi, param_path, X_path, Y_path):
    
    GP_obj = GPBuild(param_path = param_path, X_path = X_path, Y_path = Y_path)
    
    l = GP_obj.l        
    X = GP_obj.x # haven't divide by length scale      
    dim = GP_obj.dim           

    candidate, pre_UB = get_UB(GP_obj.sigma_f,  GP_obj.alpha, GP_obj.x/GP_obj.l, torch.zeros(dim), torch.ones(dim)/GP_obj.l[0], 5)     
    
    GP_obj.Best_UB = float(pre_UB) # presolving solution
    GP_obj.Best_UB_arg = candidate*l[0] # presolving argument
    
    xL = torch.zeros(dim)
    xU = torch.ones(dim)
    
    X = X/l # convert x to z space
    xL=xL/l.squeeze().detach()
    xU=xU/l.squeeze().detach()
    
    GP_obj.xL = xL
    GP_obj.xU = xU
    GP_obj.X = X       
    GP_obj.PreProcess = Preprocess
    GP_obj.Thread_gurobi = Thread_gurobi
    GP_obj.PWL_segment = PWL_segment
    GP_obj.n_prepart = n_prepart
    GP_obj.hyperparameter = hyperparameter # hyperparameter for the heuristic rule
    GP_obj.abs_gap = abs_gap
    GP_obj.rel_gap = rel_gap
    GP_obj.n_multi = n_multi
    
    return GP_obj