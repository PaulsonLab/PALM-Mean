#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 21:38:08 2025

Builds the object the B&B solver works on: the trained GP (GPModel) plus a presolve
incumbent, the search box in lengthscale-scaled coordinates and the solver settings.

@author: tang.1856
"""
import torch
from src.GPModel import GPModel
from src.UpperBound import get_UB


def setup_problem(Preprocess, Thread_gurobi, PWL_segment, n_prepart, hyperparameter, abs_gap, rel_gap, n_multi,
                  param_path, X_path, Y_path, log_every=100):
    """
    Load the GP and attach everything the B&B needs. The solver works in lengthscale-scaled
    coordinates z = x/l, in which every kernel term depends on a plain Euclidean distance.

    Added attributes:
        X           : training inputs in z-space
        xL, xU      : the search box [0,1]^dim in z-space
        Best_UB     : incumbent value from a 5-start presolve on the whole box
        Best_UB_arg : incumbent point (original x-space)
        PreProcess, Thread_gurobi, PWL_segment, n_prepart, hyperparameter (importance threshold rho),
        abs_gap, rel_gap, n_multi : solver settings (see main.py)
        log_every   : print a progress line every log_every generated nodes (0: no progress output)
    """
    GP_obj = GPModel(param_path=param_path, X_path=X_path, Y_path=Y_path)
    l = GP_obj.l
    dim = GP_obj.dim

    # Presolve: local optimization of the posterior mean over the whole box gives the first incumbent
    candidate, pre_UB = get_UB(GP_obj.sigma_f, GP_obj.alpha, GP_obj.x/l, torch.zeros(dim), torch.ones(dim)/l[0], 5, GP_obj.kernel)
    GP_obj.Best_UB = float(pre_UB)
    GP_obj.Best_UB_arg = candidate*l[0]

    GP_obj.X = GP_obj.x/l
    GP_obj.xL = torch.zeros(dim)/l.squeeze()
    GP_obj.xU = torch.ones(dim)/l.squeeze()

    GP_obj.PreProcess = Preprocess
    GP_obj.Thread_gurobi = Thread_gurobi
    GP_obj.PWL_segment = PWL_segment
    GP_obj.n_prepart = n_prepart
    GP_obj.hyperparameter = hyperparameter
    GP_obj.abs_gap = abs_gap
    GP_obj.rel_gap = rel_gap
    GP_obj.n_multi = n_multi
    GP_obj.log_every = log_every

    return GP_obj
