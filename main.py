#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from src.BranchBound1 import BranchBound
from src.Build_GP import Build_GP
        
if __name__=='__main__':
    
    Preprocess = True # whether doing the pre-process step
    time_limit = 600 # time limit for executing the algorithm (sec)
    Thread_gurobi = 1 # number of core to solve MIQCP
    abs_gap = 0.1 # convergence criteria for the B&B algorithm
    rel_gap = 0.01 # convergence criteria for the B&B algorithm
    PWL_segment = 6 # number of piecewise linear segment
    n_prepart = 10 # number of partition for the diagonal
    hyperparameter = 0.01 # importance threshold
    n_multi = 2 # number of multistart for optimizing the upper bound
   
        
    # Set the number of CPUs
    torch.set_num_threads(Thread_gurobi)
    
    # Specify the path
    param_path = "params.json" # GP hyperparameters
    X_path = 'X.npy' # training inputs
    Y_path = 'Y.npy' # training outputs
    
    args_dict = {"Preprocess":Preprocess, "Thread_gurobi":Thread_gurobi, "PWL_segment":PWL_segment, "n_prepart":n_prepart,
                 "hyperparameter":hyperparameter, "abs_gap":abs_gap, "rel_gap":rel_gap, "n_multi":n_multi, "param_path":param_path, "X_path":X_path, "Y_path":Y_path}

    GP_obj = Build_GP(**args_dict)


    BestUpperBound, BestLowerBound, node_count, BestUpperBound_arg = BranchBound(time_limit, GP_obj)                         
    Solution = GP_obj.posterior_mean(BestUpperBound_arg.numpy())
    
    print('x* = ', BestUpperBound_arg)
    print('Minimum posterior mean = ', BestUpperBound)
    print('Absolute Gap = ', BestUpperBound - BestLowerBound)
   