#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 21:26:33 2023

@author: tang.1856
"""

import torch
import numpy as np
from scipy.optimize import minimize, fmin_l_bfgs_b

def mean(x, sigma_f, alpha, x_center):

    distance = np.linalg.norm(x_center - x, axis=1)
    mean = np.sum(alpha*sigma_f**2*np.exp(-0.5*distance**2))
    
    return mean

def get_UB(sigma_f, alpha, x_center, xL, xU, n_multi):
    sigma_f = sigma_f.numpy()
    alpha = alpha.flatten().numpy()
    x_center = x_center.numpy()
    bounds = []
    
    for i in range(len(xL)):
        bounds.append((float(xL[i]),float(xU[i])))
    
    optimal_obj = 1e5
    for multi in range(n_multi):
        x0 = xL + (xU-xL)*torch.rand(len(xL))
        x0 = x0.numpy()
        
        result = minimize(mean,x0 = x0,args=(sigma_f,alpha,x_center),method='L-BFGS-B',bounds = bounds)
        if result.fun<=optimal_obj:
            optimal_soln = torch.tensor(result.x)
            optimal_obj = torch.tensor(result.fun)
    return optimal_soln, optimal_obj
