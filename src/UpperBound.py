#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 21:26:33 2023

Upper bounding: local optimization of the true posterior mean. Any point found is
feasible, so its value is a valid upper bound on the minimum over the box.
"""
import torch
import numpy as np
from scipy.optimize import minimize


def mean(z, sigma_f, alpha, Z):
    """Posterior mean at z, with z and the training inputs Z in lengthscale-scaled coordinates."""
    distance = np.linalg.norm(Z - z, axis=1)
    return np.sum(alpha*sigma_f**2*np.exp(-0.5*distance**2))


def get_UB(sigma_f, alpha, Z, zL, zU, n_multi):
    """
    Multistart L-BFGS-B on the posterior mean over the box [zL, zU] (z-space).
    Returns the best point found (z-space) and its value.
    """
    sigma_f = sigma_f.numpy()
    alpha = alpha.flatten().numpy()
    Z = Z.numpy()
    bounds = [(float(zL[i]), float(zU[i])) for i in range(len(zL))]

    optimal_obj = float('inf')
    for _ in range(n_multi):
        z0 = (zL + (zU-zL)*torch.rand(len(zL))).numpy() # random start in the box
        result = minimize(mean, x0=z0, args=(sigma_f, alpha, Z), method='L-BFGS-B', bounds=bounds)
        if result.fun <= optimal_obj:
            optimal_soln = torch.tensor(result.x)
            optimal_obj = torch.tensor(result.fun)
    return optimal_soln, optimal_obj
