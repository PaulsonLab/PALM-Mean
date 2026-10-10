#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 09:14:51 2023

Trained GP model. GPModel loads the training data and RBF hyperparameters of an
already trained GP, computes alpha = (K + noise*I)^-1 y and evaluates the posterior mean

    mu(x) = sum_i alpha_i * sigma_f^2 * exp(-0.5 * ||(x - x_i)/l||^2).

Hyperparameters are not fitted here (see benchmarks/common.py for fitting).
"""
import json
import numpy as np
import torch


def RBF_Matrix(Z, sigma_f, noise):
    """
    Covariance matrix K + noise*I of the RBF kernel.

    Z       : training inputs divided by the lengthscales -- tensor (N x dim)
    sigma_f : kernel standard deviation (sqrt of the signal variance)
    noise   : observation-noise variance -- tensor of shape (1,)
    """
    K = (sigma_f ** 2) * torch.exp(-0.5 * torch.cdist(Z, Z, p=2).pow(2))

    I = torch.eye(Z.size(dim=0))
    I.diagonal().fill_(noise.squeeze(0)) # noise variance on the diagonal
    return K + I


class GPModel():
    """
    Trained GP with an ARD squared-exponential (RBF) kernel and zero prior mean.

    Inputs:
        param_path : JSON file with '.kernel.variance', '.kernel.lengthscales', '.likelihood.variance', and
                     optionally the normalization of the data ('x_lower', 'x_upper', 'y_mean', 'y_std'):
                         x_original = x_lower + x*(x_upper - x_lower),   y_original = y_mean + y_std*y
        X_path     : training inputs, .npy array (N x dim), normalized to [0,1]^dim
        Y_path     : training outputs, .npy array (N x 1), centered (or standardized)

    Attributes:
        x, y    : normalized training data (x not divided by the lengthscales)
        dim     : input dimension
        l       : lengthscales -- tensor (1 x dim)
        sigma_f : kernel standard deviation
        jitter  : observation-noise variance
        alpha   : (K + noise*I)^-1 y -- tensor (N x 1)
        x_lower, x_upper, y_mean, y_std : normalization (None if not given in param_path)
    """

    def __init__(self, param_path, X_path, Y_path):

        with open(param_path, "r") as f:
            params = json.load(f)

        self.x = torch.tensor(np.load(X_path)).to(torch.float64)
        self.y = torch.tensor(np.load(Y_path))
        self.dim = self.x.shape[1]

        self.l = torch.tensor([params['.kernel.lengthscales']]).to(torch.float64)
        self.sigma_f = torch.tensor(params['.kernel.variance']).to(torch.float64)**0.5
        self.jitter = torch.tensor([params['.likelihood.variance']]).to(torch.float64)

        self.x_lower, self.x_upper = params.get('x_lower'), params.get('x_upper')
        self.y_mean, self.y_std = params.get('y_mean'), params.get('y_std')

        self.K =RBF_Matrix(self.x/self.l, self.sigma_f, self.jitter)
        self.K_inv = torch.inverse(self.K)
        self.alpha = torch.matmul(self.K_inv, self.y)

    def to_original_x(self, x):
        """Map a normalized point x in [0,1]^dim to the original input scale (unchanged if no x_lower/x_upper)."""
        x = np.asarray(x, dtype=float)
        if self.x_lower is None or self.x_upper is None:
            return x
        return np.asarray(self.x_lower) + x*(np.asarray(self.x_upper) - np.asarray(self.x_lower))

    def to_original_y(self, y):
        """Map a normalized output value to the original output scale (unchanged if no y_mean/y_std)."""
        return (self.y_mean or 0.0) + (1.0 if self.y_std is None else self.y_std)*y

    def posterior_mean(self, x):
        """Posterior mean at a single point x (numpy array, normalized input space)."""
        distance = np.linalg.norm(self.x/self.l - x/self.l, axis=1)
        return np.sum(np.array(self.alpha.flatten())*self.sigma_f.numpy()**2*np.exp(-0.5*distance**2))
