#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 09:14:51 2023

This is used to fit a gp function on BO torch 
the class GPBuild is then used to to be a superclass to 
GPDNN

@author: kudva.7
"""
import torch
import numpy as np 
import json
    
 
def RBF_Matrix(X, lengthscale, fscale, noise):
    
    """
    Used to generate the RBF Kernel covariance matrix in a fast manner
    Parameters
    ----------
    X : Features of GP-- torch tensor n_samples X ndim
    
    Hyper parameters: 
        
    -> lengthscale
    -> fscale  
    -> noise  

    Returns
    -------
    K : Covariance Matrix torch matrix (n_samples X n_samples)

    """
       
    n_samples = X.size(dim = 0)
    pairwise_distances_sq = torch.cdist(X, X, p=2).pow(2)
    K = (fscale ** 2) * torch.exp(-0.5 * pairwise_distances_sq / lengthscale ** 2)
    
    # Set the diagonal elements to represent the variance (noise)
    I = torch.eye(n_samples)
    I.diagonal().fill_(noise.squeeze(0))
    
    # Final kernel with jitter
    K += I
   
    return K
   

class GPBuild():
    
    """
    This class builds GP and stores all data related to the true kernel
    
    Present SetUp: Will just be using the scaled GP for branch and bound procedures for now
    
    _____________________
    Inputs:
      
        X_train: Unscaled Inputs to the GP model -- torch tensor  (N X ndim)
        Y_train: Unscaled Output Mapping to the GP Model -- torch tensor (N X 1)
        bounds: Bounds of the problem -- torch tensor (2 X ndim)       
    
    """
    
    def __init__(self, param_path, X_path, Y_path):     
        
        # Load JSON from a file
        with open(param_path, "r") as f:
            data = json.load(f)
        
        self.x = torch.tensor(np.load(X_path)).to(torch.float64)
        self.y = torch.tensor(np.load(Y_path))
        self.dim = self.x.shape[1]
        
        self.l = torch.tensor([[data['.kernel.lengthscales']]*self.dim]).to(torch.float64)
        self.sigma_f = torch.tensor(data['.kernel.variance']).to(torch.float64)**0.5
        self.jitter = torch.tensor([data['.likelihood.variance']]).to(torch.float64)
        self.alpha = self.MeanComp()
        
            
    def MeanComp(self): # We get the components of the mean function
       
        self.K = RBF_Matrix(self.x/self.l, 1, self.sigma_f, self.jitter) # calculate the covariance matrix   
        self.K_inv = torch.inverse(self.K)
       
        alpha = torch.matmul(self.K_inv, self.y) # calculate the alpha vector
        return alpha
    
    def update(self, X_new, Y_new): # Update data for new samples
    
        # Unscaled
        self.X = torch.cat((self.X,X_new))
        self.Y = torch.cat((self.Y,Y_new))

        # Scaled values        
        self.x = self.MaxMinScale()
        self.y = self.StandardScale()
        
        # Retrieve hyper parameters for first iteration
        self.model, self.l, self.sigma_f, self.jitter = self.train()        
        self.alpha = self.MeanComp()            
        
    def posterior_mean(self, x):
        distance = np.linalg.norm(self.x/self.l - x/self.l, axis=1)
        mean = np.sum(np.array(self.alpha.flatten())*self.sigma_f.numpy()**2*np.exp(-0.5*distance**2))
        return mean
        
        
    