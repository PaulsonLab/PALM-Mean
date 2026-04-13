#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 19:22:30 2025

@author: tang.1856
"""
import torch
import sys

def active_corners(theta_min,theta_max):
    """
    This code is mainly used to generate all corners of box constraints.
    Will be incorporated in the Object that will give us the bounds.

    inputs:
    theta_min -- N dimensional tensor
    theta_max -- N dimensional tensor

    output -- 2^(N) X N dimensional tensor
    """
    size_t1 = torch.Tensor.size(theta_min)
    size_t2 = torch.Tensor.size(theta_max)

    # Show error if dimensions dont match:
    if size_t1 != size_t2:
        sys.exit('The dimensions of bounds dont match: Please enter valid inputs')

    val = size_t1[0]
    size_out = 2**(val)
    output = torch.zeros(size_out,val)
    output_iter = torch.zeros(size_out)

    for i in range(val):
        div_size = int(size_out/(2**(i+1)))
        divs = int(size_out/div_size)
        div_count = 0
        for j in range(divs):
            if bool(j%2):
                output_iter[div_count:div_count+div_size] = theta_min[i]*torch.ones(div_size)
            else:
                output_iter[div_count:div_count+div_size] = theta_max[i]*torch.ones(div_size)
            div_count = div_count + div_size
        output[:,i] = output_iter
    return output

def get_analytic(bounds, BO_obj):
    hyperparam = BO_obj.hyperparameter
    sigma_f = BO_obj.sigma_f
    alpha = BO_obj.alpha
    X = BO_obj.X
    alpha = alpha.flatten()   
    X2 = X.to(torch.float)
   
    corners = active_corners(bounds[0],bounds[1]) # get the vertex of the spatial hypercube
    dist_center_corner = torch.cdist(corners, X2,p=2).max(0).values.to(torch.float64) # maximum distance between corners and each center point
    
    # -----------------------------Below calculate the max(abs(alpha*k)) for each center-------------------------------------------------------------------------------------------------------------------------------------------------
    # Find the projection point of each center point on the box
    X_project = torch.min(X2,bounds[1].unsqueeze(0).expand_as(X))
    X_project = torch.max(X_project,bounds[0].unsqueeze(0).expand_as(X))
    
    # Calculate distance between center point and projected point
    dist_center_proj = torch.norm(X2-X_project,p=2,dim=1)     
    dist1_new = dist_center_proj.flatten().to(torch.float64)
    
    kernel_val_at_proj = abs(alpha*sigma_f**2*torch.exp(-0.5 * dist1_new**2))
    bool_index1_new = kernel_val_at_proj<hyperparam
    bool_index2_new = ~bool_index1_new
    
    index1 = (bool_index1_new.nonzero()).squeeze(-1) # kernel satisfying the heuristic rule
    index2 = (bool_index2_new.nonzero()).squeeze(-1) # kernel doesn't satisfy the heuristic rule
    
    if len(index1)>0:
        
        alpha1 = (alpha[index1]).squeeze(-1) # alpha term for analytical lower bounding group     
                       
        loop_val = len(index1)
        opt_dist = torch.empty(loop_val)
        
        # --------------------------Below calculate the analytical bound----------------------------------------------------------------------------------------
        dist_max_vals = dist_center_corner[index1]
        dist_negative_alpha = (dist_center_proj[index1]).to(torch.float64)
               
        opt_dist = torch.zeros_like(dist_max_vals)
           
        opt_dist[alpha1>0] = dist_max_vals[alpha1>0] # for positive alpha, the minimum alpha*k is at the largest distance between the center point and the corner
        opt_dist[alpha1<0] = dist_negative_alpha[alpha1<0] # for negative alpha, the minimum alpha*k is at the projection point
           
        alpha1K1 = alpha1*(sigma_f**2) * torch.exp(-0.5 * opt_dist.pow(2))
        Sigma1 = torch.sum(alpha1K1) # This is the analytical bound
    else:
        Sigma1 = torch.tensor(0)
 
    return Sigma1, index2, dist_center_proj, dist_center_corner

    