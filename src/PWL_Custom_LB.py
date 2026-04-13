#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 10:06:46 2023

@author: kudva.7
"""

import gurobipy as gp
import torch


def check_nan(interval_x, dsq_intercept):
    dsq_intercept_median = 0.5*(interval_x[0,:]+interval_x[1,:]) # intermediate point for the interval considered
    # nan_indices = torch.isnan(dsq_intercept) | torch.isinf(dsq_intercept)
    check_if_distance_increase = dsq_intercept <= interval_x[0,:]
    check_if_distance_decrease = dsq_intercept >= interval_x[1,:]
    nan_indices = torch.isnan(dsq_intercept) | torch.isinf(dsq_intercept) | check_if_distance_increase | check_if_distance_decrease
    nan_indices = torch.where(nan_indices)[0]
    replacement_value = dsq_intercept_median[nan_indices]
    
    return nan_indices, replacement_value

# Please refer to https://www.gurobi.com/documentation/9.5/examples/piecewise_py.html
# for how to define PWL objective function

def PWL_LB(xL, xU, dsqL, dsqU, BO_obj, index2, part_neg = 5, part_pos = 5, Thread=3, gap=0.01): # 4 & 3
    # num of segment line = part_neg
    # num of segment line = (part_pos)*2
    
    l_xspace = BO_obj.l[0]
    sigma = BO_obj.sigma_f
    alpha = BO_obj.alpha[index2]
    X = BO_obj.X[index2]
    
    xL = xL*l_xspace
    xU = xU*l_xspace
    X = X*l_xspace

    dsqL = dsqL[index2]**2 # shortest distance between kernel center and spatial box # operate in d^2 space
    dsqU = dsqU[index2]**2 # largest distance between kernel center and spatial box # operate in d^2 space
    N,dim = X.size() 
    
    # Split the values into types
    pos_alpha_index = (alpha > 0).squeeze(-1) # kernel with positive alpha 
    neg_alpha_index = torch.logical_not(pos_alpha_index) # kernel with negative alpha
    
    # Seperate alphas and x
    alpha_pos = alpha[pos_alpha_index].squeeze(-1)
    alpha_neg = alpha[neg_alpha_index].squeeze(-1)
    
    x_BB_pos = X[pos_alpha_index] # center points with positive alpha
    x_BB_neg = X[neg_alpha_index] # center points with negative alpha
        
    # Lower and upper bounds of L
    dsqL_pos = dsqL[pos_alpha_index]
    dsqU_pos = dsqU[pos_alpha_index]
    
    dsqL_neg = dsqL[neg_alpha_index]
    dsqU_neg = dsqU[neg_alpha_index]   
    
    # Number of decision vars
    Npos = len(dsqL_pos)
    Nneg = len(dsqL_neg)
        
    # Define decision variables
    m = gp.Model()
    m.setParam('OutputFlag', 0)
    m.Params.LogToConsole = 0 # Keep gurobi quiet     
        
    x = m.addVars(dim, lb = xL, ub = xU, vtype= gp.GRB.CONTINUOUS, name="x") # the consensus variable
   
    # First deal with the negative kernel that dU>4 and dL<2
    if Nneg>0:
        dneg1 = m.addVars(Nneg, lb = dsqL_neg, ub = dsqU_neg, vtype= gp.GRB.CONTINUOUS, name="dneg1") # define variable
        # linspace like torch 
        dsqLU_neg1 = dsqL_neg
        dd = (dsqU_neg - dsqL_neg)/part_neg

        # Get a torch tensor of d
        temp_val = dsqL_neg + dd
        
        for i in range(part_neg-1):
            dsqLU_neg1 = torch.vstack((dsqLU_neg1,temp_val))
            temp_val += dd
            
        dsqLU_neg1 = torch.vstack((dsqLU_neg1,dsqU_neg))
        neg_alpha_LU1 = alpha_neg*(sigma**2)*torch.exp(-dsqLU_neg1/(2))
        
        for i in range(Nneg): # Negative objective function
            m.setPWLObj(dneg1[i], dsqLU_neg1[:,i], neg_alpha_LU1[:,i])
            m.addConstr((gp.quicksum((x[k]/l_xspace[k] - x_BB_neg[i,k]/l_xspace[k])**2 for k in range(dim))) >=  dneg1[i])
            m.addConstr((gp.quicksum((x[k]/l_xspace[k] - x_BB_neg[i,k]/l_xspace[k])**2 for k in range(dim))) <=  dneg1[i])
 
    ########################### Positive alphas####################################

    pos_alpha_LU = alpha_pos*(sigma**2)*torch.exp(-dsqL_pos/(2))
    if Npos>0:
        step = (dsqU_pos-dsqL_pos)/part_pos
        dsqLU_pos = dsqL_pos
        dpos = m.addVars(Npos,lb = dsqL_pos,ub = dsqU_pos, vtype= gp.GRB.CONTINUOUS,name="dpos") #square of idstance for positive alpha term
        
        i = -1
        for i in range(part_pos-1):
            interval_x = torch.vstack((dsqL_pos+step*i, dsqL_pos+step*(i+1)))
            interval_y = alpha_pos*(sigma**2)*torch.exp(-interval_x/(2))
            slope = alpha_pos*(sigma**2)*torch.exp(-interval_x/(2))*(-0.5)
            c = slope*(-interval_x) + interval_y
                       
            dsq_intercept = 1.0*(c[1] - c[0])/(slope[0] - slope[1]) # intercept point x for the interval considered

            nan_indices, replacement_value = check_nan(interval_x, dsq_intercept)
            dsq_intercept[nan_indices] = replacement_value.double() # if slope of two neighbor point is similar, assume intercept equals to intermidiate point
            
            y_intercept = slope[0]*dsq_intercept + c[0] # intercept y between two points on the true function
            dsqLU_pos = torch.vstack((dsqLU_pos,dsq_intercept))
            pos_alpha_LU = torch.vstack((pos_alpha_LU,y_intercept))
            
            dsqLU_pos = torch.vstack((dsqLU_pos,dsqL_pos+step*(i+1))) # next point on the true function
            intermediate_y = alpha_pos*(sigma**2)*torch.exp(-(dsqL_pos+step*(i+1))/(2))
            pos_alpha_LU = torch.vstack((pos_alpha_LU,intermediate_y))
            
        interval_x = torch.vstack((dsqL_pos+step*(i+1), dsqU_pos))
        interval_y = alpha_pos*(sigma**2)*torch.exp(-interval_x/(2))
        slope = alpha_pos*(sigma**2)*torch.exp(-interval_x/(2))*(-0.5)
        c = slope*(-interval_x) + interval_y
        
        dsq_intercept = 1.0*(c[1] - c[0])/(slope[0] - slope[1]) # intercept point for the interval considered       
        
        nan_indices, replacement_value = check_nan(interval_x, dsq_intercept)
        dsq_intercept[nan_indices] = replacement_value.double() # if slope of two neighbor point is similar, assume intercept equals to intermidiate point
        
        y_intercept = slope[0]*dsq_intercept + c[0] # intercept between two points on the true function
        dsqLU_pos = torch.vstack((dsqLU_pos,dsq_intercept))
        pos_alpha_LU = torch.vstack((pos_alpha_LU,y_intercept))
        
        dsqLU_pos = torch.vstack((dsqLU_pos,dsqU_pos)) # next point on the true function
        intermediate_y = alpha_pos*(sigma**2)*torch.exp(-(dsqU_pos)/(2))
        pos_alpha_LU = torch.vstack((pos_alpha_LU,intermediate_y))
        
        for i in range(Npos): # positive objective function
            m.setPWLObj(dpos[i],dsqLU_pos[:,i],pos_alpha_LU[:,i])
            m.addConstr((gp.quicksum((x[k]/l_xspace[k] - x_BB_pos[i,k]/l_xspace[k])**2 for k in range(dim))) >=  dpos[i])
            m.addConstr((gp.quicksum((x[k]/l_xspace[k] - x_BB_pos[i,k]/l_xspace[k])**2 for k in range(dim))) <=  dpos[i])

    
    m.setParam('Threads',Thread)
    m.setParam('MIPGap',0.001)
    m.setParam('MIPGapAbs',gap)
    m.setParam('TimeLimit',10)

    m.optimize()   
    sol = torch.tensor(m.ObjBound) # get the lower bound (LB of LB is gauranteed to be a proper LB)
    
    return sol

    
    
    




