#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 19:22:30 2025

Closed-form (analytical) bounds of kernel terms over a box, and the importance split.
Everything is in lengthscale-scaled coordinates z = x/l.
"""
import torch


def distance_bounds(bounds, Z):
    """
    Smallest and largest Euclidean distance between each center Z_i and the box [bounds[0], bounds[1]].
    The nearest point is the projection of Z_i onto the box; the farthest point is a vertex,
    chosen coordinatewise as the box edge farther from Z_i.
    """
    Z = Z.to(torch.float64)
    bounds = bounds.to(torch.float64)
    dist_near = torch.norm(Z - torch.clamp(Z, bounds[0], bounds[1]), p=2, dim=1)
    dist_far = torch.norm(torch.maximum((Z-bounds[0]).abs(), (Z-bounds[1]).abs()), p=2, dim=1)
    return dist_near, dist_far


def analytic_bound(alpha, sigma_f, dist_near, dist_far, kernel):
    """
    Sign-aware lower bound of sum_i alpha_i * sigma_f^2 * k(d_i) with d_i in [dist_near, dist_far]. Every
    supported kernel decreases with distance, so a positive term is smallest at the largest distance and a
    negative term at the smallest distance.
    """
    opt_dist = torch.where(alpha > 0, dist_far, dist_near)
    return torch.sum(alpha*(sigma_f**2)*kernel.k(opt_dist))


def get_analytic(bounds, BO_obj):
    """
    Split the kernel terms on a box into unimportant and important ones.

    A term is unimportant if its largest possible magnitude on the box, |alpha_i| * k_i(dist_near),
    is below the threshold rho (BO_obj.hyperparameter); unimportant terms are bounded analytically.

    Returns:
        Sigma1    : analytical lower bound of the unimportant terms
        index2    : indices of the important terms (bounded by the PWL MIQCP)
        dist_near : smallest distance between each center and the box
        dist_far  : largest distance between each center and the box
    """
    alpha = BO_obj.alpha.flatten()
    dist_near, dist_far = distance_bounds(bounds, BO_obj.X)

    importance = abs(alpha*BO_obj.sigma_f**2*BO_obj.kernel.k(dist_near))
    unimportant = importance < BO_obj.hyperparameter
    index1 = unimportant.nonzero().squeeze(-1)
    index2 = (~unimportant).nonzero().squeeze(-1)

    Sigma1 = analytic_bound(alpha[index1], BO_obj.sigma_f, dist_near[index1], dist_far[index1], BO_obj.kernel)
    return Sigma1, index2, dist_near, dist_far


def get_analytic_all(bounds, BO_obj):
    """Analytical lower bound with every kernel term treated as unimportant (cheap, valid but loose)."""
    dist_near, dist_far = distance_bounds(bounds, BO_obj.X)
    return analytic_bound(BO_obj.alpha.flatten(), BO_obj.sigma_f, dist_near, dist_far, BO_obj.kernel)
