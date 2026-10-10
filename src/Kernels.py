#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stationary kernels supported by PALM-Mean, as functions of the lengthscale-scaled distance
r = ||(x - x')/l|| and without the signal variance sigma_f^2:

    rbf       k(r) = exp(-r^2/2)
    matern12  k(r) = exp(-r)
    matern32  k(r) = (1 + sqrt(3) r) exp(-sqrt(3) r)
    matern52  k(r) = (1 + sqrt(5) r + 5 r^2/3) exp(-sqrt(5) r)

All are decreasing in r. Matern 3/2 and 5/2 are concave for r < r_infl and convex for r > r_infl
(r_infl = 1/sqrt(3) and (5 + sqrt(5))/10); Matern 1/2 is convex everywhere (r_infl = 0). The RBF
kernel is convex in the squared distance d = r^2, which is the variable its PWL relaxation uses.
The functions accept torch tensors or numpy arrays.
"""
import math
import numpy as np
import torch

SQRT3 = math.sqrt(3.0)
SQRT5 = math.sqrt(5.0)


def _exp(r):
    return torch.exp(r) if torch.is_tensor(r) else np.exp(r)


class Kernel():
    """
    name       : kernel name (see KERNELS)
    k(r)       : kernel value at scaled distance r (without sigma_f^2)
    dk(r)      : derivative dk/dr
    inflection : r at which k changes from concave to convex (None for rbf, whose PWL uses d = r^2)
    """

    def __init__(self, name, k, dk, inflection):
        self.name, self.k, self.dk, self.inflection = name, k, dk, inflection


KERNELS = {
    "rbf": Kernel("rbf", lambda r: _exp(-0.5*r**2), lambda r: -r*_exp(-0.5*r**2), None),
    "matern12": Kernel("matern12", lambda r: _exp(-r), lambda r: -_exp(-r), 0.0),
    "matern32": Kernel("matern32", lambda r: (1 + SQRT3*r)*_exp(-SQRT3*r), lambda r: -3*r*_exp(-SQRT3*r), 1/SQRT3),
    "matern52": Kernel("matern52", lambda r: (1 + SQRT5*r + 5*r**2/3)*_exp(-SQRT5*r),
                       lambda r: -(5/3)*r*(1 + SQRT5*r)*_exp(-SQRT5*r), (5 + SQRT5)/10),
}

# Accepted spellings of the '.kernel.type' entry in params.json
ALIASES = {"rbf": "rbf", "se": "rbf", "squaredexponential": "rbf", "squared_exponential": "rbf",
           "matern12": "matern12", "matern1/2": "matern12", "matern32": "matern32", "matern3/2": "matern32",
           "matern52": "matern52", "matern5/2": "matern52"}


def get_kernel(name):
    """Kernel object for a name such as 'rbf', 'Matern32' or 'matern5/2' (case-insensitive)."""
    key = ALIASES.get(str(name).lower().replace(" ", "").replace("-", ""))
    if key is None:
        raise ValueError("Unknown kernel '%s'. Supported: rbf, matern12, matern32, matern52." % name)
    return KERNELS[key]
