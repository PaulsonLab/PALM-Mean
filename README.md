# PALM-Mean: An Efficient Spatial Branch-and-Bound Algorithm for Global Optimization of Gaussian Process Posterior Mean Functions

This repository contains the code to reproduce the PALM-Mean algorithm proposed in the paper _An Efficient Spatial Branch-and-Bound Algorithm for Global Optimization of Gaussian Process Posterior Mean Functions_.

# Installation
```sh
pip install -r requirements.txt
```

## Running Experiments

Experiments can be run using the `main.py` script. You must specify the path for GP trainin data and hyperparameter settings.
Gurobi license is required to solve the lower bounding problem (MIQCP).

We currently only support squared exponential kernel function. Support of Matern class kernel function will be released in future.

**Basic Command**
```
python main.py
```
