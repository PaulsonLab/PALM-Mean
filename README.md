# PALM-Mean: An Efficient Spatial Branch-and-Bound Algorithm for Global Optimization of Gaussian Process Posterior Mean Functions

This repository contains the code to reproduce the PALM-Mean algorithm proposed in the paper _An Efficient Spatial Branch-and-Bound Algorithm for Global Optimization of Gaussian Process Posterior Mean Functions_.

# Installation
```sh
pip install -r requirements.txt
```

## Running Experiments

Experiments can be run using the `main.py` script. Users need to specify the path for the .npy and .json files containing GP training data and hyperparameter settings.
Gurobi license is required to solve the lower bounding problem (MIQCP).

Hyperparameter setting for PALM-Mean can be specified in main.py.

We currently only support squared exponential kernel function. Support of Matern class kernel function will be released in future.

**Basic Command**
```
python main.py
```
