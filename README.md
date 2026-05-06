# PALM-Mean: An Efficient Spatial Branch-and-Bound Algorithm for Global Optimization of Gaussian Process Posterior Mean Functions

This repository contains the code to reproduce the PALM-Mean algorithm proposed in the paper _An Efficient Spatial Branch-and-Bound Algorithm for Global Optimization of Gaussian Process Posterior Mean Functions_.

# Installation
```sh
pip install -r requirements.txt
```

## Running Experiments

*    Experiments can be run using the `main.py` script. Users need to specify the path for the .npy and .json files containing Gaussian processes (GPs) training data and hyperparameter settings.
Gurobi license is required to solve the lower bounding problem (MIQCP).

*    Hyperparameter setting for PALM-Mean can be specified in [main.py](https://github.com/PaulsonLab/PALM-Mean/blob/b07e0ff4c91f2701f775884f85484fdcfc14f6e0/main.py).

*    We currently only support GP posterior mean with squared exponential kernel function. Support of Matern class kernel function will be released in future.

**Basic Command**
```
python main.py
```
