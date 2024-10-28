# Usage #

This repository implements the Attraction-Repulsion Swarming (ARS) t-SNE variant from [??]. This code is adapted from [Barnes-Hut t-SNE](https://github.com/lvdmaaten/bhtsne/tree/master). Please see that repository for instructions on how to compile the C++ code. Example usage from python is given below

`
import arstsne
import numpy as np
import matplotlib.pyplot as plt 
import graphlearning as gl

data,labels = gl.datasets.load('mnist')
x = arstsne.run_ars_tsne(data, perplexity=30, max_iter=1000, time_step=1, theta1=2, theta2=3, alpha=10, num_early=250)
plt.scatter(x[:,0],x[:,1],s=1,c=labels)
plt.show()
`

The script `experiments.py` runs a number of experiments with ARS t-SNE with different parameter values, as reported in the paper. All results are stored in the `results/` folder. 


