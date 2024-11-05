# Usage #

This repository implements the Attraction-Repulsion Swarming (ARS) t-SNE variant from [1]. This code is adapted from [Barnes-Hut t-SNE](https://github.com/lvdmaaten/bhtsne/tree/master). Please see that repository for instructions on how to compile the C++ code. Example usage to run the code from this repository in python is given below.

```
import arstsne
import numpy as np
import matplotlib.pyplot as plt 
import graphlearning as gl

data,labels = gl.datasets.load('mnist')
x = arstsne.run_ars_tsne(data, perplexity=30, max_iter=1000, time_step=1, theta1=2, theta2=3, alpha=10, num_early=250)
plt.scatter(x[:,0],x[:,1],s=1,c=labels)
plt.show()
```

The script `experiments.py` runs a number of experiments with ARS t-SNE with different parameter values, as reported in the paper. All results are stored in the `results/` folder. 

# GraphLearning Implementation #

The code is also now available in the [GraphLearning](https://github.com/jwcalder/GraphLearning) Python package, which does not require the user to compile C code. Example usage is below.

```
import graphlearning as gl 
import numpy as np
import matplotlib.pyplot as plt

#Load the MNIST data
data,labels = gl.datasets.load('mnist')

#In order to run the code more quickly, 
#you may want to subsample MNIST. 
size = 70000
if size < data.shape[0]: #If less than 70000
    ind = np.random.choice(data.shape[0], size=size, replace=False)
    data = data[ind,:]
    labels = labels[ind]

#Run ARS t-SNE and plot the result
Y = gl.graph.ars(data, prog=True)
plt.scatter(Y[:,0],Y[:,1],c=labels,s=1)
plt.show()
```

# References #

[1] J. Lu, J. Calder. [Attraction-Repulsion Swarming: A Generalized Framework of t-SNE via Force Normalization and Tunable Interactions](https://arxiv.org/abs), Submitted, 2024.

