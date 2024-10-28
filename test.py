import arstsne
import numpy as np
import matplotlib.pyplot as plt 
import graphlearning as gl

data,labels = gl.datasets.load('mnist')
x = arstsne.run_ars_tsne(data, perplexity=30, max_iter=1000, time_step=1, theta1=2, theta2=3, alpha=10, num_early=250)
plt.scatter(x[:,0],x[:,1],s=1,c=labels)
plt.show()
