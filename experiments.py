import numpy as np
import arstsne
import matplotlib.pyplot as plt 
import graphlearning as gl
import os

verbose = False
perplexity = 30
time_step = 1
num_early = 250

subset = False
size = 2500

for (dataset,metric) in [('mnist','raw'),('cifar10','simclr')]:
 data,labels = gl.datasets.load(dataset, metric=metric)
 if subset:
  data = data[labels <= 3,:]
  labels = labels[labels <= 3]
  ind = np.random.choice(data.shape[0],size=size,replace=False)
  data = data[ind,:]
  labels = labels[ind]

 for max_iter in [500,1000,5000]:
  for theta1 in [1.0,2.0,3.0]:
   for theta2 in [1.0,2.0,3.0]:
    for alpha in [1.0,2.0,10.0]:

     print(dataset,metric,max_iter,theta1,theta2,alpha)

     fname = './results/%s_%s_%.1f_%.1f_%.1f_%d'%(dataset,metric,theta1,theta2,alpha,max_iter)
     if not os.path.isfile(fname+'.npy'):
         x = np.load(fname+'.npy')
     else:
         print('Could not find saved results, computing...')
         x = arstsne.run_ars_tsne(data, perplexity=perplexity, verbose=verbose, max_iter=max_iter, time_step=time_step, theta1=theta1, theta2=theta2, alpha=alpha, num_early=num_early)
         np.save(fname+'.npy',x)
     plt.figure()
     plt.scatter(x[:,0],x[:,1],s=1,c=labels)
     plt.savefig(fname+'.png',dpi=300,bbox_inches='tight',pad_inches=0.01)
     plt.close()
