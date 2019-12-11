# coding: utf-8
import numpy as np
from matplotlib import pyplot as plt
 
# 高斯分布
mean = [0,0]
cov = [[0,1],[1,0]]
x, y = np.random.multivariate_normal(mean, cov, 10000).T

hist, xedges, yedges = np.histogram2d(x,y)
X,Y = np.meshgrid(xedges,yedges)
plt.imshow(hist, interpolation='nearest')
plt.grid(True)
plt.colorbar()
plt.show()

'''
plt.hist2d(x, y, bins=200)
plt.colorbar()
plt.grid()
plt.show()
'''

