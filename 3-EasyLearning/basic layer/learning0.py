#%%
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf

def bornulli(p):
    return 1 if np.random.rand() > p else 0

def gaussian(mu, std):
    return np.random.normal(mu, std)

x = np.linspace(0,1,101)
y = -x*np.log2(x)-(1-x)*np.log2(1-x)
y[np.isnan(y)]=0
plt.plot(x,y)

fig = plt.figure()
ax = Axes3D(fig)
x = np.linspace(0.1,2,31)
y = np.linspace(-2,2,31)
X,Y = np.meshgrid(x,y)
Z = -np.log(X)+X*X+Y*Y/2-0.5
ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap='rainbow')

fig = plt.figure()
ax = Axes3D(fig)
X = np.linspace(0.01, 0.99, 101)
Y = np.linspace(0.01, 0.99, 101)
X, Y = np.meshgrid(X, Y)
Z = -X*np.log2(Y) - (1-X)*np.log2(1-Y)
ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap='rainbow')
plt.show()

