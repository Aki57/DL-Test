#%%
import matplotlib.pyplot as plt
import numpy as np

#%%
def gd(x_start, step, g):
    x = np.array(x_start, dtype='float64')
    for i in range(100):
        grad = g(x)
        x -= grad * step
        print('[ Epoch {0}] grad = {1}, x = {2}'.format(i, grad, x))
        if abs(grad).any() < 1e-6:
            break
    return x

#%%
def momentum(x_start, step, g, discount = 0.7):
    x = np.array(x_start, dtype='float64')
    pre_grad = np.zeros_like(x)
    for i in range(100):
        grad = g(x)
        pre_grad = pre_grad*discount + grad*step
        x -= pre_grad
        print('[ Epoch {0}] grad = {1}, x = {2}'.format(i, grad, x))
        if abs(sum(grad)) < 1e-6:
            break
    return x

#%%
def f(x):
    return x[0] * x[0] + 50 * x[1] * x[1]

def g(x):
    return np.array([2 * x[0], 100 * x[1]])

xi = np.linspace(-200, 200, 1000)
yi = np.linspace(-100, 100, 1000)
X, Y = np.meshgrid(xi, yi)
Z = X * X + 50 * Y * Y

res, x_arr = gd([150, 75], 0.015, g)
plt.contour(X,Y,Z,x_arr)
plt.show()
