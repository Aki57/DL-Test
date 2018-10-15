import numpy as np
import matplotlib.pyplot as plt

def gd(x_start, step, g):   # gd means Gradient Descent
    x = x_start
    for i in range(20):
        grad = g(x)
        x -= grad * step
        print '[ Epoch {0} ] grad = {1}, x = {2}'.format(i, grad, x)
        if abs(grad) < 1e-6:
            break;
    return x

def f(x):
    return x * x - 2 * x + 1

def g(x):
    return 2 * x - 2

x = np.linspace(-5,7,100)
y = f(x)
plt.plot(x, y)

# small step
gd(5,0.1,g)

# large step
gd(5,100,g)

# linger step
gd(5, 1, g)

# linger step 2
gd(4, 1, g)

# another question
def f2(x):
    return 4 * x * x - 4 * x + 1
def g2(x):
    return 8 * x - 4
gd(5,0.25,g2)
