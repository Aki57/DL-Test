import numpy as np
import matplotlib.pyplot as plt


def gd(x_start, step, g):
    x = x_start
    xarr = [x]
    for i in range(20):
        grad = g(x)
        x -= grad * step
        xarr.append(x.copy())
        print '[ Epoch {0} ] grad = {1}, x = {2}'.format(i, grad, x)
        if sum(abs(grad)) < 1e-6:
            break;
    return x, xarr

def momentum(x_start, step, g, discount = 0.7):
    x = np.array(x_start, dtype='float64')
    pre_grad = np.zeros_like(x)
    xarr = [x.copy()]
    for i in range(50):
        grad = g(x)
        pre_grad = pre_grad * discount + grad * step
        x -= pre_grad
        xarr.append(x.copy())

        print '[ Epoch {0} ] grad = {1}, x = {2}'.format(i, grad, x)
        if sum(abs(grad)) < 1e-6:
            break;
    return x, xarr

def f(x):
    return (x[0] - 2) * x[0] + 100 * x[0] * x[1] + 10 * (x[1] + 2) * x[1]

def g(x):
    return np.array([2 * x[0] - 2 + 100 * x[1], 100 * x[0] + 20 * x[1] + 20])

xi = np.linspace(-20, 20, 1000)
yi = np.linspace(-10, 10, 1000)
X, Y = np.meshgrid(xi, yi)
Z = (X - 2) * X + 1000 * X * Y + 10 * (Y + 2) * Y

def contour(X, Y, Z, arr = None, title='Trace'):
    plt.figure(figsize=(15,7))
    xx = X.flatten()
    yy = Y.flatten()
    zz = Z.flatten()
    plt.contour(X, Y, Z, colors='black')
    if arr is not None:
        arr = np.array(arr)
        for i in range(len(arr) - 1):
            plt.plot(arr[i:i+2,0],arr[i:i+2,1])
    plt.title(title)
    plt.show()

contour(X, Y, Z)

# gradient descent
res, x_arr = gd([5,5], 0.0008, g)
contour(X,Y,Z, x_arr, 'gradient descent')

# momentum
res, x_arr = momentum([5, 5], 3e-4, g)
contour(X,Y,Z, x_arr, 'momentum')

def adagrad(x_start, step, g, delta=1e-8):
    x = np.array(x_start, dtype='float64')
    sum_grad = np.zeros_like(x)
    x_arr = [x.copy()]
    for i in range(50):
        grad = g(x)
        sum_grad += grad * grad
        x -= step * grad / (np.sqrt(sum_grad) + delta)
        x_arr.append(x.copy())
        if abs(sum(grad)) < 1e-6:
            break;
    return x, x_arr

res, x_arr = adagrad([5, 5], 1.3, g)
contour(X,Y,Z, x_arr, 'adagrad')

def rmsprop(x_start, step, g, rms_decay = 0.9, delta=1e-8):
    x = np.array(x_start, dtype='float64')
    sum_grad = np.zeros_like(x)
    passing_dot = [x.copy()]
    for i in range(50):
        grad = g(x)
        sum_grad = rms_decay * sum_grad + (1 - rms_decay) * grad * grad
        x -= step * grad / (np.sqrt(sum_grad) + delta)
        passing_dot.append(x.copy())        
        if abs(sum(grad)) < 1e-6:
            break;
    return x, passing_dot

res, x_arr = rmsprop([5, 5], 0.3, g)
contour(X,Y,Z, x_arr, 'rmsprop')

def adadelta(x_start, step, g, momentum = 0.9, delta=1e-1):
    x = np.array(x_start, dtype='float64')
    sum_grad = np.zeros_like(x)
    sum_diff = np.zeros_like(x)
    passing_dot = [x.copy()]
    for i in range(50):
        grad = g(x)
        sum_grad = momentum * sum_grad + (1 - momentum) * grad * grad
        diff = np.sqrt((sum_diff + delta) / (sum_grad + delta)) * grad
        x -= step * diff
        sum_diff = momentum * sum_diff + (1 - momentum) * (diff * diff)
        passing_dot.append(x.copy())
        if abs(sum(grad)) < 1e-6:
            break;
    return x, passing_dot

res, x_arr = adadelta([5, 5], 0.4, g)
contour(X,Y,Z, x_arr,'adadelta')

def adam(x_start, step, g, beta1 = 0.9, beta2 = 0.999,delta=1e-8):
    x = np.array(x_start, dtype='float64')
    sum_m = np.zeros_like(x)
    sum_v = np.zeros_like(x)
    passing_dot = [x.copy()]
    for i in range(1, 50):
        grad = g(x)
        sum_m = beta1 * sum_m + (1 - beta1) * grad
        sum_v = beta2 * sum_v + (1 - beta2) * grad * grad
        correction = np.sqrt(1 - beta2 ** i) / (1 - beta1 ** i)
        x -= step * correction * sum_m / (np.sqrt(sum_v) + delta)
        passing_dot.append(x.copy())
        if abs(sum(grad)) < 1e-6:
            break;
    return x, passing_dot

res, x_arr = adam([5, 5], 0.5, g)
contour(X,Y,Z, x_arr, 'adam')

res, x_arr = gd([-0.23,0.0], 0.0008, g)
contour(X,Y,Z, x_arr, 'gradient descent')

res, x_arr = momentum([-0.23,0], 5e-4, g)
contour(X,Y,Z, x_arr, 'momentum')

res, x_arr = adagrad([-0.23, 0], 1.3, g)
contour(X,Y,Z, x_arr, 'adagrad')

res, x_arr = rmsprop([-0.23, 0], 0.3, g)
contour(X,Y,Z, x_arr, 'rmsprop')

res, x_arr = adadelta([-0.23, 0], 0.4, g)
contour(X,Y,Z, x_arr, 'adadelta')

res, x_arr = adam([-0.23, 0], 0.1, g)
contour(X,Y,Z, x_arr, 'adam')


