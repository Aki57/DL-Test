import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf

class FC:
    def __init__(self, in_num, out_num, lr =0.01):
        self._in_num = in_num
        self._out_num = out_num
        self.w = np.random.randn(out_num, in_num)*10
        self.b = np.zeros(out_num)
    
    def _sigmoid(self, in_data):
        return 1 / (1 + np.exp(-in_data))

    def forward(self, in_data):
        return self._sigmoid(np.dot(self.w, in_data) + self.b)

x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)
X_f = X.flatten()
Y_f = Y.flatten()
data = zip(X_f, Y_f)

"""
# one layer
fc = FC(2, 1)
Z1 = np.array([fc.forward(d) for d in data])
Z1 = Z1.reshape((100, 100))

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(X,Y,Z1)

# two layers - 1
fc = FC(2, 3)
fc.w = np.array([[0.4, 0.6], [0.3, 0.7], [0.2, 0.8]])
fc.b = np.array([0.5, 0.5, 0.5])

fc2 = FC(3, 1)
fc2.w = np.array([0.3, 0.2, 0.1])
fc2.b = np.array([0.5])

Z1 = np.array([fc.forward(d) for d in data])
Z2 = np.array([fc2.forward(d) for d in Z1])
Z2 = Z2.reshape((100, 100))

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(X,Y,Z2)

# two layers - 2
fc = FC(2, 3)
fc.w = np.array([[-0.4, 0.6], [-0.3, 0.7], [0.2, -0.8]])
fc.b = np.array([-0.5, 0.5, 0.5])

fc2 = FC(3, 1)
fc2.w = np.array([-3, 2, -1])
fc2.b = np.array([0.5])

Z1 = np.array([fc.forward(d) for d in data])
Z2 = np.array([fc2.forward(d) for d in Z1])
Z2 = Z2.reshape((100, 100))

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(X,Y,Z2)

# two layers - 3
fc = FC(2, 3)
fc2 = FC(3, 1)

Z1 = np.array([fc.forward(d) for d in data])
Z2 = np.array([fc2.forward(d) for d in Z1])
Z2 = Z2.reshape((100, 100))

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(X,Y,Z2)
"""

# five layers
fc = FC(2, 10)
fc2 = FC(10, 20)
fc3 = FC(20, 30)
fc4 = FC(30, 50)
fc5 = FC(50, 1)

Z1 = np.array([fc.forward(d) for d in data])
Z2 = np.array([fc2.forward(d) for d in Z1])
Z3 = np.array([fc3.forward(d) for d in Z2])
Z4 = np.array([fc4.forward(d) for d in Z3])
Z5 = np.array([fc5.forward(d) for d in Z4])
Z5 = Z5.reshape((100, 100))

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(X,Y,Z5)

plt.show()
