import numpy as np

x = np.array([[0.2, 0.3, 0.6], [0.5, 0.7, 0.1], [0.3, 0.4, 1], [0.3, 0.3, 0.3]])
y = np.array([1, 1, 0, 1])

w1 = np.array([[1, 1, 1],[1, 1, 1], [1, 1, 1], [1, 1, 1]])
b1 = np.array([[-0.8, -1.0, -1.2, -1.6]])

z1 = np.dot(w1, x.T) + b1.T
print('z1=' + str(z1))

x2 = z1
x2[x2 < 0] = 0
print('x2=' + str(x2))

w2 = np.dot(y, np.linalg.inv(x2.T))

z2 = np.dot(w2, x2.T)
print('z2=' + str(z2))
