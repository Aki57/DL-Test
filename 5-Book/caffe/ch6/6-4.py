import numpy as np
import matplotlib.pyplot as plt

x = np.random.randn(200)
y = x * 2
err = np.random.rand(200) * 2
y += err
data = np.vstack((x,y))
plt.scatter(data[0,:], data[1,:])
plt.show()

# minus mean
mean = np.mean(data, axis=1)
data -= mean.reshape((mean.shape[0], 1))
plt.scatter(data[0,:], data[1,:])
plt.show()

# calc co-variance
cov = np.dot(data, data.T) / (data.shape[1] - 1)
print cov

# calc W
eig_val, eig_vec = np.linalg.eig(cov)
S_sqrt = np.sqrt(np.diag(eig_val))
W = np.dot(eig_vec, np.dot(np.linalg.inv(S_sqrt), eig_vec.T))

# calc ZCA transform
Y = np.dot(W, data)
plt.scatter(Y[0,:], Y[1,:])
cov2 = np.dot(Y, Y.T) / (data.shape[1] - 1)
print cov2
plt.show()

