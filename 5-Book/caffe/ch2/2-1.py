import numpy as np

# normal matrix-vector multiplication 
A = np.array([[1,0,0],[0,1,0],[1,1,0]])
b = np.array([1,1,1])
print np.dot(A.T, b)

# dimension-reduced multiplication
A2 = np.array([[1,0,0],[0,1,0]])
b2 = np.array([2,2,0])
print np.dot(A2, b2)


