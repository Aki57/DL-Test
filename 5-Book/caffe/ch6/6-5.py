import numpy as np

def naive_softmax(x):
	y = np.exp(x)
	return y / np.sum(y)

a = np.random.rand(10)
print a
print naive_softmax(a)

b = np.random.rand(10) * 1000
print b
print naive_softmax(b)

np.exp(709)

def high_level_softmax(x):
	max_val = np.max(x)
	x -= max_val
	return naive_softmax(x)

b = np.random.rand(10) * 1000
print b
print high_level_softmax(b)

def practical_softmax(x):
	max_val = np.max(x)
	x -= max_val
	y = np.exp(x)
	y[y < 1e-20] = 1e-20
	return y / np.sum(y)

def naive_sigmoid_loss(x, t):
	y = 1 / (1 + np.exp(-x))
	return np.sum(t * np.log(y) + (1-t) * np.log(1-y)) / y.shape[0]

a = np.random.rand(10)
b = a > 0.5
print a 
print b
print naive_sigmoid_loss(a,b)

a = np.random.rand(10) * 1000
b = a > 500
print a 
print b
print naive_sigmoid_loss(a,b)

def high_level_sigmoid_loss(x, t):
	first = (t - (x > 0)) * x
	second = np.log(1 + np.exp(x - 2 * x * (x > 0)))
	return -np.sum(first - second) / x.shape[0]

a = np.random.rand(10) * 1000
b = a > 500
print a
print b
print high_level_sigmoid_loss(a,b)


