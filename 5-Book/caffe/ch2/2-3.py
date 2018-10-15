import numpy as np

# bornulli distribution
def bornulli(p):
	return 1 if np.random.rand() > p else 0

# gaussian distribution
def gaussian(mu, std):
	return np.random.normal(mu, std)
