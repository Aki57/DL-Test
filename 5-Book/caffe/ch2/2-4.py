import numpy as np
import matplotlib.pyplot as plt
# entropy of bornulli distribution
x = np.linspace(0.01,0.99, 101)
y = -x * np.log2(x) - (1-x) * np.log2(1-x)
plt.plot(x,y)
plt.show()
