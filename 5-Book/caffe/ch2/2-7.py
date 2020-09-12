import numpy as np
import sklearn.datasets as d
import matplotlib.pyplot as plt
# creating simple regression datasets
reg_data = d.make_regression(100, 1, 1, 1, 1.0)
plt.plot(reg_data[0], reg_data[1])
plt.show()

# creating simple classification datasets
cls_data = d.make_classification(100, 2, 2, 0, 0, 2)
print(len(cls_data))
print(cls_data[1])
cls0_data = cls_data[0][cls_data[1] == 0]
cls1_data = cls_data[0][cls_data[1] == 1]

plt.scatter(cls0_data[:,0], cls0_data[:,1], c='green')
plt.scatter(cls1_data[:,0], cls1_data[:,1], c='red')
plt.show()

