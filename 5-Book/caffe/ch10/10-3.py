import numpy as np
import math

def mutual_info(x_var, y_var):
    sum = 0.0
    x_set = set(x_var)
    y_set = set(y_var)
    for x_val in x_set:
        px = float(np.sum(x_var == x_val)) / x_var.size
        x_idx = np.where(x_var == x_val)[0]
        for y_val in y_set:
            py = float(np.sum(y_var == y_val)) / y_var.size
            y_idx = np.where(y_var == y_val)[0]
            pxy = float(np.intersect1d(x_idx, y_idx).size) / x_var.size
            if pxy > 0.0:
                sum += pxy * math.log((pxy / (px * py)), 10)
    return sum

a = np.array([0,0,5,6,0,4,4,3,1,2])
b = np.array([3,4,5,5,3,7,7,6,5,1])
print mutual_info(a,b)
# 0.653521

a = np.array([0,0,5,6,0,4,4,3,1,2])
b = np.array([3,3,5,6,3,7,7,9,4,8])
print mutual_info(a,b)
# 0.796658