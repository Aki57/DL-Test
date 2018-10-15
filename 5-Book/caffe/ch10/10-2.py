import numpy as np
import math
def KL(p, q):
    if 0 in q:
        raise ValueError
    return sum(_p * math.log(_p/_q) for (_p,_q) in zip(p, q) if _p != 0)

def JS(p, q):
    M = [0.5 * (_p + _q) for (_p, _q) in zip(p, q)]
    return 0.5 * (KL(p, M) + KL(q, M))

def exp(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    a /= a.sum()
    b /= b.sum()
    print a
    print b
    print KL(a,b)
    print JS(a,b)

# exp 1
exp([1,2,3,4,5],[5,4,3,2,1])
# exp 2
exp([1,2,3,4,5],[1e-12,4,3,2,1])
exp([1,2,3,4,5],[5,4,3,2,1e-12])
