import numpy as np
import random
import math
import sys

def subgradient_descent(x, grad, lr, rate):
    x_new = x -  lr * (grad + rate * np.sign(x))
    return x_new

def genData(n, p, s):
    A = np.random.normal(0, 1, (n, p))
    opt_x = np.zeros(p)
    random_index_list = random.sample(range(p), s)
    for i in random_index_list:
        opt_x[i] = np.random.normal(0, 10)
    e = np.random.normal(0, 1, n)      
    b = np.dot(A, opt_x.T) + e.T
    return A, b

A, b = genData(100, 50, 20)
lmd = math.sqrt(2 * 100 * math.log(50))
print(A)
print(b)

def obj_func(A, x, b, lmd):
    f_value = math.pow(np.linalg.norm(np.dot(A,x.T) - b),2) / 2
    g_value = lmd * np.linalg.norm(x,1)
    return f_value + g_value
    
def subgradient_process(A, b, lmd):
    x = np.zeros(A.shape[1])
    step_length = 0.1
    while(True):
        obj_value = obj_func(A, x, b, lmd)
        while(True):
            prime_grad = np.dot(A.T, np.dot(A, x.T) - b)
            print(prime_grad)
            x_new = subgradient_descent(x, prime_grad, step_length, lmd)
            obj_value2 = obj_func(A, x_new, b, lmd)
            if obj_value2 <= obj_value:
                break
            else:
                step_length = step_length * 0.5 # BackTrace
            print(step_length, obj_value, obj_value2)
            #print x_new
        if abs(obj_value - obj_value2) < 1e-6:
            break
        else:
            x = x_new
        print('x=' + str(x))
        print('obj_value2=' + str(obj_value2))
    return x

xarr = subgradient_process(A, b, lmd)
print(xarr)

def gradient_descent(x_k, step_length,A,b):
    f_prime = np.dot(A.T, np.dot(A,x_k.T) - b) # derivertive of f_function
    y = x_k - step_length * f_prime
    return y

def prox_operation(lmd,step_length,y):
    new_y = np.sign(y)
    prox_vec = new_y * np.maximum(0, np.absolute(y) - step_length*lmd)   # NOTE: here is step_length * lmd
    return prox_vec

def f_func(A,x,b):
    f_value = math.pow(np.linalg.norm(np.dot(A,x.T) - b),2) / 2
    return f_value

def m_func(A, x_k, b, step_length, x):
    part_1 = f_func(A,x_k,b)
    #part_1 = math.pow(np,linalg.norm(np.dot(A,x_k.T) - b),2) / 2 
    part_2 = np.dot( (np.dot(A.T, np.dot(A,x_k.T) - b)).T, (x-x_k).T )
    part_3 = math.pow(np.linalg.norm(x-x_k),2) / (2*step_length)
    m_value = part_1 + part_2 + part_3
    return m_value

def proximal_process(A, b, lmd):
    x_k = np.array([0.0]*A.shape[1])
    step_length = 10
    while(1):
        obj_value = obj_func(A,x_k,b,lmd)
        while(1):
            grad = gradient_descent(x_k, step_length, A, b)
            x_k_plus_1 = prox_operation(lmd,step_length, grad)
            f_value = f_func(A,x_k_plus_1,b)
            m_value = m_func(A,x_k, b, step_length, x_k_plus_1)
            if f_value <= m_value:
                break
            step_length = step_length * 0.5
        if f_func(A,x_k,b) <= f_func(A,x_k_plus_1,b):
            break
        else:
            x_k =  x_k_plus_1
    return x_k

xarr = proximal_process(A, b, lmd)
print(xarr)
