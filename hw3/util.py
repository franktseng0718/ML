import random
import math
import numpy as np

def gaussian_data_generator(m, s):
    U=random.random()
    V=random.random()
    z=math.sqrt(-2*math.log(U))*math.cos(2*math.pi*V)
    sample=z * (s**0.5) + m
    return sample

def polynomial_basis_generator(n, a, w):
    x0 = random.uniform(-1,1)
    x = [math.pow(x0,i) for i in range(len(w))]
    y = np.sum(w*x)
    e = gaussian_data_generator(0,a)
    return x0, y+e