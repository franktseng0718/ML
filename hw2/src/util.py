import numpy as np
import math
def get_prior(train_y):
    print(train_y)
    p = np.zeros(10)
    for i in range(10):
        filter = (train_y==i)
        filter_train_y = train_y[filter]
        p[i] = len(filter_train_y) / len(train_y)
    return p

def print_imagination_number(pixvalueprob, threshold):
    print('Imagination of numbers in Bayesian classifier:')
    for c in range(10):
        print('{}:'.format(c))
        for i in range(28):
            for j in range(28):
                print('1' if np.argmax(pixvalueprob[c,i*28+j])>=threshold else '0',end=' ')
            print()
        print()
    print()

def get_mean(A):
    return np.mean(A)

def get_variance(array,eps_var):
    var=np.var(array)
    return var if var > eps_var else eps_var #avoid Gaussian formula divided by zero

def gaussain_prob(x,mu,var):
    return ((1/math.sqrt(2*math.pi*var))*math.exp((-(x-mu)**2)/(2*var)))