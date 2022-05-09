import numpy as np
from numpy.linalg import inv, det, cholesky
import csv
def load_data(path='data/input.data'):
    x=[]
    y=[]
    f = open(path, 'r')
    for line in f.readlines():
        datapoint = line.split(' ')
        x.append(float(datapoint[0]))
        y.append(float(datapoint[1]))
    f.close()
    x=np.asarray(x)
    y=np.asarray(y)
    return x,y

def kernel(X1,X2,sigma=1, alpha=1,length_scale=1):
    '''
    using rational quadratic kernel function: k(x_i, x_j) = sigma^2 *(1 + (x_i-x_j)^2 / (2*alpha * length_scale^2))^-alpha
    :param X1: (n) ndarray
    :param X2: (m) ndarray
    return: (n,m)  ndarray
    '''
    square_error = np.power(X1.reshape(-1,1)-X2.reshape(1,-1),2.0)
    # print(square_error)
    kernel = sigma**2 * np.power(1+square_error/(2*alpha*length_scale**2),-alpha)

    return kernel

def predict(x_line,X,y,K,beta,alpha=1,length_scale=1):
    '''
    vectorize calculate k_x_xstar !!
    :param x_line: sampling in linspace(-60,60)
    :param X:  (n) ndarray
    :param y: (n) ndarray
    :param K: (n,n) ndarray
    :param beta:
    :return: (len(x_line),1) ndarray, (len(x_line),len(x_line)) ndarray
    '''
    k_x_xstar = kernel(X,x_line,alpha=1,length_scale=1)
    k_xstar_xstar = kernel(x_line,x_line,alpha=1,length_scale=1)
    means = k_x_xstar.T @ np.linalg.pinv(K) @ y.reshape(-1,1)
    vars = k_xstar_xstar + (1/beta)*np.identity(len(k_xstar_xstar))\
        - k_x_xstar.T @ np.linalg.pinv(K) @ k_x_xstar

    return means,vars

def NegativeLogLikelihood(theta, X, Y, beta):
    theta = theta.ravel()
    K = kernel(X, X, theta[0], theta[1], theta[2])
    K += np.identity(len(X), dtype=np.float64) * (1 / beta)
    nll = np.sum(np.log(np.diagonal(cholesky(K))))
    nll += 0.5 * Y.T @ inv(K) @ Y
    nll += 0.5 * len(X) * np.log(2 * np.pi)
    return nll

def openCSV(filename):
    with open(filename, 'r') as file:
        content = list(csv.reader(file))
        content = np.array(content)
    return content



