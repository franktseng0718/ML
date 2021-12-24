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

def run_gradient(A, w, b, lr=0.01):
    g = 1
    eps = 1e-2
    count = 0
    while np.sqrt(np.sum(g**2)) > eps and count < 100000:
        #print(count)
        count += 1
        g = A.T@(b-1/(1+np.exp(-A@w)))
        w = w + lr*g
        #print('w={}'.format(w.reshape(1,-1)))
        #print('g={}'.format(g.reshape(1,-1)))

    return w

def run_Newton(A, w, b, lr=0.01):
    g=100
    eps = 1e-2
    count = 0
    while np.sqrt(np.sum(g**2)) > eps and count < 100000:
        count += 1
        N = len(A)
        D = np.zeros((N, N))

        for i in range(N):
            D[i, i] = np.exp(-A[i]@w)/np.power(1+np.exp(-A[i]@w), 2)
        H = A.T@D@A

        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError as error:
            print(str(error))
            print('Hessian matrix non invertible, switch to Gradient descent')
            return run_gradient(A,w,b)
    
    
        g = H_inv@A.T@(b-1/(1+np.exp(-A@w)))
        w = w + lr*g
        #print('w={}'.format(w.reshape(1,-1)))
        #print('g={}'.format(g.reshape(1,-1)))
    return w

def predict(A,w):
    '''
    predict whether is class0 or class1
    :param A: (2N,3) shape matrix
    :param w: (3,1) shape matrix
    :return: (2N,1) shape matrix
    '''
    N = len(A)
    b_predict = np.empty((N,1))
    for i in range(N):
        b_predict[i] = 0 if A[i]@w < 0 else 1

    return b_predict

def confusion_matrix(A, b, b_predict):
    b_concate = np.hstack((b, b_predict))
    TP = FP = FN = TN = 0
    for pair in b_concate:
        if pair[0]==pair[1]==0:
            TP += 1
        elif pair[0]==pair[1]==1:
            TN += 1
        elif pair[0]==1 and pair[1]==0:
            FP += 1
        else:
            FN += 1
    matrix = np.empty((2, 2))
    matrix[0, 0],matrix[0, 1],matrix[1, 0],matrix[1, 1] = TP, FN, FP, TN
    C0_predict=[]
    C1_predict=[]
    for i in range(len(A)):
        if b_predict[i]==0:
            C0_predict.append(A[i, 0:2])
        else:
            C1_predict.append(A[i, 0:2])
    return matrix, np.array(C0_predict), np.array(C1_predict)

def print_confusion_matrix(matrix):
    # print confusion_matrix
    print('Confusion Matrix:')
    print('               Predict cluster 1  Predict cluster 2')
    print('Is cluster 1        {:.0f}               {:.0f}       '.format(matrix[0,0],matrix[0,1]))
    print('Is cluster 2        {:.0f}               {:.0f}       '.format(matrix[1,0],matrix[1,1]))
    print()
    print('Sensitivity (Successfully predict cluster 1): {}'.format(matrix[0, 0]/(matrix[0, 0] + matrix[0, 1])))
    print('Specificity (Successfully predict cluster 2): {}'.format(matrix[1, 1]/(matrix[1, 1] + matrix[1, 0])))

def plot_confusion_matrix(c,TP,FN,FP,TN):
    print('------------------------------------------------------------')
    print()
    print('Confusion Matrix {}:'.format(c))
    print('\t\t\t  Predict number {} Predict not number {}'.format(c, c))
    print('Is number  \t{}\t\t{}\t\t\t\t{}'.format(c,TP,FN))
    print('Isn\'t number {}\t\t{}\t\t\t\t{}'.format(c,FP,TN))
    print()
    print('Sensitivity (Successfully predict number {}    ): {:.5f}'.format(c,TP/(TP+FN)))
    print('Specificity (Successfully predict not number {}): {:.5f}'.format(c,TN/(TN+FP)))
    print()