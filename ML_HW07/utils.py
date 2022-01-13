import numpy as np
from PIL import Image
import os, re, sys
import scipy.spatial.distance
from datetime import datetime
import matplotlib.pyplot as plt
SHAPE = (50, 50)
kernels = ['linear kernel', 'polynomial kernel', 'rbf kernel']
K = [1, 3, 5, 7, 9, 11]
def readPGM(filename):
    image = Image.open(filename)
    image = image.resize(SHAPE, Image.ANTIALIAS)
    image = np.array(image)
    label = int(re.findall(r'subject(\d+)', filename)[0])
    return [image.ravel().astype(np.float64), label]

def readData(path):            
    data = []
    filename = []
    label = []
    for pgm in os.listdir(path):
        res = readPGM(f'{path}/{pgm}')
        data.append(res[0])
        filename.append(pgm)
        label.append(res[1])
    return [np.asarray(data), np.asarray(filename), np.asarray(label)]

def draw(target_data, target_filename, title, W, mu=None):
    if mu is None:
        mu = np.zeros(target_data.shape[1])
    projection = (target_data - mu) @ W
    reconstruction = projection @ W.T + mu
    folder = f"{title}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    os.mkdir(folder)
    os.mkdir(f'{folder}/{title}')
    if W.shape[1] == 25:
        plt.clf()
        for i in range(5):
            for j in range(5):
                idx = i * 5 + j
                plt.subplot(5, 5, idx + 1)
                plt.imshow(W[:, idx].reshape(SHAPE[::-1]), cmap='gray')
                plt.axis('off')
        plt.savefig(f'./{folder}/{title}/{title}.png')
    for i in range(W.shape[1]):
        plt.clf()
        plt.title(f'{title}_{i + 1}')
        plt.imshow(W[:, i].reshape(SHAPE[::-1]), cmap='gray')
        plt.savefig(f'./{folder}/{title}/{title}_{i + 1}.png')
    
    if reconstruction.shape[0] == 10:
        plt.clf()
        for i in range(2):
            for j in range(5):
                idx = i * 5 + j
                plt.subplot(2, 5, idx + 1)
                plt.imshow(reconstruction[idx].reshape(SHAPE[::-1]), cmap='gray')
                plt.axis('off')
        plt.savefig(f'./{folder}/reconstruction.png')
    for i in range(reconstruction.shape[0]):
        plt.clf()
        plt.title(target_filename[i])
        plt.imshow(reconstruction[i].reshape(SHAPE[::-1]), cmap='gray')
        plt.savefig(f'./{folder}/{target_filename[i]}.png')

def distance(vec1, vec2):
    return np.sum((vec1 - vec2) ** 2)



def linearKernel(X):
    return X @ X.T

def polynomialKernel(X, gamma, coef, degree):
    return np.power(gamma * (X @ X.T) + coef, degree)

def rbfKernel(X, gamma):
    return np.exp(-gamma * scipy.spatial.distance.cdist(X, X, 'sqeuclidean'))

def getKernel(X, kernel_type):
    if kernel_type == 1:
        kernel = linearKernel(X)
    elif kernel_type == 2:
        kernel = polynomialKernel(X, 5, 10, 2)
    else:
        kernel = rbfKernel(X, 1e-7)
    return kernel

def kernelPCA(X, dims, kernel_type):
    kernel = getKernel(X, kernel_type)
    n = kernel.shape[0]
    one = np.ones((n, n), dtype=np.float64) / n
    kernel = kernel - one @ kernel - kernel @ one + one @ kernel @ one
    eigen_val, eigen_vec = np.linalg.eigh(kernel)
    for i in range(eigen_vec.shape[1]):
        eigen_vec[:, i] = eigen_vec[:, i] / np.linalg.norm(eigen_vec[:, i])
    idx = np.argsort(eigen_val)[::-1]
    eigen_val = np.column_stack(eigen_val[idx])
    W = eigen_vec[:, idx][:, :dims].real
    return kernel @ W

def kernelLDA(X, label, dims, kernel_type):
    label = np.asarray(label)
    c = np.unique(label)
    kernel = getKernel(X, kernel_type)
    n = kernel.shape[0]
    mu = np.mean(kernel, axis=0)
    N = np.zeros((n, n), dtype=np.float64)
    M = np.zeros((n, n), dtype=np.float64)
    for i in c:
        K_i = kernel[np.where(label == i)[0], :]
        l = K_i.shape[0]
        mu_i = np.mean(K_i, axis=0)
        N += K_i.T @ (np.eye(l) - (np.ones((l, l), dtype=np.float64) / l)) @ K_i
        M += l * ((mu_i - mu).T @ (mu_i - mu))
    eigen_val, eigen_vec = np.linalg.eig(np.linalg.pinv(N) @ M)
    for i in range(eigen_vec.shape[1]):
        eigen_vec[:, i] = eigen_vec[:, i] / np.linalg.norm(eigen_vec[:, i])
    idx = np.argsort(eigen_val)[::-1]
    W = eigen_vec[:, idx][:, :dims].real
    return kernel @ W