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

def faceRecognition(X, X_label, test, test_label, method, kernel_type=None):
    if kernel_type is None:
        print(f'Face recognition with {method} and KNN:')
    else:
        print(f'Face recognition with Kernel {method}({kernels[kernel_type - 1]}) and KNN:')
    dist_mat = []
    for i in range(test.shape[0]):
        dist = []
        for j in range(X.shape[0]):
            dist.append((distance(X[j], test[i]), X_label[j]))
        dist.sort(key=lambda x: x[0])
        dist_mat.append(dist)
    for k in K:
        correct = 0
        total = test.shape[0]
        for i in range(test.shape[0]):
            dist = dist_mat[i]
            neighbor = np.asarray([x[1] for x in dist[:k]])
            neighbor, count = np.unique(neighbor, return_counts=True)
            predict = neighbor[np.argmax(count)]
            if predict == test_label[i]:
                correct += 1
        print(f'K={k:>2}, accuracy: {correct / total:>.3f} ({correct}/{total})')
    print()