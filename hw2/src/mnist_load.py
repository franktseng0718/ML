import gzip
import os
import struct
from array import array
import random
import numpy as np

class MNIST(object):
    def __init__(self, path='.', gz=False):
        self.path = path
        self.test_img_fname = 't10k-images-idx3-ubyte'
        self.test_lbl_fname = 't10k-labels-idx1-ubyte'

        self.train_img_fname = 'train-images-idx3-ubyte'
        self.train_lbl_fname = 'train-labels-idx1-ubyte'

        self.gz = gz
        self.emnistRotate = False

        self.test_images = []
        self.test_labels = []

        self.train_images = []
        self.train_labels = []

    def load_testing(self):
        ims, labels = self.load(os.path.join(self.path, self.test_img_fname),
                                os.path.join(self.path, self.test_lbl_fname))
        self.train_images = ims
        self.train_labels = labels
        return self.train_images, self.train_labels

    def load_training(self):
        ims, labels = self.load(os.path.join(self.path, self.train_img_fname),
                                os.path.join(self.path, self.train_lbl_fname))
        self.test_images = ims
        self.test_labels = labels
        return self.test_images, self.test_labels

    def opener(self, path_fn, *args, **kwargs):
        if self.gz:
            return gzip.open(path_fn + '.gz', *args, **kwargs)
        else:
            return open(path_fn, *args, **kwargs)

    def load(self, path_img, path_lbl):
        with self.opener(path_lbl, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049,'
                                 'got {}'.format(magic))

            labels = array("B", file.read())

        with self.opener(path_img, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051,'
                                 'got {}'.format(magic))

            image_data = array("B", file.read())
        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)

        for i in range(size):
            images[i][:] = image_data[i * rows * cols:(i + 1) * rows * cols]

        return images, labels